"""End-to-end Foundation-stage pipeline: generate -> split -> train -> evaluate.

Run directly for a demo/smoke run: `python -m ids_ml.pipeline --out-dir data/generated/demo`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ids_ml.data.generator import GenerationConfig, generate_dataset, save_dataset, split_dataset
from ids_ml.features import (
    SCENARIO_CLASSES,
    build_preprocessor,
    extract_labels,
    extract_scenario_labels,
    transform_features,
)
from ids_ml.models.evaluate import evaluate, evaluate_multiclass
from ids_ml.models.train import TrainConfig, save_artifact, train_model, train_multiclass_model


def run(
    seed: int,
    out_dir: Path | str,
    *,
    num_hosts: int = 60,
    duration_minutes: float = 180.0,
) -> dict:
    out_dir = Path(out_dir)

    gen_config = GenerationConfig(seed=seed, num_hosts=num_hosts, duration_minutes=duration_minutes)
    df, manifest = generate_dataset(gen_config)
    save_dataset(df, manifest, out_dir / "dataset")

    train_df, val_df, test_df = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=seed)

    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_labels(train_df)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_labels(val_df)
    X_test = transform_features(preprocessor, test_df)
    y_test = extract_labels(test_df)

    train_config = TrainConfig(seed=seed)
    model, history = train_model(X_train, y_train, X_val, y_val, train_config)

    test_metrics = evaluate(model, X_test, y_test)
    artifact_paths = save_artifact(
        model,
        preprocessor,
        train_config,
        test_metrics,
        out_dir / "artifact",
        input_dim=X_train.shape[1],
    )

    result = {
        "manifest": manifest,
        "history": history,
        "test_metrics": test_metrics,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_multiclass(
    seed: int,
    out_dir: Path | str,
    *,
    num_hosts: int = 60,
    duration_minutes: float = 180.0,
    gen_config: GenerationConfig | None = None,
) -> dict:
    """Attack-type classifier: benign + each attack family, one NN head.

    Pass gen_config directly for full control (e.g. enabling the
    scenario-library agent's opt-in families); otherwise built from
    num_hosts/duration_minutes with the Foundation-stage scenario defaults.
    """
    out_dir = Path(out_dir)
    gen_config = gen_config or GenerationConfig(seed=seed, num_hosts=num_hosts, duration_minutes=duration_minutes)

    df, manifest = generate_dataset(gen_config)
    save_dataset(df, manifest, out_dir / "dataset")

    train_df, val_df, test_df = split_dataset(df, val_frac=0.15, test_frac=0.15, seed=seed)

    preprocessor = build_preprocessor(train_df)
    X_train = transform_features(preprocessor, train_df)
    y_train = extract_scenario_labels(train_df)
    X_val = transform_features(preprocessor, val_df)
    y_val = extract_scenario_labels(val_df)
    X_test = transform_features(preprocessor, test_df)
    y_test = extract_scenario_labels(test_df)

    class_names = list(SCENARIO_CLASSES)
    train_config = TrainConfig(seed=seed)
    model, history = train_multiclass_model(X_train, y_train, X_val, y_val, train_config, len(class_names))

    test_metrics = evaluate_multiclass(model, X_test, y_test, class_names)
    artifact_paths = save_artifact(
        model,
        preprocessor,
        train_config,
        test_metrics,
        out_dir / "artifact_multiclass",
        input_dim=X_train.shape[1],
        output_dim=len(class_names),
        class_names=class_names,
    )

    result = {
        "manifest": manifest,
        "history": history,
        "test_metrics": test_metrics,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    (out_dir / "run_summary_multiclass.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="data/generated/demo")
    parser.add_argument("--num-hosts", type=int, default=60)
    parser.add_argument("--duration-minutes", type=float, default=180.0)
    parser.add_argument(
        "--mode", choices=["binary", "multiclass"], default="binary",
        help="binary: normal-vs-suspicious (Foundation stage). multiclass: attack-type attribution.",
    )
    args = parser.parse_args()

    run_fn = run if args.mode == "binary" else run_multiclass
    result = run_fn(
        args.seed,
        args.out_dir,
        num_hosts=args.num_hosts,
        duration_minutes=args.duration_minutes,
    )
    print(json.dumps(result["test_metrics"], indent=2, sort_keys=True))
    print(f"split sizes: {result['split_sizes']}")
    print(f"artifacts written under: {args.out_dir}")


if __name__ == "__main__":
    main()
