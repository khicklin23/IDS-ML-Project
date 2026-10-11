# Product vision

## Target experience
Build a professional IDS simulation and neural network training workbench. A user generates a synthetic network environment, trains and compares detectors, replays activity, investigates alerts, and understands why predictions changed. The first small classifier is a foundation for this broader product, not the final scope.

Python supplies reproducible synthetic telemetry and PyTorch supplies the learned detector. Streamlit is the planned first UI; revisit presentation technology only if it prevents the intended experience. No framework migration is decided here.

## Visual direction: security operations meets neural networks
Use a polished dark command-center theme: deep navy/charcoal, readable typography, restrained cyan/teal for traffic and neural activity, amber/red for severity. Combine network topology and flow paths with interactive neural layer diagrams, activation views, training curves, and confidence distributions. Animation should communicate actual replay events or sampled model state, with pause and reduced-motion controls.

Make the overview readable at a glance, then offer investigation detail. Use consistent spacing, accessible contrast, labeled charts, sortable tables, and clear empty/loading/error states. Visual polish is a product requirement from the first working UI.

Every panel identifies whether it shows generated ground truth, model inference, or an explanatory schematic. Neural visuals must not imply that decorative activity is a measured activation or a causal explanation.

## Main workspaces
1. **Operations overview:** simulated sensor health, hosts and connections, throughput, alerts by severity, detection latency, and an event timeline. Separate replay time from wall-clock time.
2. **Scenario and data lab:** select environment size, protocols, background traffic, scenario mix, noise, imbalance, seed, event count/duration, and burst timing. Preview distributions and estimated output size before generation; export datasets and their manifests.
3. **Training studio:** architecture and hyperparameter controls, run progress, losses and metrics, experiment comparison, checkpoints, model/preprocessing versions, and a useful neural view tied to model state.
4. **Detection and replay:** play, pause, and change replay speed; run versioned models over generated telemetry; inspect scores, tune thresholds, and compare a small rule baseline with the neural detector.
5. **Alert investigation:** filter and sort alerts, inspect related flows and host timelines, review supporting features, group related alerts into incidents, acknowledge and annotate them, and export findings.
6. **Evaluation:** confusion matrix, per-class precision/recall/F1, false positives on benign activity, precision-recall curves, threshold tradeoffs, and performance across unseen scenarios and changed traffic distributions.

## IDS capability roadmap
Initial capability: flow/event telemetry, normal-versus-suspicious scoring, severity, alert history, host/connection views, and repeatable replay. Expand toward multiclass scenario recognition, anomaly scores, rule/model comparison, incident correlation, sensor status, investigation workflow, and reporting.

Candidate simulated scenarios: scanning, authentication bursts, service flood patterns, unusual outbound transfers, beacon-like periodic traffic, and lateral-movement-like sequences. These are generated behavioral patterns with documented assumptions. Start with a few well-tested families before expanding coverage.

Live packet capture, real protocol inspection, malware execution, real network blocking, and enterprise integrations are separate future decisions. Synthetic flow patterns cannot establish those capabilities. Any response action in the initial product acts on simulation state and is labeled accordingly.

## Reliable synthetic data
- Define a versioned schema including event time, host/flow identity, protocol/service, durations, packet/byte counts, connection outcomes, and scenario provenance. Add fields only when the generator can support them consistently.
- Generate coherent host populations and event sequences with seeded randomness, realistic-enough overlap/noise, burst behavior, and configurable class imbalance. Preserve schema/parameter/seed manifests for reproducibility.
- Keep scenario labels and generator-only metadata out of model inputs. Split by host/session/scenario as appropriate; avoid sending related or duplicated flows into both training and test sets. Fit preprocessing on training data only.
- Test determinism, constraints, distribution behavior, and cross-field consistency. Include unfamiliar seeds, scenario variations, benign lookalikes, and distribution shifts in evaluation.
- Stream or batch larger runs, report progress, and support cancellation. Agree measurable performance targets on the available hardware before promising dataset scale or replay throughput.

## Training depth
Begin with a reliable small PyTorch baseline. Add configurable networks, validation-based early stopping, class imbalance handling, reproducible run records, checkpoint selection, threshold selection, and artifact compatibility checks. Compare complexity against a simpler baseline; introduce temporal models or anomaly detection only when a documented data/task need and evaluation justify them.

Keep generation, feature preparation, training, evaluation, replay/inference, alert storage, and UI components separate. This supports meaningful experiments and lets the interface evolve without retraining logic hidden in page callbacks.

## Delivery stages and acceptance
- **Foundation:** resolve setup; establish schema, seeded generator, baseline training, and meaningful checks.
- **First complete experience:** attractive overview, data generation, training progress, replay, alert inspection, and held-out evaluation work together using one reproducible scenario bundle.
- **Expanded simulation:** multiple scenario families, configurable experiments, model comparison, incident grouping, report export, and bounded larger datasets.
- **Advanced research:** improve temporal/anomaly modeling and robustness only after baseline results and failure modes are recorded.

Success means a user can reproduce a dataset and training run, replay activity through the saved detector, inspect a specific alert, explain a threshold tradeoff, and export results through a cohesive professional UI. Label synthetic conditions and model versions throughout. Do not advertise production detection accuracy from synthetic scores.

## Alignment for both agents
This is the user's target vision as of 2026-10-10. Detailed layout, scenario distributions, model size, storage, and performance targets remain implementation decisions. Both agents must read this document before proposing data, training, or UI changes, and link their task's acceptance criteria to the relevant stage. Vision alignment does not itself authorize installations, commits, pushes, or immediate application implementation.
