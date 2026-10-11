import pandas as pd
import pytest

from ids_ml.data.schema import ALL_COLUMNS, SCHEMA_VERSION, SchemaError, coerce_dtypes, validate


def _valid_row(**overrides) -> dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
        "seed": 1,
        "flow_id": "run-1-0000000",
        "event_time": 0.0,
        "session_id": "host-0000",
        "src_host_id": "host-0000",
        "dst_host_id": "host-0001",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "duration_s": 1.0,
        "src_port": 1234,
        "dst_port": 443,
        "protocol": "tcp",
        "service": "https",
        "packets_sent": 5,
        "packets_recv": 5,
        "bytes_sent": 500,
        "bytes_recv": 500,
        "outcome": "established",
        "prior_gap_s": 0.0,
        "scenario": "benign",
        "label": "normal",
    }
    row.update(overrides)
    return row


def _df(*rows) -> pd.DataFrame:
    return coerce_dtypes(pd.DataFrame(list(rows))[list(ALL_COLUMNS)])


def test_valid_frame_passes():
    df = _df(_valid_row())
    validate(df)  # should not raise


def test_missing_column_rejected():
    df = _df(_valid_row()).drop(columns=["protocol"])
    with pytest.raises(SchemaError):
        validate(df)


def test_bad_enum_rejected():
    df = _df(_valid_row(protocol="carrier-pigeon"))
    with pytest.raises(SchemaError):
        validate(df)


def test_negative_count_rejected():
    df = _df(_valid_row(bytes_sent=-1))
    with pytest.raises(SchemaError):
        validate(df)


def test_out_of_range_port_rejected():
    df = _df(_valid_row(dst_port=70000))
    with pytest.raises(SchemaError):
        validate(df)


def test_duplicate_flow_id_rejected():
    df = _df(_valid_row(), _valid_row(event_time=1.0))
    with pytest.raises(SchemaError):
        validate(df)


def test_label_scenario_mismatch_rejected():
    df = _df(_valid_row(scenario="port_scan", label="normal"))
    with pytest.raises(SchemaError):
        validate(df)


def test_wrong_schema_version_rejected():
    df = _df(_valid_row(schema_version="0.0.1"))
    with pytest.raises(SchemaError):
        validate(df)


def test_empty_frame_rejected():
    df = _df(_valid_row()).iloc[0:0]
    with pytest.raises(SchemaError):
        validate(df)


@pytest.mark.parametrize("scenario", ["service_flood", "outbound_transfer"])
def test_new_scenarios_keep_binary_label_semantics(scenario):
    validate(_df(_valid_row(scenario=scenario, label="suspicious")))
    with pytest.raises(SchemaError, match="label does not match"):
        validate(_df(_valid_row(scenario=scenario, label="normal")))
