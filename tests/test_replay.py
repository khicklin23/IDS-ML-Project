import pandas as pd
import pytest

from ids_ml.replay import ReplayController


def _df() -> pd.DataFrame:
    return pd.DataFrame({"event_time": [0.0, 5.0, 10.0, 20.0], "flow_id": ["a", "b", "c", "d"]})


def test_initial_state():
    rc = ReplayController(_df())
    assert rc.sim_time == 0.0
    assert not rc.playing
    assert not rc.is_finished()


def test_advance_while_paused_does_nothing():
    rc = ReplayController(_df())
    rc.advance(5.0)
    assert rc.sim_time == 0.0


def test_play_advances_and_respects_speed():
    rc = ReplayController(_df(), speed=2.0)
    rc.play()
    rc.advance(3.0)
    assert rc.sim_time == 6.0


def test_advance_clamps_and_marks_finished():
    rc = ReplayController(_df())
    rc.play()
    rc.advance(100.0)
    assert rc.sim_time == 20.0
    assert rc.is_finished()
    assert not rc.playing


def test_seek_clamps_within_bounds():
    rc = ReplayController(_df())
    rc.seek(-5.0)
    assert rc.sim_time == 0.0
    rc.seek(1000.0)
    assert rc.sim_time == 20.0


def test_events_up_to():
    rc = ReplayController(_df())
    rc.seek(10.0)
    assert list(rc.events_up_to()["flow_id"]) == ["a", "b", "c"]


def test_window():
    rc = ReplayController(_df())
    rc.seek(20.0)
    # window is (sim_time - window_s, sim_time]: lower bound exclusive so a
    # replayed event is never double-counted across adjacent windows.
    assert list(rc.window(20.0)["flow_id"]) == ["b", "c", "d"]
    assert list(rc.window(9.9)["flow_id"]) == ["d"]


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        ReplayController(pd.DataFrame({"event_time": []}))


def test_rejects_nonpositive_speed():
    with pytest.raises(ValueError):
        ReplayController(_df(), speed=0)
