"""Replay engine: drives a simulated clock over scored flow data.

Keeps simulated event time strictly separate from wall-clock time, per
docs/PRODUCT_VISION.md's Operations overview requirement. A UI calls
advance(wall_dt) once per tick; this module has no knowledge of Streamlit,
threads, or timers.
"""

from __future__ import annotations

import pandas as pd


class ReplayController:
    def __init__(self, df: pd.DataFrame, *, speed: float = 1.0):
        if df.empty:
            raise ValueError("cannot replay an empty DataFrame")
        if speed <= 0:
            raise ValueError("speed must be > 0")

        self._df = df.sort_values("event_time", kind="stable").reset_index(drop=True)
        self.min_time = float(self._df["event_time"].iloc[0])
        self.max_time = float(self._df["event_time"].iloc[-1])
        self.sim_time = self.min_time
        self.speed = float(speed)
        self.playing = False

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = float(speed)

    def seek(self, sim_time: float) -> None:
        self.sim_time = min(max(sim_time, self.min_time), self.max_time)

    def reset(self) -> None:
        self.sim_time = self.min_time
        self.playing = False

    def is_finished(self) -> bool:
        return self.sim_time >= self.max_time

    def advance(self, wall_dt_seconds: float) -> float:
        """Advance simulated time by wall_dt_seconds * speed if playing. Returns new sim_time."""
        if wall_dt_seconds < 0:
            raise ValueError("wall_dt_seconds must be >= 0")
        if self.playing:
            self.sim_time = min(self.sim_time + wall_dt_seconds * self.speed, self.max_time)
            if self.sim_time >= self.max_time:
                self.playing = False
        return self.sim_time

    def events_up_to(self, sim_time: float | None = None) -> pd.DataFrame:
        t = self.sim_time if sim_time is None else sim_time
        return self._df[self._df["event_time"] <= t]

    def window(self, window_s: float, *, sim_time: float | None = None) -> pd.DataFrame:
        t = self.sim_time if sim_time is None else sim_time
        return self._df[(self._df["event_time"] > t - window_s) & (self._df["event_time"] <= t)]
