from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import TextIO


@dataclass
class ProgressBar:
    label: str
    total: int
    enabled: bool = True
    stream: TextIO | None = None
    width: int = 28
    min_interval_seconds: float = 0.5
    log_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 0)
        self.current = 0
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0
        self.stream = self.stream or sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def advance(self, amount: int = 1, *, detail: str | None = None, force: bool = False) -> None:
        self.current += int(amount)
        self.render(detail=detail, force=force)

    def render(self, *, detail: str | None = None, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        interval = self.min_interval_seconds if self.is_tty else self.log_interval_seconds
        if not force and now - self.last_rendered_at < interval:
            return
        self.last_rendered_at = now
        elapsed = max(now - self.started_at, 1.0e-9)
        rate = self.current / elapsed
        if self.total:
            fraction = min(max(self.current / self.total, 0.0), 1.0)
            filled = min(self.width, int(round(self.width * fraction)))
            bar = "#" * filled + "-" * (self.width - filled)
            percent = 100.0 * fraction
            eta_seconds = (self.total - self.current) / rate if rate > 0 else float("inf")
            text = (
                f"{self.label} [{bar}] {self.current}/{self.total} "
                f"{percent:5.1f}% elapsed {self._format_duration(elapsed)} "
                f"eta {self._format_duration(eta_seconds)}"
            )
        else:
            text = f"{self.label} {self.current} elapsed {self._format_duration(elapsed)}"
        if detail:
            text = f"{text} | {detail}"
        if self.is_tty:
            self.stream.write("\r" + text[:160])
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def finish(self, *, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self.render(detail=detail, force=True)
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds == float("inf"):
            return "--:--"
        seconds = max(int(round(seconds)), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
