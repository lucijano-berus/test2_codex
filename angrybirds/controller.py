"""Controller interfaces for Angry Birds YOLO project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class ControlState:
    """Represents the current state of the slingshot control."""

    aim_offset: float
    pullback: float
    launch: bool = False
    ability: bool = False

    def clamp(self) -> "ControlState":
        self.aim_offset = max(-1.0, min(1.0, self.aim_offset))
        self.pullback = max(0.0, min(1.0, self.pullback))
        return self


class Controller(Protocol):
    """Protocol for control backends."""

    def poll(self) -> Optional[ControlState]:
        """Return the latest control state.

        None indicates that control data is not currently available. In that case the
        previous value should usually be kept.
        """

    def close(self) -> None:
        """Release any resources when the controller is shut down."""
        raise NotImplementedError
