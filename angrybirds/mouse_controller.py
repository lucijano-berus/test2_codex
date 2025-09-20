"""Mouse based fallback controller."""
from __future__ import annotations

import pygame

from .controller import ControlState, Controller


class MouseController(Controller):
    """Maps the mouse drag in the game window to the slingshot controls."""

    def __init__(self) -> None:
        self._active = False
        self._pull_start: tuple[int, int] | None = None

    def poll(self) -> ControlState | None:
        mouse_buttons = pygame.mouse.get_pressed(3)
        pos = pygame.mouse.get_pos()

        if mouse_buttons[0]:
            if not self._active:
                self._active = True
                self._pull_start = pos
            if self._pull_start is None:
                return None
            dx = pos[0] - self._pull_start[0]
            dy = pos[1] - self._pull_start[1]
            pull_dist = min((dx * dx + dy * dy) ** 0.5, 240)
            pullback = pull_dist / 240
            aim_offset = max(-1.0, min(1.0, -dy / 240))
            return ControlState(aim_offset=aim_offset, pullback=pullback)
        else:
            if self._active:
                self._active = False
                return ControlState(aim_offset=0.0, pullback=0.0, launch=True)
        return None

    def close(self) -> None:
        pass
