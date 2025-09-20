"""Definitions of demo levels for the Angry Birds clone."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pygame


@dataclass
class Material:
    name: str
    color: tuple[int, int, int]
    durability: float


WOOD = Material("wood", (181, 101, 29), 40.0)
GLASS = Material("glass", (130, 200, 255), 20.0)
STONE = Material("stone", (140, 140, 140), 80.0)


@dataclass
class Block:
    rect: pygame.Rect
    material: Material
    health: float

    @classmethod
    def create(cls, x: int, y: int, w: int, h: int, material: Material) -> "Block":
        rect = pygame.Rect(x, y, w, h)
        return cls(rect=rect, material=material, health=material.durability)


@dataclass
class Pig:
    center: pygame.Vector2
    radius: int = 22
    health: float = 20.0


@dataclass
class BirdSpec:
    name: str
    color: tuple[int, int, int]
    mass: float
    ability: str | None = None


RED_BIRD = BirdSpec("red", (220, 20, 60), mass=1.0)
YELLOW_BIRD = BirdSpec("yellow", (255, 215, 0), mass=0.9, ability="dash")
BLACK_BIRD = BirdSpec("black", (40, 40, 40), mass=1.2, ability="explode")


@dataclass
class Level:
    name: str
    birds: Sequence[BirdSpec]
    blocks: list[Block]
    pigs: list[Pig]


def level_one() -> Level:
    blocks = [
        Block.create(780, 440, 60, 160, WOOD),
        Block.create(840, 480, 60, 120, GLASS),
        Block.create(900, 440, 60, 160, WOOD),
        Block.create(780, 400, 180, 40, STONE),
    ]
    pigs = [
        Pig(center=pygame.Vector2(810, 560)),
        Pig(center=pygame.Vector2(870, 560)),
        Pig(center=pygame.Vector2(840, 360), radius=24, health=30.0),
    ]
    birds = [RED_BIRD, YELLOW_BIRD, BLACK_BIRD]
    return Level("Wooden Watchtower", birds=birds, blocks=blocks, pigs=pigs)


LEVELS: list[Level] = [level_one()]
