"""Pygame based Angry Birds inspired game with YOLO control support."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import pygame

from .controller import ControlState, Controller
from .levels import BirdSpec, Block, Level, LEVELS, Pig

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
GROUND_Y = 620
SLINGSHOT_POS = pygame.Vector2(220, GROUND_Y - 20)
GRAVITY = pygame.Vector2(0, 980)
LAUNCH_POWER = 780


@dataclass
class GameStats:
    score: int = 0
    pigs_cleared: int = 0
    birds_used: int = 0


class Bird:
    def __init__(self, spec: BirdSpec) -> None:
        self.spec = spec
        self.position = SLINGSHOT_POS.copy()
        self.velocity = pygame.Vector2()
        self.radius = 22
        self.in_flight = False
        self.active = True
        self.ability_used = False
        self.rest_timer = 0.0

    def reset(self) -> None:
        self.position = SLINGSHOT_POS.copy()
        self.velocity.update(0, 0)
        self.in_flight = False
        self.active = True
        self.ability_used = False
        self.rest_timer = 0.0

    def launch(self, aim_offset: float, pullback: float) -> None:
        angle_deg = 50 + aim_offset * 35
        angle = math.radians(angle_deg)
        power = max(0.2, min(1.0, pullback))
        speed = LAUNCH_POWER * power / max(0.5, self.spec.mass)
        direction = pygame.Vector2(math.cos(angle), -math.sin(angle))
        self.velocity = direction * speed
        self.position = SLINGSHOT_POS.copy()
        self.in_flight = True
        self.rest_timer = 0.0

    def update(self, dt: float) -> None:
        if not self.in_flight:
            return
        self.velocity += GRAVITY * dt
        self.position += self.velocity * dt
        if self.position.y >= GROUND_Y - self.radius:
            self.position.y = GROUND_Y - self.radius
            if self.velocity.length() < 60:
                self.velocity.update(0, 0)
                self.rest_timer += dt
            else:
                self.velocity.y *= -0.25
                self.velocity.x *= 0.5
        else:
            self.rest_timer = 0.0

    def trigger_ability(self) -> bool:
        if self.ability_used or not self.in_flight:
            return False
        if self.spec.ability == "dash":
            self.velocity *= 1.8
            self.ability_used = True
            return True
        if self.spec.ability == "explode":
            self.ability_used = True
            return True
        return False


class AngryBirdsGame:
    def __init__(self, controller: Controller, level: Level | None = None) -> None:
        pygame.init()
        self.controller = controller
        self.level = level or LEVELS[0]
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("YOLO Angry Birds")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        self.big_font = pygame.font.SysFont("arial", 36, bold=True)
        self.birds: list[Bird] = [Bird(spec) for spec in self.level.birds]
        self.current_bird_index = 0
        self.stats = GameStats()
        self.running = True
        self.message: Optional[str] = None
        self.message_time = 0.0
        self._explosion_request: Optional[pygame.Vector2] = None
        self._explosion_radius = 90

    @property
    def current_bird(self) -> Optional[Bird]:
        if self.current_bird_index < len(self.birds):
            return self.birds[self.current_bird_index]
        return None

    def _remaining_pigs(self) -> list[Pig]:
        return [pig for pig in self.level.pigs if pig.health > 0]

    def _apply_damage(self, amount: float, pig: Pig | None = None, block: Block | None = None) -> None:
        if pig is not None:
            pig.health -= amount
            if pig.health <= 0:
                pig.health = 0
                self.stats.score += 5000
                self.stats.pigs_cleared += 1
        if block is not None:
            block.health -= amount
            if block.health <= 0:
                block.health = 0
                self.stats.score += 1000

    def _bird_damage_value(self, bird: Bird) -> float:
        return bird.spec.mass * bird.velocity.length() * 0.6

    def _handle_collisions(self, bird: Bird) -> None:
        if not bird.in_flight:
            return
        for pig in self._remaining_pigs():
            distance = bird.position.distance_to(pig.center)
            if distance <= bird.radius + pig.radius:
                self._apply_damage(self._bird_damage_value(bird), pig=pig)
                bird.velocity *= 0.65
        for block in self.level.blocks:
            if block.health <= 0:
                continue
            if block.rect.inflate(6, 6).collidepoint(bird.position):
                self._apply_damage(self._bird_damage_value(bird), block=block)
                normal = pygame.Vector2(0, -1)
                if bird.position.x < block.rect.left:
                    normal = pygame.Vector2(-1, 0)
                elif bird.position.x > block.rect.right:
                    normal = pygame.Vector2(1, 0)
                elif bird.position.y < block.rect.top:
                    normal = pygame.Vector2(0, -1)
                else:
                    normal = pygame.Vector2(0, 1)
                bird.velocity.reflect_ip(normal)
                bird.velocity *= 0.5

    def _trigger_explosion(self, position: pygame.Vector2) -> None:
        for pig in self._remaining_pigs():
            distance = position.distance_to(pig.center)
            if distance < self._explosion_radius:
                self._apply_damage(80 * (1 - distance / self._explosion_radius), pig=pig)
        for block in self.level.blocks:
            if block.health <= 0:
                continue
            block_center = pygame.Vector2(block.rect.center)
            distance = position.distance_to(block_center)
            if distance < self._explosion_radius:
                self._apply_damage(40 * (1 - distance / self._explosion_radius), block=block)
        self.message = "Boom!"
        self.message_time = time.monotonic()

    def _check_level_end(self) -> None:
        if not self._remaining_pigs():
            self.message = "Level cleared!"
            self.message_time = time.monotonic()
            unused = len(self.birds) - self.stats.birds_used
            self.stats.score += unused * 10000
            self.running = False
        elif self.current_bird is None:
            self.message = "Out of birds!"
            self.message_time = time.monotonic()
            self.running = False

    def _draw_blocks(self) -> None:
        for block in self.level.blocks:
            if block.health <= 0:
                continue
            pygame.draw.rect(self.screen, block.material.color, block.rect)
            if block.health < block.material.durability:
                ratio = block.health / block.material.durability
                if ratio < 0.4:
                    color = (160, 40, 40)
                    pygame.draw.rect(self.screen, color, block.rect, 3)

    def _draw_pigs(self) -> None:
        for pig in self.level.pigs:
            if pig.health <= 0:
                continue
            pygame.draw.circle(self.screen, (120, 200, 120), pig.center, pig.radius)
            health_ratio = max(0.0, pig.health / 30.0)
            eye_offset = 6
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (int(pig.center.x - eye_offset), int(pig.center.y - 6)),
                6,
            )
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (int(pig.center.x + eye_offset), int(pig.center.y - 6)),
                6,
            )
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (int(pig.center.x - eye_offset), int(pig.center.y - 6)),
                3,
            )
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (int(pig.center.x + eye_offset), int(pig.center.y - 6)),
                3,
            )

    def _draw_slingshot(self, control_state: ControlState | None) -> None:
        base_pos = SLINGSHOT_POS
        pygame.draw.line(self.screen, (60, 30, 15), base_pos + (-25, 40), base_pos + (25, 40), 8)
        if control_state is not None:
            offset = pygame.Vector2(control_state.pullback * 80, control_state.aim_offset * -80)
            pull_pos = base_pos - offset
            pygame.draw.line(self.screen, (120, 70, 40), base_pos + (-12, 0), pull_pos, 4)
            pygame.draw.line(self.screen, (120, 70, 40), base_pos + (12, 0), pull_pos, 4)
            pygame.draw.circle(self.screen, (160, 120, 90), pull_pos, 22)

    def _draw_bird(self, bird: Bird) -> None:
        pygame.draw.circle(self.screen, bird.spec.color, (int(bird.position.x), int(bird.position.y)), bird.radius)

    def _draw_ui(self) -> None:
        hud = self.font.render(
            f"Score: {self.stats.score} | Pigs: {self.stats.pigs_cleared}/{len(self.level.pigs)} | Bird {self.current_bird_index + 1}/{len(self.birds)}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(hud, (20, 20))
        instructions = [
            "Move the tracked object to aim.",
            "Let go (or move away) to launch.",
            "Space/Right-click triggers special ability.",
        ]
        for i, text in enumerate(instructions):
            surf = self.font.render(text, True, (230, 230, 230))
            self.screen.blit(surf, (20, 50 + i * 20))
        if self.message and time.monotonic() - self.message_time < 2.5:
            msg_surface = self.big_font.render(self.message, True, (255, 255, 0))
            rect = msg_surface.get_rect(center=(SCREEN_WIDTH // 2, 80))
            self.screen.blit(msg_surface, rect)

    def game_loop(self) -> None:
        control_state: Optional[ControlState] = None
        running = True
        while running and self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if self.current_bird and self.current_bird.trigger_ability():
                        if self.current_bird.spec.ability == "explode":
                            self._explosion_request = self.current_bird.position.copy()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    if self.current_bird and self.current_bird.trigger_ability():
                        if self.current_bird.spec.ability == "explode":
                            self._explosion_request = self.current_bird.position.copy()

            polled = self.controller.poll()
            if polled is not None:
                control_state = polled
            if control_state and control_state.launch and self.current_bird and not self.current_bird.in_flight:
                self.current_bird.launch(control_state.aim_offset, control_state.pullback)
                self.stats.birds_used += 1
            if control_state and control_state.ability and self.current_bird:
                if self.current_bird.trigger_ability() and self.current_bird.spec.ability == "explode":
                    self._explosion_request = self.current_bird.position.copy()
            if control_state and not self.current_bird:
                control_state = None

            self.screen.fill((135, 206, 235))
            pygame.draw.rect(self.screen, (120, 200, 120), (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y))
            self._draw_slingshot(control_state if self.current_bird and not self.current_bird.in_flight else None)

            for block in self.level.blocks:
                block.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

            bird = self.current_bird
            if bird:
                if not bird.in_flight and control_state is not None:
                    offset = pygame.Vector2(control_state.pullback * 80, control_state.aim_offset * -80)
                    bird.position = SLINGSHOT_POS - offset
                bird.update(dt)
                self._handle_collisions(bird)
                self._draw_bird(bird)
                if bird.rest_timer > 1.2:
                    self.current_bird_index += 1
                    bird.in_flight = False
                    bird.active = False
                    if self.current_bird:
                        self.current_bird.reset()
                    self._check_level_end()

            if self._explosion_request is not None:
                self._trigger_explosion(self._explosion_request)
                self._explosion_request = None

            self._draw_blocks()
            self._draw_pigs()
            self._draw_ui()

            pygame.display.flip()

        self.controller.close()
        pygame.quit()


def run_game(controller: Controller, level: Level | None = None) -> None:
    game = AngryBirdsGame(controller, level)
    game.game_loop()
