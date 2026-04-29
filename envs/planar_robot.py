"""
PlanarRobot3DOFEnv — Custom Gymnasium Environment
===================================================
A 3-DOF planar robot (three rotational joints in a 2D plane) learning to reach a target.

Observation Space (8,):
    [θ1, θ2, θ3]        — joint angles (rad)
    [dθ1, dθ2, dθ3]     — joint angular velocities (rad/s)
    [x_target, y_target] — target position (m)

Action Space (3,):
    [u1, u2, u3] ∈ [-1, 1] — normalized torque/angular velocity for each joint

Reward:
    + (prev_dist - curr_dist)   — dense: reward for approaching target
    + 10.0                      — sparse bonus: reaching the target
    - 0.1 * joint_limit_penalty — penalty for exceeding joint limits
    - 0.001 * action_penalty    — penalty for large actions (smooth control)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class PlanarRobot3DOFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        link_lengths: tuple = (0.4, 0.3, 0.2),
        max_episode_steps: int = 500,
        goal_threshold: float = 0.05,
        dt: float = 0.05,
        max_angular_vel: float = 2.0,
        joint_limit: float = np.pi,
    ):
        super().__init__()

        # ── Robot parameters ──────────────────────────────────────────────
        self.link_lengths   = np.array(link_lengths, dtype=np.float32)
        self.max_reach      = float(np.sum(self.link_lengths))
        self.dt             = dt
        self.max_angular_vel = max_angular_vel
        self.joint_limit    = joint_limit
        self.goal_threshold = goal_threshold
        self.max_episode_steps = max_episode_steps

        # ── Spaces ────────────────────────────────────────────────────────
        obs_low = np.array(
            [-joint_limit, -joint_limit, -joint_limit,
             -max_angular_vel, -max_angular_vel, -max_angular_vel,
             -self.max_reach, -self.max_reach],
            dtype=np.float32,
        )
        obs_high = -obs_low

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # ── Internal state ────────────────────────────────────────────────
        self.angles       = np.zeros(3, dtype=np.float32)
        self.velocities   = np.zeros(3, dtype=np.float32)
        self.target_pos   = np.zeros(2, dtype=np.float32)
        self._step_count  = 0
        self._prev_dist   = 0.0

        # ── Rendering ─────────────────────────────────────────────────────
        self.render_mode  = render_mode
        self.screen       = None
        self.clock        = None
        self.screen_size  = 600
        self.scale        = self.screen_size / (2.2 * self.max_reach)
        self.origin       = np.array([self.screen_size // 2, self.screen_size // 2])

    # ══════════════════════════════════════════════════════════════════════
    # Core API
    # ══════════════════════════════════════════════════════════════════════

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomize initial joint angles
        self.angles     = self.np_random.uniform(-np.pi / 2, np.pi / 2, size=3).astype(np.float32)
        self.velocities = np.zeros(3, dtype=np.float32)

        # Randomize target position within the reachable workspace
        r     = self.np_random.uniform(0.1 * self.max_reach, 0.9 * self.max_reach)
        theta = self.np_random.uniform(-np.pi, np.pi)
        self.target_pos = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)

        self._step_count = 0
        ee_pos = self.forward_kinematics(self.angles)
        self._prev_dist = float(np.linalg.norm(ee_pos - self.target_pos))

        obs  = self._get_obs()
        info = self._get_info(ee_pos)
        return obs, info

    def step(self, action: np.ndarray):
        # Clip & scale action → angular velocity
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        delta_vel = action * self.max_angular_vel * self.dt

        # Update velocities & angles (simple Euler integration)
        self.velocities = np.clip(
            self.velocities + delta_vel,
            -self.max_angular_vel,
            self.max_angular_vel,
        ).astype(np.float32)

        self.angles = np.clip(
            self.angles + self.velocities * self.dt,
            -self.joint_limit,
            self.joint_limit,
        ).astype(np.float32)

        # Compute end-effector position
        ee_pos   = self.forward_kinematics(self.angles)
        curr_dist = float(np.linalg.norm(ee_pos - self.target_pos))

        # Compute reward
        reward, reached = self._compute_reward(curr_dist, action)
        self._prev_dist  = curr_dist

        # Termination conditions
        terminated = bool(reached)
        self._step_count += 1
        truncated  = self._step_count >= self.max_episode_steps

        obs  = self._get_obs()
        info = self._get_info(ee_pos)
        info["distance_to_target"] = curr_dist

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("Planar Robot 3-DOF RL")
            else:
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        # ── Draw ──────────────────────────────────────────────────────────
        self.screen.fill((30, 30, 30))  # dark background

        # Draw grid
        self._draw_grid()

        # Draw target
        target_screen = self._to_screen(self.target_pos)
        pygame.draw.circle(self.screen, (255, 80, 80), target_screen, 12)
        pygame.draw.circle(self.screen, (255, 150, 150), target_screen,
                           int(self.goal_threshold * self.scale), 2)

        # Draw robot links
        joint_positions = self._get_joint_positions()
        colors = [(100, 180, 255), (100, 230, 180), (255, 200, 100)]
        for i in range(3):
            p1 = self._to_screen(joint_positions[i])
            p2 = self._to_screen(joint_positions[i + 1])
            pygame.draw.line(self.screen, colors[i], p1, p2, 6 - i)
            pygame.draw.circle(self.screen, (220, 220, 220), p1, 8)

        # Draw end-effector
        ee_screen = self._to_screen(joint_positions[-1])
        pygame.draw.circle(self.screen, (255, 255, 80), ee_screen, 10)

        # Draw HUD
        font = pygame.font.SysFont("monospace", 16)
        ee_pos   = joint_positions[-1]
        dist     = np.linalg.norm(ee_pos - self.target_pos)
        hud_lines = [
            f"Step : {self._step_count}/{self.max_episode_steps}",
            f"Dist : {dist:.3f} m",
            f"EE   : ({ee_pos[0]:.2f}, {ee_pos[1]:.2f})",
            f"Tgt  : ({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f})",
        ]
        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (10, 10 + i * 20))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock  = None

    # ══════════════════════════════════════════════════════════════════════
    # Kinematics
    # ══════════════════════════════════════════════════════════════════════

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Return end-effector (x, y) position."""
        l1, l2, l3 = self.link_lengths
        θ1, θ2, θ3 = angles
        x = (l1 * np.cos(θ1)
             + l2 * np.cos(θ1 + θ2)
             + l3 * np.cos(θ1 + θ2 + θ3))
        y = (l1 * np.sin(θ1)
             + l2 * np.sin(θ1 + θ2)
             + l3 * np.sin(θ1 + θ2 + θ3))
        return np.array([x, y], dtype=np.float32)

    def _get_joint_positions(self) -> list:
        """Return list of 4 positions: base + 3 joints + end-effector."""
        l1, l2, l3 = self.link_lengths
        θ1, θ2, θ3 = self.angles
        p0 = np.array([0.0, 0.0])
        p1 = p0 + l1 * np.array([np.cos(θ1), np.sin(θ1)])
        p2 = p1 + l2 * np.array([np.cos(θ1 + θ2), np.sin(θ1 + θ2)])
        p3 = p2 + l3 * np.array([np.cos(θ1 + θ2 + θ3), np.sin(θ1 + θ2 + θ3)])
        return [p0, p1, p2, p3]

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _compute_reward(self, curr_dist: float, action: np.ndarray):
        reward = float(self._prev_dist - curr_dist)          # dense shaping
        reached = curr_dist < self.goal_threshold
        if reached:
            reward += 10.0                                   # sparse bonus for reaching target
        # Penalty for approaching joint limits
        violation = np.sum(np.abs(self.angles) > self.joint_limit * 0.95)
        reward -= 0.1 * violation
        # Smooth control penalty for large actions
        reward -= 0.001 * float(np.sum(action ** 2))
        return reward, reached

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [self.angles, self.velocities, self.target_pos], dtype=np.float32
        )

    def _get_info(self, ee_pos: Optional[np.ndarray] = None) -> dict:
        if ee_pos is None:
            ee_pos = self.forward_kinematics(self.angles)
        return {
            "end_effector_pos": ee_pos.copy(),
            "target_pos":       self.target_pos.copy(),
            "joint_angles":     self.angles.copy(),
        }

    def _to_screen(self, pos: np.ndarray) -> tuple:
        """Convert world coordinates (m) to screen pixels."""
        x = int(self.origin[0] + pos[0] * self.scale)
        y = int(self.origin[1] - pos[1] * self.scale)  # flip Y axis
        return (x, y)

    def _draw_grid(self):
        import pygame
        step = int(self.scale * 0.2)
        if step < 10:
            return
        color = (50, 50, 50)
        for x in range(0, self.screen_size, step):
            pygame.draw.line(self.screen, color, (x, 0), (x, self.screen_size))
        for y in range(0, self.screen_size, step):
            pygame.draw.line(self.screen, color, (0, y), (self.screen_size, y))
        # Draw coordinate axes
        cx, cy = self.origin
        pygame.draw.line(self.screen, (70, 70, 70), (cx, 0), (cx, self.screen_size), 2)
        pygame.draw.line(self.screen, (70, 70, 70), (0, cy), (self.screen_size, cy), 2)
