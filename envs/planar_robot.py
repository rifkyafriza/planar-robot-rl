"""
PlanarRobot3DOFEnv — Custom Gymnasium Environment
===================================================
A 3-DOF planar robot (three rotational joints in a 2D plane) learning to reach a target.

Observation Space (8,):
    [θ1, θ2, θ3]        — joint angles (rad)
    [dθ1, dθ2, dθ3]     — joint angular velocities (rad/s)
    [x_target, y_target] — target position (m)

Action Space (3,):
    [u1, u2, u3] ∈ [-1, 1] — normalized torque for each joint

Physics:
    Torque → angular acceleration (τ = I * α, I per-link)
    Euler integration with damping

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
        link_masses: tuple = (1.0, 0.8, 0.5),
        max_episode_steps: int = 500,
        goal_threshold: float = 0.05,
        dt: float = 0.05,
        max_angular_vel: float = 3.0,
        joint_limit: float = np.pi,
        damping: float = 0.1,
        min_target_dist: float = 0.15,
    ):
        super().__init__()

        # ── Robot parameters ──────────────────────────────────────────────
        self.link_lengths    = np.array(link_lengths, dtype=np.float32)
        self.link_masses     = np.array(link_masses, dtype=np.float32)
        self.max_reach       = float(np.sum(self.link_lengths))
        self.dt              = dt
        self.max_angular_vel = max_angular_vel
        self.joint_limit     = joint_limit
        self.goal_threshold  = goal_threshold
        self.max_episode_steps = max_episode_steps
        self.damping         = damping
        self.min_target_dist = min_target_dist

        # Moment of inertia per joint: I = sum(m_i * l_i^2) for links beyond joint
        # Simplified: each joint controls its own link + distal links
        self.inertia = np.array([
            float(np.sum(self.link_masses[i:] * self.link_lengths[i:] ** 2))
            for i in range(3)
        ], dtype=np.float32)

        # Max torque scale (so action [-1,1] maps to physically meaningful torque)
        self.max_torque = self.inertia * max_angular_vel * 4.0

        # ── Spaces ────────────────────────────────────────────────────────
        # Use slightly expanded bounds to avoid check_env warnings during transients
        vel_bound = max_angular_vel * 1.2
        obs_low = np.array(
            [-joint_limit] * 3 + [-vel_bound] * 3 + [-self.max_reach] * 2,
            dtype=np.float32,
        )
        obs_high = np.array(
            [joint_limit] * 3 + [vel_bound] * 3 + [self.max_reach] * 2,
            dtype=np.float32,
        )

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

        # Generate reachable target — retry until distance from origin is meaningful
        for _ in range(50):
            target_angles = self.np_random.uniform(-self.joint_limit, self.joint_limit, size=3).astype(np.float32)
            candidate = self.forward_kinematics(target_angles)
            if np.linalg.norm(candidate) >= self.min_target_dist:
                self.target_pos = candidate
                break
        else:
            self.target_pos = candidate  # fallback

        self._step_count = 0
        ee_pos = self.forward_kinematics(self.angles)
        self._prev_dist = float(np.linalg.norm(ee_pos - self.target_pos))

        obs  = self._get_obs()
        info = self._get_info(ee_pos)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Torque = action * max_torque
        torque = action * self.max_torque

        # Angular acceleration: α = (τ - damping * ω) / I
        angular_acc = (torque - self.damping * self.velocities) / self.inertia

        # Euler integration: ω' = ω + α*dt, θ' = θ + ω'*dt
        self.velocities = np.clip(
            self.velocities + angular_acc * self.dt,
            -self.max_angular_vel,
            self.max_angular_vel,
        ).astype(np.float32)

        self.angles = np.clip(
            self.angles + self.velocities * self.dt,
            -self.joint_limit,
            self.joint_limit,
        ).astype(np.float32)

        # Compute end-effector position
        ee_pos    = self.forward_kinematics(self.angles)
        curr_dist = float(np.linalg.norm(ee_pos - self.target_pos))

        # Compute reward
        reward, reached = self._compute_reward(curr_dist, action)
        self._prev_dist  = curr_dist

        # Termination
        terminated = bool(reached)
        self._step_count += 1
        truncated  = self._step_count >= self.max_episode_steps

        obs  = self._get_obs()
        info = self._get_info(ee_pos)
        info["distance_to_target"] = curr_dist

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

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
        self.screen.fill((30, 30, 30))
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
        ee_pos = joint_positions[-1]
        dist   = np.linalg.norm(ee_pos - self.target_pos)
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
            return None

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
            reward += 10.0                                   # sparse bonus
        # Penalty for approaching joint limits (within 5% of limit)
        violation = np.sum(np.abs(self.angles) > self.joint_limit * 0.95)
        reward -= 0.1 * float(violation)
        # Smooth control penalty
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
        cx, cy = self.origin
        pygame.draw.line(self.screen, (70, 70, 70), (cx, 0), (cx, self.screen_size), 2)
        pygame.draw.line(self.screen, (70, 70, 70), (0, cy), (self.screen_size, cy), 2)
