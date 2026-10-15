#! /usr/bin/env python

import copy
import gymnasium as gym
import numpy as np

import pooltool as pt
from gymnasium import spaces


class RLBilliard(gym.Env):
  """Custom Gymnasium Environment for a 2-ball billiards pocketing task using pooltool."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, render_mode=None, target_pocket="top_right",
               red_x_range=(0.2, 0.8), red_y_range=(1.25, 1.75),
               white_x_range=(0.2, 0.8), white_y_range=(0.75, 1.25)):
    super().__init__()

    self.render_mode = render_mode
    self.table = pt.Table.default()
    self.R = 0.028575  # Standard ball radius in meters
    self.target_pocket = target_pocket

    # Ball placement area boundaries (parameterized)
    self.red_x_range = red_x_range
    self.red_y_range = red_y_range
    self.white_x_range = white_x_range
    self.white_y_range = white_y_range

    # Identify Target Pocket ID based on configuration
    if self.target_pocket == "top_left":
      # Top-left: minimize x² + (l - y)²
      self.target_pocket_id = min(
          self.table.pockets.keys(),
          key=lambda p_id: self.table.pockets[p_id].center[0] ** 2
          + (self.table.l - self.table.pockets[p_id].center[1]) ** 2,
      )
      self.target_pocket_pos = np.array([0.0, self.table.l])
    else:  # top_right
      # Top-right: minimize (w - x)² + (l - y)²
      self.target_pocket_id = min(
          self.table.pockets.keys(),
          key=lambda p_id: (self.table.w - self.table.pockets[p_id].center[0]) ** 2
          + (self.table.l - self.table.pockets[p_id].center[1]) ** 2,
      )
      self.target_pocket_pos = np.array([self.table.w, self.table.l])

    # Action Space:
    # Action[0] = control_angle_magnitude in [0, 6] degrees
    # Action[1] = initial cue velocity V0 in [0.5, 3.0] m/s
    self.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
    )
    # Observation Space: (Matches normalized output in _get_obs)
    # [red_x, red_y, white_x, white_y, cut_angle]
    self.observation_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    self.shot = None
    self.shots = []  # Store multiple shots for continuous display
    self.cut_side = "STRAIGHT_IN"
    self.cut_angle = 0.0
    self.aim_angle_deg = 0.0

  def _determine_cut_side(self, white_pos, red_pos):
    """Calculates cut side and cut angle between cue vector and pocket vector."""
    v_cue = np.array(red_pos) - np.array(white_pos)
    v_pocket = self.target_pocket_pos - np.array(red_pos)

    cross_product = v_cue[0] * v_pocket[1] - v_cue[1] * v_pocket[0]

    unit_cue = v_cue / (np.linalg.norm(v_cue) + 1e-8)
    unit_pocket = v_pocket / (np.linalg.norm(v_pocket) + 1e-8)
    dot_product = np.clip(np.dot(unit_cue, unit_pocket), -1.0, 1.0)
    cut_angle_deg = np.rad2deg(np.arccos(dot_product))

    if np.abs(cross_product) < 1e-5:
      side = "STRAIGHT_IN"
    elif cross_product > 0:
      side = "LEFT"  # Apply negative control angle
    else:
      side = "RIGHT"  # Apply positive control angle

    return side, cut_angle_deg

  def _get_obs(self):
    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]

    return np.array(
        [
            red_pos[0] / self.table.w,
            red_pos[1] / self.table.l,
            white_pos[0] / self.table.w,
            white_pos[1] / self.table.l,
            self.cut_angle / 180.0,
        ],
        dtype=np.float32,
    )

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # 1. Initialize Balls
    self.balls = pt.get_rack(pt.GameType.THREECUSHION, self.table)
    del self.balls["yellow"]

    # 2. Sample Red Ball: x in red_x_range, y in red_y_range
    red_x = self.np_random.uniform(self.red_x_range[0], self.red_x_range[1])
    red_y = self.np_random.uniform(self.red_y_range[0], self.red_y_range[1])
    self.balls["red"].state.rvw[0] = [red_x, red_y, self.R]

    # 3. Sample White Ball: x in white_x_range, y in white_y_range
    white_x = self.np_random.uniform(self.white_x_range[0], self.white_x_range[1])
    white_y = self.np_random.uniform(self.white_y_range[0], self.white_y_range[1])
    self.balls["white"].state.rvw[0] = [white_x, white_y, self.R]

    # 4. Calculate aiming and cut side geometry
    dx_aim = red_x - white_x
    dy_aim = red_y - white_y
    self.aim_angle_deg = np.rad2deg(np.arctan2(dy_aim, dx_aim)) % 360

    self.cut_side, self.cut_angle = self._determine_cut_side(
        (white_x, white_y), (red_x, red_y)
    )

    observation = self._get_obs()
    info = {"cut_side": self.cut_side, "cut_angle": self.cut_angle}

    return observation, info

  def _calculate_max_cut_angle_deg(self, white_pos, red_pos):
    """Calculates the maximum angle deviation (in degrees) for tangent contact."""
    d = np.hypot(red_pos[0] - white_pos[0], red_pos[1] - white_pos[1])

    # Ball contact radius threshold
    ratio = (2.0 * self.R) / d

    # Clip ratio to [0.0, 1.0] to prevent numerical issues with arcsin
    ratio = np.clip(ratio, 0.0, 1.0)

    theta_max_rad = np.arcsin(ratio)
    return np.rad2deg(theta_max_rad)

  def step(self, action):

    raw_angle = np.clip(action[0], -1.0, 1.0)
    raw_velocity = np.clip(action[1], -1.0, 1.0)

    # Get current ball positions
    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]

  # 1. Compute dynamic max angle for tangent contact
    theta_max_deg = self._calculate_max_cut_angle_deg(white_pos, red_pos)

    # Map raw_angle [-1, 1] -> [0.0, theta_max_deg]
    angle_perturbation = (raw_angle + 1.0) / 2.0 * theta_max_deg
    # Map raw_velocity [-1, 1] -> [0.5, 3.0]
    
    v0 = 0.5 + (raw_velocity + 1.0) / 2.0 * (3.0 - 0.5)

    # print(f"Raw Action: {action}, Mapped Angle: {angle_magnitude:.2f}°, Mapped Velocity: {v0:.2f} m/s")

    # Apply sign based on determined cut side
    if self.cut_side == "LEFT":
      signed_control_angle = -angle_perturbation
    elif self.cut_side == "RIGHT":
      signed_control_angle = angle_perturbation
    else:  # STRAIGHT_IN
      signed_control_angle = 0.0

    final_phi = (self.aim_angle_deg + signed_control_angle) % 360

    # Execute simulation (deepcopy balls so initial reset state remains clean across steps)
    cue = pt.Cue(cue_ball_id="white")
    self.shot = pt.System(
        table=self.table, balls=copy.deepcopy(self.balls), cue=cue
    )
    self.shot.cue.set_state(v0, phi=final_phi, theta=0.0, a=0.0, b=0.0)
    self.shot.strike()

    try:
      pt.simulate(self.shot, inplace=True)
    except Exception:
      try:
        pt.simulate(self.shot, inplace=True, solver="analytic")
      except Exception:
        try:
          pt.simulate(self.shot, inplace=True, solver="numpy")
        except Exception:
          self.shot.events = []

    # Check pocket conditions using event_type.has_pocket()
    events = getattr(self.shot, "events", []) or []

    red_pocketed_target: bool = any(
        event.event_type.has_pocket()
        and "red" in [agent.id for agent in event.agents]
        and self.target_pocket_id in [agent.id for agent in event.agents]
        for event in events
    )

    white_pocketed: bool = any(
        event.event_type.has_pocket()
        and "white" in [agent.id for agent in event.agents]
        for event in events
    )

    # Reward logic: +1 if red in target pocket; 0 if both are pocketed or missed
    if red_pocketed_target and not white_pocketed:
      reward = 1.0
    else:
      reward = 0.0

    terminated = True  # Single-shot environment
    truncated = False

    observation = self._get_obs()
    info = {
        "red_pocketed_target": red_pocketed_target,
        "target_pocket": self.target_pocket,
        "white_pocketed": white_pocketed,
        "observation": observation,
        "signed_control_angle": signed_control_angle,
        "final_phi": final_phi,
    }

    return observation, reward, terminated, truncated, info

  def render(self):
    if self.shots:
      for i, shot in enumerate(self.shots):
        print(f"\nDisplaying Shot {i+1}...")
        pt.show(shot)
    elif self.shot is not None:
      pt.show(self.shot)


# --- Test Run ---
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches

  env = RLBilliard()
  obs, info = env.reset()

  # Visualize table and ball placement areas
  fig, ax = plt.subplots(1, 1, figsize=(12, 8))

  # Draw table (green)
  table_rect = patches.Rectangle(
      (0, 0), env.table.w, env.table.l,
      linewidth=2, edgecolor='darkgreen', facecolor='green', alpha=0.6,
      label='Table'
  )
  ax.add_patch(table_rect)

  # Draw red ball placement area (red shadow)
  red_width = env.red_x_range[1] - env.red_x_range[0]
  red_height = env.red_y_range[1] - env.red_y_range[0]
  red_area = patches.Rectangle(
      (env.red_x_range[0], env.red_y_range[0]), red_width, red_height,
      linewidth=2, edgecolor='darkred', facecolor='red', alpha=0.3,
      label='Red Ball Area'
  )
  ax.add_patch(red_area)

  # Draw white ball placement area (gray shadow)
  white_width = env.white_x_range[1] - env.white_x_range[0]
  white_height = env.white_y_range[1] - env.white_y_range[0]
  white_area = patches.Rectangle(
      (env.white_x_range[0], env.white_y_range[0]), white_width, white_height,
      linewidth=2, edgecolor='darkgray', facecolor='gray', alpha=0.3,
      label='White Ball Area'
  )
  ax.add_patch(white_area)

  # Set axis properties
  ax.set_xlim(-0.2, env.table.w + 0.2)
  ax.set_ylim(-0.2, env.table.l + 0.2)
  ax.set_aspect('equal')
  ax.grid(True, alpha=0.3)
  ax.legend(loc='upper right', fontsize=10)
  ax.set_xlabel('Width (m)', fontsize=11)
  ax.set_ylabel('Length (m)', fontsize=11)
  ax.set_title('Billiards Table with Ball Placement Areas', fontsize=13, fontweight='bold')

  # Save figure
  plt.tight_layout()
  plt.savefig('billiards_table_layout.png', dpi=150, bbox_inches='tight')
  print("✓ Table layout saved to 'billiards_table_layout.png'")
  plt.close()

  print(f"Initial Cut Side: {info['cut_side']}")

  # First shot with action[0] = -1
  print("\n--- Shot 1: action[0] = -1 ---")
  action1 = np.array([-1.0, 0.5], dtype=np.float32)
  obs, reward, terminated, truncated, info = env.step(action1)
  shot1 = env.shot

  print(f"Executed Control Angle: {info['signed_control_angle']:.2f}°")
  print(f"Reward Received: {reward}")
  print(f"Red Pocketed in Target Pocket ({env.target_pocket}): {info['red_pocketed_target']}")
  print(f"White Scratch: {info['white_pocketed']}")

  # Reset for second shot
  obs, info = env.reset()
  print(f"\n--- Shot 2: action[0] = 1 ---")
  action2 = np.array([1.0, 0.5], dtype=np.float32)
  obs, reward, terminated, truncated, info = env.step(action2)
  shot2 = env.shot

  print(f"Executed Control Angle: {info['signed_control_angle']:.2f}°")
  print(f"Reward Received: {reward}")
  print(f"Red Pocketed in Target Pocket ({env.target_pocket}): {info['red_pocketed_target']}")
  print(f"White Scratch: {info['white_pocketed']}")

  # Store both shots and render them together
  env.shots = [shot1, shot2]
  env.render()
