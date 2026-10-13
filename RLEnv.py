#! /usr/bin/env python

import copy
import gymnasium as gym
import numpy as np

import pooltool as pt
from gymnasium import spaces


class RLBilliard(gym.Env):
  """Custom Gymnasium Environment for a 2-ball billiards pocketing task using pooltool."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, render_mode=None):
    super().__init__()

    self.render_mode = render_mode
    self.table = pt.Table.default()
    self.R = 0.028575  # Standard ball radius in meters

    # Identify Top-Left Pocket ID
    self.top_left_pocket_id = min(
        self.table.pockets.keys(),
        key=lambda p_id: self.table.pockets[p_id].center[0] ** 2
        + (self.table.l - self.table.pockets[p_id].center[1]) ** 2,
    )
    self.top_left_pocket = np.array([0.0, self.table.l])

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
    # [red_x, red_y, white_x, white_y, dx_to_pocket, dy_to_pocket, cut_angle]
    self.observation_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    self.shot = None
    self.cut_side = "STRAIGHT_IN"
    self.cut_angle = 0.0
    self.aim_angle_deg = 0.0

  def _determine_cut_side(self, white_pos, red_pos, pocket_pos):
    """Calculates cut side and cut angle between cue vector and pocket vector."""
    v_cue = np.array(red_pos) - np.array(white_pos)
    v_pocket = np.array(pocket_pos) - np.array(red_pos)

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
    vec_pocket = self.top_left_pocket - red_pos

    return np.array(
        [
            red_pos[0] / self.table.w,
            red_pos[1] / self.table.l,
            white_pos[0] / self.table.w,
            white_pos[1] / self.table.l,
            vec_pocket[0] / self.table.w,
            vec_pocket[1] / self.table.l,
            self.cut_angle / 180.0,
        ],
        dtype=np.float32,
    )

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # 1. Initialize Balls
    self.balls = pt.get_rack(pt.GameType.THREECUSHION, self.table)
    del self.balls["yellow"]

    # 2. Sample Red Ball: x in [0.2, 0.6], y in [1.25, 1.75]
    red_x = self.np_random.uniform(0.2, 0.6)
    red_y = self.np_random.uniform(1.25, 1.75)
    self.balls["red"].state.rvw[0] = [red_x, red_y, self.R]

    # 3. Sample White Ball: x in [0.2, 0.8], y in [0.75, 1.25]
    white_x = self.np_random.uniform(0.2, 0.8)
    white_y = self.np_random.uniform(0.75, 1.25)
    self.balls["white"].state.rvw[0] = [white_x, white_y, self.R]

    # 4. Calculate aiming and cut side geometry
    dx_aim = red_x - white_x
    dy_aim = red_y - white_y
    self.aim_angle_deg = np.rad2deg(np.arctan2(dy_aim, dx_aim)) % 360

    self.cut_side, self.cut_angle = self._determine_cut_side(
        (white_x, white_y), (red_x, red_y), self.top_left_pocket
    )

    observation = self._get_obs()
    info = {"cut_side": self.cut_side, "cut_angle": self.cut_angle}

    return observation, info

  def step(self, action):

    raw_angle = np.clip(action[0], -1.0, 1.0)
    raw_velocity = np.clip(action[1], -1.0, 1.0)

    # Map raw_angle [-1, 1] -> [0.0, 6.0]
    angle_magnitude = (raw_angle + 1.0) / 2.0 * 6.0

    # Map raw_velocity [-1, 1] -> [0.5, 3.0]
    v0 = 0.5 + (raw_velocity + 1.0) / 2.0 * (3.0 - 0.5)

    # print(f"Raw Action: {action}, Mapped Angle: {angle_magnitude:.2f}°, Mapped Velocity: {v0:.2f} m/s")

    # Apply sign based on determined cut side
    if self.cut_side == "LEFT":
      signed_control_angle = -angle_magnitude
    elif self.cut_side == "RIGHT":
      signed_control_angle = angle_magnitude
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

    red_pocketed_top_left: bool = any(
        event.event_type.has_pocket()
        and "red" in [agent.id for agent in event.agents]
        and self.top_left_pocket_id in [agent.id for agent in event.agents]
        for event in events
    )

    white_pocketed: bool = any(
        event.event_type.has_pocket()
        and "white" in [agent.id for agent in event.agents]
        for event in events
    )

    # Reward logic: +1 if red in top-left; 0 if both are pocketed or missed
    if red_pocketed_top_left and not white_pocketed:
      reward = 1.0
    else:
      reward = 0.0

    terminated = True  # Single-shot environment
    truncated = False

    observation = self._get_obs()
    info = {
        "red_pocketed_top_left": red_pocketed_top_left,
        "white_pocketed": white_pocketed,
        "observation": observation,
        "signed_control_angle": signed_control_angle,
        "final_phi": final_phi,
    }

    return observation, reward, terminated, truncated, info

  def render(self):
    if self.shot is not None:
      pt.show(self.shot)


# --- Test Run ---
if __name__ == "__main__":
  env = RLBilliard()
  obs, info = env.reset()

  print(f"Initial Cut Side: {info['cut_side']}")

  # Sample action: [angle_magnitude = 2.5, velocity = 2.0]
  action = np.random.uniform(-1.0, 1.0, size=(2,))
  obs, reward, terminated, truncated, info = env.step(action)

  print(f"Executed Control Angle: {info['signed_control_angle']:.2f}°")
  print(f"Reward Received: {reward}")
  print(f"Red Pocketed in Top-Left: {info['red_pocketed_top_left']}")
  print(f"White Scratch: {info['white_pocketed']}")

  # Render 3D viewer
  env.render()
