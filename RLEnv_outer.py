#! /usr/bin/env python

import copy
from billiards_tests import test_difficulty
import gymnasium as gym
import numpy as np

import pooltool as pt
from gymnasium import spaces
from billiards_visualizer import visualize_balls


class RLBilliard(gym.Env):
  """Custom Gymnasium Environment for a 2-ball billiards pocketing task using pooltool."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, render_mode=None,
               red_x_range=(0.1, 0.9), red_y_range=(1.25, 1.75),
               white_x_range=(0.1, 0.9), white_y_range=(0.75, 1.25),
               verbose=True):
    super().__init__()

    self.render_mode = render_mode
    self.verbose = verbose
    self.table = pt.Table.default()
    self.R = 0.028575  # Standard ball radius in meters

    # Ball placement area boundaries (parameterized)
    self.red_x_range = red_x_range
    self.red_y_range = red_y_range
    self.white_x_range = white_x_range
    self.white_y_range = white_y_range

    # Target pocket: always top-left
    # Top-left: minimize x² + (l - y)²
    self.target_pocket_id = min(
        self.table.pockets.keys(),
        key=lambda p_id: self.table.pockets[p_id].center[0] ** 2
        + (self.table.l - self.table.pockets[p_id].center[1]) ** 2,
    )
    self.target_pocket_pos = np.array([0.0, self.table.l])

    # Action Space:
    # Action[0] = control_angle_magnitude in [0, 6] degrees (from -1 to 1 normalized)
    # Action[1] = initial cue velocity V0 in [0.5, 3.0] m/s (from -1 to 1 normalized)
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
    self.shot_counter = 0  # Track total number of shots simulated
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

  def _reset_random(self):
    """
    Internal method to randomly initialize ball positions.
    
    Returns:
        tuple: (observation, info)
    """
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

  def reset(self, seed=None, options=None):
    """
    Reset the environment to a new episode with random ball positions.
    
    Generates a random table configuration.
    
    Returns:
        tuple: (observation, info) where observation is the initial state
    """
    super().reset(seed=seed)

    # Generate random ball positions
    obs, info = self._reset_random()

    return obs, info

  def _calculate_max_cut_angle_deg(self, white_pos, red_pos):
    """Calculates the maximum angle deviation (in degrees) for tangent contact."""
    d = np.hypot(red_pos[0] - white_pos[0], red_pos[1] - white_pos[1])

    # Ball contact radius threshold
    ratio = (2.0 * self.R) / d

    # Clip ratio to [0.0, 1.0] to prevent numerical issues with arcsin
    ratio = np.clip(ratio, 0.0, 1.0)

    theta_max_rad = np.arcsin(ratio)
    return np.rad2deg(theta_max_rad) + 0.1

  def _calculate_pocket_angle(self, white_pos, red_pos, pocket_pos):
    """
    Calculates the angle between pocket-to-white and pocket-to-red vectors.
    
    Args:
        white_pos: Position of white ball [x, y]
        red_pos: Position of red ball [x, y]
        pocket_pos: Position of target pocket [x, y]
    
    Returns:
        float: Angle in degrees between the two vectors
    """
    # Create vectors from pocket to each ball
    vec_to_white = np.array(white_pos) - np.array(pocket_pos)
    vec_to_red = np.array(red_pos) - np.array(pocket_pos)
    
    # Calculate angle using dot product
    dot_product = np.dot(vec_to_white, vec_to_red)
    magnitudes = np.linalg.norm(vec_to_white) * np.linalg.norm(vec_to_red)
    cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
    angle_between_deg = np.rad2deg(np.arccos(cos_angle))
    
    return angle_between_deg

  def _calculate_red_angle(self, white_pos, red_pos, pocket_pos):
    """
    Calculates the angle between pocket->red and red->white vectors.
    This represents how well-aligned the cue path is for hitting the red ball into the pocket.
    
    Args:
        white_pos: Position of white ball [x, y]
        red_pos: Position of red ball [x, y]
        pocket_pos: Position of target pocket [x, y]
    
    Returns:
        float: Angle in degrees between the two vectors
                0° = best angle (straight shot to pocket)
                90° = worst angle (perpendicular)
    """
    # Vector from pocket to red ball
    vec_pocket_to_red = np.array(red_pos) - np.array(pocket_pos)
    # Vector from red ball to white ball (cue approach)
    vec_red_to_white = np.array(white_pos) - np.array(red_pos)
    
    # Calculate angle using dot product
    dot_product = np.dot(vec_pocket_to_red, vec_red_to_white)
    magnitudes = np.linalg.norm(vec_pocket_to_red) * np.linalg.norm(vec_red_to_white)
    
    # Handle zero magnitudes (prevents division by zero)
    if magnitudes < 1e-10:
        return 0.0
    
    cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
    angle_between_deg = np.rad2deg(np.arccos(cos_angle))
    
    return angle_between_deg

  def calculate_position_difficulty(self):
    """
    Calculates the difficulty of the current position on a scale of 0-100.
    
    Difficulty depends on two factors:
    1. Red angle: Angle between pocket->red and red->white vectors
       0° = best angle (straight shot), 90° = worst angle (perpendicular)
    2. Distance to red ball: Closer distances are easier, farther distances are harder
    
    Returns:
        float: Difficulty score between 0 (easiest) and 100 (hardest)
    """
    # Get current ball positions and pocket center
    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]
    pocket_center = self.table.pockets[self.target_pocket_id].center
    pocket_center = np.array([float(pocket_center[0]), float(pocket_center[1])])
    
    # Calculate red angle (pocket->red vs red->white)
    red_angle = self._calculate_red_angle(white_pos, red_pos, pocket_center)
    
    # Calculate distance from pocket to red ball
    distance_to_red = np.linalg.norm(np.array(red_pos) - pocket_center)
    
    # Normalize angle: 0° = 0 difficulty, 90° = max difficulty
    # Normalize to 0-100 scale
    angle_difficulty = (red_angle / 90.0) * 100.0
    
    # Normalize distance: 
    # Maximum distance on table is diagonal from corner to corner
    max_distance = np.linalg.norm(np.array([self.table.w, self.table.l]))
    # Closer to pocket is easier (lower difficulty), farther is harder
    distance_difficulty = (distance_to_red / max_distance) * 100.0
    
    # Combine both factors with equal weight (50% each)
    total_difficulty = (angle_difficulty * 0.5) + (distance_difficulty * 0.5)
    
    # Clip to 0-100 range
    total_difficulty = np.clip(total_difficulty, 0.0, 100.0)
    
    return total_difficulty


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
    self.shot.cue.set_state(v0, phi=final_phi, theta=45.0, a=0.0, b=0.0)
    self.shot.strike()
    self.shot_counter += 1  # Increment shot counter

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
        "white_pocketed": white_pocketed,
        "observation": observation,
        "signed_control_angle": signed_control_angle,
        "final_phi": final_phi,
    }

    info["shot_counter"] = self.shot_counter
    info["position_difficulty"] = self.calculate_position_difficulty()

    if self.verbose:
      print(f"Shot {self.shot_counter}: red_pocketed_target={red_pocketed_target}, white_pocketed={white_pocketed}, reward={reward}, position_difficulty={self.calculate_position_difficulty()}")

    return observation, reward, terminated, truncated, info

  def visualize_balls(self, filename=None):
    """
    Visualizes the current ball positions on the table.
    Delegates to the billiards_visualizer module.
    
    Args:
        filename (str, optional): If provided, saves the image to this file.
                                  Otherwise displays the plot.
    """
    visualize_balls(self, filename)


  def render(self):
    if self.shots:
      for i, shot in enumerate(self.shots):
        if self.verbose:
          print(f"\nDisplaying Shot {i+1}...")
        pt.show(shot)
    elif self.shot is not None:
      pt.show(self.shot)

  def test_simple_shot(self, action=None, verbose=True):
    """
    Test method: Create a random table, play an action, and render it.
    
    Args:
        action (np.ndarray, optional): Action to execute. If None, uses [0.0, 0.5]
        verbose (bool): Whether to print debug information
    
    Returns:
        tuple: (observation, reward, terminated, truncated, info)
    """
    if action is None:
      action = np.array([0.0, 0.5])
    
    print("=" * 70)
    print("Test: Simple Shot (Random Table + Action + Render)")
    print("=" * 70)
    
    # Reset to random table
    print("\n1. Resetting environment to random table...")
    obs, info = self.reset()
    print(f"   ✓ Reset successful")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Cut side: {info.get('cut_side', 'N/A')}")
    print(f"   - Cut angle: {info.get('cut_angle', 'N/A'):.2f}°")
    
    # Execute action
    print(f"\n2. Executing action: {action}")
    obs, reward, terminated, truncated, info = self.step(action)
    print(f"   ✓ Step successful")
    print(f"   - Reward: {reward}")
    print(f"   - Red pocketed (target): {info.get('red_pocketed_target', 'N/A')}")
    print(f"   - White pocketed: {info.get('white_pocketed', 'N/A')}")
    print(f"   - Shot counter: {info.get('shot_counter', 'N/A')}")
    print(f"   - Position difficulty: {info.get('position_difficulty', 'N/A'):.2f}")
    
    # Render
    print(f"\n3. Rendering shot...")
    try:
      self.render()
      print(f"   ✓ Render successful")
    except Exception as e:
      print(f"   ✗ Render failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
    
    return obs, reward, terminated, truncated, info


# --- Test Run ---
if __name__ == "__main__":
  import numpy as np
  
  env = RLBilliard(verbose=True)
  
  # Run the simple shot test
  action = np.array([0.0, 0.5])
  env.test_simple_shot(action=action)

