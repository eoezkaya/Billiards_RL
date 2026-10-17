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
               use_curriculum=True, fixed_difficulty=100.0,
               curriculum_initial_difficulty=20.0, curriculum_counter_1=50_000,
               curriculum_final_difficulty=100.0, curriculum_counter_2=100_000,
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

    # Curriculum learning parameters
    self.use_curriculum = use_curriculum
    self.fixed_difficulty = fixed_difficulty
    self.curriculum_initial_difficulty = curriculum_initial_difficulty
    self.curriculum_counter_1 = curriculum_counter_1
    self.curriculum_final_difficulty = curriculum_final_difficulty
    self.curriculum_counter_2 = curriculum_counter_2
    
    # Current difficulty level (updated based on curriculum)
    if not self.use_curriculum:
        self.current_curriculum_difficulty = self.fixed_difficulty
    else:
        self.current_curriculum_difficulty = self._calculate_curriculum_difficulty()

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
    Internal method to randomly initialize ball positions without curriculum control.
    Used by reset_to_difficulty to generate random configurations to test.
    
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
    Reset the environment to a new episode.
    
    If curriculum learning is enabled (use_curriculum=True):
        - Generates a table position matching the current curriculum difficulty level
        - Updates difficulty for next reset based on curriculum schedule
    
    If curriculum learning is disabled (use_curriculum=False):
        - Generates a table position at fixed difficulty (fixed_difficulty parameter)
        - Difficulty stays constant across all resets
    
    Returns:
        tuple: (observation, info) where observation is the initial state
    """
    super().reset(seed=seed)

    if self.use_curriculum:
      # Use curriculum-based difficulty
      obs, info = self.reset_to_difficulty(
          target_difficulty=self.current_curriculum_difficulty,
          max_attempts=100
      )
      # Update curriculum difficulty for next reset
      self.update_curriculum_difficulty()
    else:
      # Use fixed difficulty (no curriculum)
      obs, info = self.reset_to_difficulty(
          target_difficulty=self.fixed_difficulty,
          max_attempts=100
      )

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

  def calculate_position_difficulty(self, target_pocket_id=None, visualize=False):
    """
    Calculates the difficulty of the current position on a scale of 0-100.
    
    Difficulty depends on two factors:
    1. Red angle: Angle between pocket->red and red->white vectors
       0° = best angle (straight shot), 90° = worst angle (perpendicular)
    2. Distance to red ball: Closer distances are easier, farther distances are harder
    
    Args:
        target_pocket_id (str, optional): Pocket ID to calculate difficulty for.
                                         If None, uses self.target_pocket_id (top-left)
        visualize (bool, optional): If True, displays the positions and triangle visualization.
                                   Default is False.
    
    Returns:
        float: Difficulty score between 0 (easiest) and 100 (hardest)
    """
    # Use default target pocket if not specified
    if target_pocket_id is None:
      target_pocket_id = self.target_pocket_id
    
    # Get current ball positions and pocket center
    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]
    pocket_center = self.table.pockets[target_pocket_id].center
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
    
    # Display visualization if requested
    if visualize:
      from billiards_visualizer import visualize_billiards_position
      visualize_billiards_position(self, red_angle, total_difficulty, target_pocket_id=target_pocket_id)
    
    return total_difficulty

  def _calculate_curriculum_difficulty(self, target_counter=None):
    """
    Internal method to calculate difficulty level based on shot counter.
    
    Uses three phases:
    - Phase 1 (counter <= c1): Difficulty = a (initial)
    - Phase 2 (c1 < counter < c2): Linearly interpolate from a to b
    - Phase 3 (counter >= c2): Difficulty = b (final, constant)
    """
    if target_counter is None:
      target_counter = self.shot_counter
    
    # Phase 1: Before counter_1, return initial difficulty
    if target_counter <= self.curriculum_counter_1:
      return self.curriculum_initial_difficulty
    
    # Phase 2: Between counter_1 and counter_2, linearly interpolate
    elif target_counter < self.curriculum_counter_2:
      progress = (target_counter - self.curriculum_counter_1) / (
          self.curriculum_counter_2 - self.curriculum_counter_1
      )
      difficulty = self.curriculum_initial_difficulty + (
          self.curriculum_final_difficulty - self.curriculum_initial_difficulty
      ) * progress
      return difficulty
    
    # Phase 3: After counter_2, return final difficulty (constant)
    else:
      return self.curriculum_final_difficulty

  def update_curriculum_difficulty(self, target_counter=None):
    """
    Updates self.current_curriculum_difficulty based on the curriculum schedule.
    
    Call this method to sync the instance's current_curriculum_difficulty parameter
    with the curriculum progression based on shot counter.
    
    Args:
        target_counter (int, optional): Counter value to calculate for.
                                       If None, uses current self.shot_counter
    
    Returns:
        float: The updated current_curriculum_difficulty value
    """
    self.current_curriculum_difficulty = self._calculate_curriculum_difficulty(target_counter)
    return self.current_curriculum_difficulty

  def get_curriculum_difficulty(self, target_counter=None):
    """
    Calculates difficulty level based on shot counter for curriculum learning.
    
    Uses three phases:
    - Phase 1 (counter <= c1): Difficulty = a (initial)
    - Phase 2 (c1 < counter < c2): Linearly interpolate from a to b
    - Phase 3 (counter >= c2): Difficulty = b (final, constant)
    
    This enables progressive difficulty increase as the agent trains.
    
    Args:
        target_counter (int, optional): Counter value to calculate for.
                                       If None, uses current self.shot_counter
    
    Returns:
        float: Difficulty level between 0 (easiest) and 100 (hardest)
    """
    return self._calculate_curriculum_difficulty(target_counter)

  def reset_to_difficulty(self, target_difficulty, max_attempts=1000):
    """
    Generates a random table setup with difficulty at most the target level.
    Keeps generating random positions until one with acceptable difficulty is found.
    
    Args:
        target_difficulty (float): Maximum difficulty level (0-100)
        max_attempts (int): Maximum number of attempts before giving up (default: 1000)
    
    Returns:
        tuple: (observation, info) from reset(), or None if max_attempts exceeded
        
    Raises:
        ValueError: If target_difficulty is not in range [0, 100]
    """
    if not (0.0 <= target_difficulty <= 100.0):
      raise ValueError(f"target_difficulty must be in range [0, 100], got {target_difficulty}")
    
    attempts = 0
    while attempts < max_attempts:
      # Generate random ball positions (without curriculum control)
      obs, info = self._reset_random()
      
      # Calculate difficulty of this configuration
      position_difficulty = self.calculate_position_difficulty()
      
      # If difficulty meets target, return this configuration
      if position_difficulty <= target_difficulty:
        if self.verbose:
          print(f"✓ Found suitable position at difficulty {position_difficulty:.1f}/100 (target: {target_difficulty}/100)")
        info["position_difficulty"] = position_difficulty
        info["current_curriculum_difficulty"] = self.current_curriculum_difficulty
        return obs, info
      
      attempts += 1
    
    # If we couldn't find suitable position, log warning and return last attempt
    if self.verbose:
      print(f"⚠ Could not find position with difficulty <= {target_difficulty}/100 after {max_attempts} attempts")
      print(f"  Returning position with difficulty {position_difficulty:.1f}/100 (closest found)")
    info["position_difficulty"] = position_difficulty
    info["current_curriculum_difficulty"] = self.current_curriculum_difficulty
    return obs, info



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
    info["current_curriculum_difficulty"] = self.current_curriculum_difficulty
    info["position_difficulty"] = self.calculate_position_difficulty()

    if self.verbose:
      print(f"Shot {self.shot_counter}: red_pocketed_target={red_pocketed_target}, white_pocketed={white_pocketed}, reward={reward}, current_curriculum_difficulty={self.current_curriculum_difficulty}, position_difficulty={self.calculate_position_difficulty()}")

    self.update_curriculum_difficulty()

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

  def generate_episode_dataset(self, num_episodes=1000, output_dir="episode_visualizations"):
    """
    Generates multiple episodes and creates visualizations for each.
    Saves all images to a directory with difficulty scores.
    
    Args:
        num_episodes (int): Number of episodes to generate (default: 1000)
        output_dir (str): Directory name to store all visualizations
    """
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      if self.verbose:
        print(f"Created directory: {output_dir}")
    
    if self.verbose:
      print(f"\nGenerating {num_episodes} episode visualizations...")
      print(f"Saving to: {output_dir}/\n")
    
    difficulties = []
    
    for episode in range(num_episodes):
      # Reset environment for new episode
      obs, info = self.reset()
      
      # Calculate difficulty for this episode
      difficulty = self.calculate_position_difficulty()
      difficulties.append(difficulty)
      
      # Generate filename with episode number and difficulty
      filename = os.path.join(output_dir, f"episode_{episode:04d}_diff_{difficulty:05.1f}.png")
      
      # Save visualization
      self.visualize_balls(filename)
      
      # Print progress every 100 episodes
      if (episode + 1) % 100 == 0 and self.verbose:
        print(f"  Generated {episode + 1}/{num_episodes} episodes")
    
    # Print summary statistics
    if self.verbose:
      print(f"\n✓ Successfully generated {num_episodes} episodes!")
      print(f"  Average difficulty: {np.mean(difficulties):.1f}/100")
      print(f"  Min difficulty: {np.min(difficulties):.1f}/100")
      print(f"  Max difficulty: {np.max(difficulties):.1f}/100")
      print(f"  All images saved to: {output_dir}/")

  def render(self):
    if self.shots:
      for i, shot in enumerate(self.shots):
        if self.verbose:
          print(f"\nDisplaying Shot {i+1}...")
        pt.show(shot)
    elif self.shot is not None:
      pt.show(self.shot)


# --- Test Run ---
if __name__ == "__main__":
  from billiards_tests import test_initial, test_difficulty, test_position_difficulty, test_calculate_position_difficulty_with_visualization
  
  env = RLBilliard(use_curriculum=False, verbose=True)
#  test_initial(env)
#  test_difficulty(env, target_difficulty=50)
#  test_position_difficulty(env, target_pocket='lt')
  test_calculate_position_difficulty_with_visualization(env)
  

