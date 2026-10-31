#! /usr/bin/env python

import copy
import gymnasium as gym
import numpy as np

import pooltool as pt
from gymnasium import spaces

red_x_range = (0.1, 0.9)
red_y_range = (0.1, 1.95)
white_x_range = (0.1, 0.9)
white_y_range = (0.1, 1.95)

class RLBilliard_pocket_invariant(gym.Env):
  """Custom Gymnasium Environment for a 2-ball billiards pocketing task using pooltool."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, render_mode=None,
               red_x_range=red_x_range, red_y_range=red_y_range,
               white_x_range=white_x_range, white_y_range=white_y_range,
               use_curriculum=True, fixed_difficulty=100.0,
               curriculum_initial_difficulty=20.0, curriculum_counter_1=20_000,
               curriculum_final_difficulty=80.0, curriculum_counter_2=100_000,
               target_pocket_id='lt',
               training_corner_pocket=True,
               verbose=True):
    super().__init__()

    self.render_mode = render_mode
    self.verbose = verbose
    self.table = pt.Table.default()
    self.R = 0.028575  # Standard ball radius in meters
    self.max_cue_speed = 4.0  # Maximum cue speed in m/s
    self.min_cue_speed = 0.8  # Minimum cue speed in m/s
    self.linear_velocity = False  # Use quadratic scaling for cue velocity by default

    # Ball placement area boundaries (parameterized)
    self.red_x_range = red_x_range
    self.red_y_range = red_y_range
    self.white_x_range = white_x_range
    self.white_y_range = white_y_range
    self.play_cushion = False
    self.train_corner_pocket = training_corner_pocket

    # Target pocket: can be 'lb', 'lt', 'rb', 'rt', 'ml', or 'mr'
    if target_pocket_id not in ['lb', 'lt', 'rb', 'rt', 'lc', 'rc']:
      raise ValueError(f"target_pocket_id must be 'lb', 'lt', 'rb', 'rt', 'lc', or 'rc', got '{target_pocket_id}'")
    
    self.target_pocket_id = target_pocket_id
    
    # Get pocket center from table
    
    pocket_center = self.table.pockets[target_pocket_id].center
    
    self.target_pocket_pos = np.array([float(pocket_center[0]), float(pocket_center[1])])

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
    # [red_radius, red_angle, white_radius, white_angle, pocket_angle, red_angle_tri, white_angle_tri,
    #  white_dist_left_cushion, white_dist_top_cushion]
    # Note: triangle angles normalized to [0,1], cushion distances normalized by table dimensions
    self.observation_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    self.shot = None
    self.shots = []  # Store multiple shots for continuous display
    self.shot_counter = 0  # Track total number of shots simulated
    self.cut_side = "STRAIGHT_IN"
    self.cut_angle = 0.0
    self.aim_angle_deg = 0.0
    self.cushion_info = None  # Store cushion shot information when calculated

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

  def set_target_pocket(self, pocket_id):
    """
    Change the target pocket for the environment.
    
    This method allows switching between different pockets without creating
    a new environment instance. Useful for adaptive strategies that play
    different pockets based on position difficulty.
    
    Args:
        pocket_id (str): The target pocket ID. Must be one of:
                        'lb' (bottom-left), 'lt' (top-left), 'rb' (bottom-right), 
                        'rt' (top-right), 'ml' (middle-left), 'mr' (middle-right)
    
    Raises:
        ValueError: If pocket_id is not a valid pocket
    
    Returns:
        None
    """
    if pocket_id not in ['lb', 'lt', 'rb', 'rt', 'lc', 'rc']:
      raise ValueError(f"pocket_id must be 'lb', 'lt', 'rb', 'rt', 'lc', or 'rc', got '{pocket_id}'")
    
    self.target_pocket_id = pocket_id
    pocket_center = self.table.pockets[pocket_id].center
    self.target_pocket_pos = np.array([float(pocket_center[0]), float(pocket_center[1])])
    
    if self.verbose:
      pocket_names = {
          'lt': 'Top-Left', 'rt': 'Top-Right', 
          'lb': 'Bottom-Left', 'rb': 'Bottom-Right',
          'ml': 'Middle-Left', 'mr': 'Middle-Right'
      }
      print(f"Target pocket changed to: {pocket_names[pocket_id]} ({pocket_id})")

    # update pocket and related coordinates
    self.pocket_coordinates = self._get_obs_pocket_relative()
    self.pocket_polar_coords = self.get_obs_pocket_relative_polar()

    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]
        
    # Calculate cut side again based on updated target pocket
    self.cut_side, self.cut_angle = self._determine_cut_side(
      (white_pos[0], white_pos[1]), (red_pos[0], red_pos[1]))



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
    maxR = np.sqrt(self.table.w**2 + self.table.l**2)
    
    # Get triangle angles
    triangle_angles = self.calculate_triangle_angles()
    
    # Normalize triangle angles: angles in degrees [0, 180] -> [0, 1]
    pocket_angle_norm = triangle_angles['pocket_angle'] / 180.0
    red_angle_norm = triangle_angles['red_angle'] / 180.0
    white_angle_norm = triangle_angles['white_angle'] / 180.0
    
    # Get white ball (cue ball) position
    white_pos = self.balls["white"].state.rvw[0][:2]
    
    # Calculate distances to cushions based on target pocket
    # For middle pockets: measure distance to closest relevant cushion
    if self.target_pocket_id in ['lb', 'lt']:
        # Left pockets: distance to left cushion
        dist_to_vertical_cushion = white_pos[0]
    elif self.target_pocket_id in ['rb', 'rt']:
        # Right pockets: distance to right cushion
        dist_to_vertical_cushion = self.table.w - white_pos[0]
    elif self.target_pocket_id == 'lc':
        # Middle-left: distance to left cushion
        dist_to_vertical_cushion = white_pos[0]
    elif self.target_pocket_id == 'rc':
        # Middle-right: distance to right cushion
        dist_to_vertical_cushion = self.table.w - white_pos[0]
    
    if self.target_pocket_id in ['lt', 'rt']:
        # Top pockets: distance to top cushion
        dist_to_horizontal_cushion = self.table.l - white_pos[1]
    elif self.target_pocket_id in ['lb', 'rb']:
        # Bottom pockets: distance to bottom cushion
        dist_to_horizontal_cushion = white_pos[1]
    elif self.target_pocket_id in ['lc', 'rc']:
        # Middle pockets: distance is irrelevant (set to 0)
        dist_to_horizontal_cushion = 0
    
    
    # Normalize cushion distances by table dimensions
    white_dist_vertical_norm = dist_to_vertical_cushion / self.table.w
    white_dist_horizontal_norm = dist_to_horizontal_cushion / self.table.l
    
    return np.array(
        [
            self.pocket_polar_coords['red_radius'] / maxR,
            self.pocket_polar_coords['red_angle'] / 90.0,
            self.pocket_polar_coords['white_radius'] / maxR,
            self.pocket_polar_coords['white_angle'] / 90.0,
            pocket_angle_norm,
            red_angle_norm,
            white_angle_norm,
            white_dist_vertical_norm,
            white_dist_horizontal_norm,
        ],
        dtype=np.float32,
    )

  def _get_obs_pocket_relative(self):
    """
    Get pocket-relative coordinates (not normalized, no cut_angle).
    
    Creates a coordinate system where the origin is at the target pocket:
    - +x points toward the table interior (horizontal)
    - +y points toward the table interior (vertical)
    - -x points toward the table exterior
    
    Coordinate transformations for each pocket:
    - Bottom-Left (lb): x' = x, y' = y (no change, pocket at origin)
    - Bottom-Right (rb): x' = w - x, y' = y (flip x, origin at bottom-right)
    - Top-Left (lt): x' = x, y' = l - y (flip y, origin at top-left)
    - Top-Right (rt): x' = w - x, y' = l - y (flip both, origin at top-right)
    
    Returns:
        np.ndarray: A 4-element float32 array with unnormalized pocket-relative coordinates:
                   [red_x_pocket, red_y_pocket, white_x_pocket, white_y_pocket] (in meters)
    """
    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]
    
    # Transform to pocket-relative coordinates based on target pocket
    if self.target_pocket_id == 'lb':
        # Bottom-left: no transformation needed, pocket is at origin
        red_x_pocket = red_pos[0]
        red_y_pocket = red_pos[1]
        white_x_pocket = white_pos[0]
        white_y_pocket = white_pos[1]
    
    elif self.target_pocket_id == 'rb':
        # Bottom-right: flip x-coordinates (pocket at right edge)
        red_x_pocket = self.table.w - red_pos[0]
        red_y_pocket = red_pos[1]
        white_x_pocket = self.table.w - white_pos[0]
        white_y_pocket = white_pos[1]
    
    elif self.target_pocket_id == 'lt':
        # Top-left: flip y-coordinates (pocket at top edge)
        red_x_pocket = red_pos[0]
        red_y_pocket = self.table.l - red_pos[1]
        white_x_pocket = white_pos[0]
        white_y_pocket = self.table.l - white_pos[1]
    
    elif self.target_pocket_id == 'rt':
        # Top-right: flip both x and y coordinates (pocket at top-right corner)
        red_x_pocket = self.table.w - red_pos[0]
        red_y_pocket = self.table.l - red_pos[1]
        white_x_pocket = self.table.w - white_pos[0]
        white_y_pocket = self.table.l - white_pos[1]
    
    elif self.target_pocket_id == 'lc':
        # Left-center: pocket at left edge center, no x-flip, y stays as is
        red_x_pocket = red_pos[0]
        red_y_pocket = red_pos[1] - self.table.l / 2.0
        white_x_pocket = white_pos[0]
        white_y_pocket = white_pos[1] - self.table.l / 2.0
    
    elif self.target_pocket_id == 'rc':
        # Right-center: pocket at right edge center, flip x, y distance from center
        red_x_pocket = self.table.w - red_pos[0]
        red_y_pocket = red_pos[1] - self.table.l / 2.0
        white_x_pocket = self.table.w - white_pos[0]
        white_y_pocket = white_pos[1] - self.table.l / 2.0
    
    else:
        raise ValueError(f"Unknown target_pocket_id: {self.target_pocket_id}")
    
    return np.array(
        [
            red_x_pocket,
            red_y_pocket,
            white_x_pocket,
            white_y_pocket
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

    white_x = self.np_random.uniform(self.white_x_range[0], self.white_x_range[1])

    if self.train_corner_pocket:
      while True:
          white_y = self.np_random.uniform(self.white_y_range[0], self.white_y_range[1])
          if white_y < red_y:
            break
    else:
          white_y = self.np_random.uniform(self.white_y_range[0], self.white_y_range[1])

    self.balls["white"].state.rvw[0] = [white_x, white_y, self.R]

    red_pos = self.balls["red"].state.rvw[0][:2]
    white_pos = self.balls["white"].state.rvw[0][:2]
    
    # Calculate cut side and cut angle based on current ball positions
    self.cut_side, self.cut_angle = self._determine_cut_side(
      (white_pos[0], white_pos[1]), (red_pos[0], red_pos[1]))

    # 4. Calculate aiming angle geometry
    dx_aim = red_x - white_x
    dy_aim = red_y - white_y
    self.aim_angle_deg = np.rad2deg(np.arctan2(dy_aim, dx_aim)) % 360
    
    # 5. Calculate cushion shot info if play_cushion is enabled
    if self.play_cushion:
        self.cushion_info = self.calculate_cushion_shot_angle(
            white_pos=(white_x, white_y), 
            red_pos=(red_x, red_y)
        )
    else:
        self.cushion_info = None

    self.pocket_coordinates = self._get_obs_pocket_relative()
    self.pocket_polar_coords = self.get_obs_pocket_relative_polar()

    observation = self._get_obs()
    info = {}

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

  def reset_with_positions(self, red_pos, white_pos):
    """
    Reset the environment with specific ball positions.
    
    This method allows you to initialize the environment with predetermined
    ball positions instead of random generation. Useful for testing specific
    scenarios, debugging, or setting up custom table states.
    
    Args:
        red_pos (tuple or list): Red ball position [x, y] in meters
        white_pos (tuple or list): White ball position [x, y] in meters
    
    Returns:
        tuple: (observation, info) where observation is the initial state
    
    Raises:
        ValueError: If ball positions are outside table boundaries or too close together
    
    Example:
        >>> env = RLBilliard()
        >>> obs, info = env.reset_with_positions(
        ...     red_pos=(0.5, 1.5),
        ...     white_pos=(0.5, 0.9)
        ... )
    """
    # Initialize seed
    super().reset(seed=None)
    
    # Initialize balls
    self.balls = pt.get_rack(pt.GameType.THREECUSHION, self.table)
    del self.balls["yellow"]
    
    # Validate and set ball positions
    red_x, red_y = float(red_pos[0]), float(red_pos[1])
    white_x, white_y = float(white_pos[0]), float(white_pos[1])
    
    # Check bounds
    if not (0 <= red_x <= self.table.w and 0 <= red_y <= self.table.l):
      raise ValueError(f"Red ball position ({red_x}, {red_y}) is outside table bounds (0-{self.table.w}, 0-{self.table.l})")
    
    if not (0 <= white_x <= self.table.w and 0 <= white_y <= self.table.l):
      raise ValueError(f"White ball position ({white_x}, {white_y}) is outside table bounds (0-{self.table.w}, 0-{self.table.l})")
    
    # Check minimum distance between balls (must be at least 2 * radius apart)
    min_distance = 2 * self.R
    distance = np.hypot(red_x - white_x, red_y - white_y)
    if distance < min_distance:
      raise ValueError(f"Balls too close: distance={distance:.4f}m, minimum required={min_distance:.4f}m")
    
    # Set ball positions
    self.balls["red"].state.rvw[0] = [red_x, red_y, self.R]
    self.balls["white"].state.rvw[0] = [white_x, white_y, self.R]
    
    # Calculate aiming and cut side geometry
    dx_aim = red_x - white_x
    dy_aim = red_y - white_y
    self.aim_angle_deg = np.rad2deg(np.arctan2(dy_aim, dx_aim)) % 360
    
    self.cut_side, self.cut_angle = self._determine_cut_side(
        (white_x, white_y), (red_x, red_y)
    )
    
    observation = self._get_obs()
    info = {
        "cut_side": self.cut_side,
        "cut_angle": self.cut_angle,
        "red_pos": (red_x, red_y),
        "white_pos": (white_x, white_y)
    }
    
    if self.verbose:
      print(f"Reset with positions: Red=({red_x:.3f}, {red_y:.3f}), White=({white_x:.3f}, {white_y:.3f})")
    
    return observation, info

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

  def calculate_triangle_angles(self, target_pocket_id=None):
    """
    Calculates the three interior angles of the triangle formed by:
    - Red ball position
    - White ball position
    - Target pocket position
    
    Args:
        target_pocket_id (str, optional): Pocket ID to calculate angles for.
                                        If None, uses self.target_pocket_id (top-left)
    
    Returns:
        dict: Dictionary containing:
            - 'pocket_angle': Angle at the pocket vertex (in degrees)
            - 'red_angle': Angle at the red ball vertex (in degrees)
            - 'white_angle': Angle at the white ball vertex (in degrees)
            - 'total': Sum of angles (should be ~180°, for validation)
    """
    # Use default target pocket if not specified
    if target_pocket_id is None:
      target_pocket_id = self.target_pocket_id
    
    # Get current ball positions and pocket center
    red_pos = np.array(self.balls["red"].state.rvw[0][:2])
    white_pos = np.array(self.balls["white"].state.rvw[0][:2])
    
    # Get pocket center (handle middle pockets which aren't predefined in pooltool)
    if target_pocket_id in ['ml', 'mr']:
      if target_pocket_id == 'ml':
        pocket_center = (0.0, self.table.l / 2.0)
      else:  # 'mr'
        pocket_center = (self.table.w, self.table.l / 2.0)
    else:
      pocket_center = self.table.pockets[target_pocket_id].center
    
    pocket_pos = np.array([float(pocket_center[0]), float(pocket_center[1])])
    
    # Calculate angle at pocket vertex (angle between pocket->red and pocket->white)
    vec_pocket_to_red = red_pos - pocket_pos
    vec_pocket_to_white = white_pos - pocket_pos
    
    dot_product = np.dot(vec_pocket_to_red, vec_pocket_to_white)
    magnitudes = np.linalg.norm(vec_pocket_to_red) * np.linalg.norm(vec_pocket_to_white)
    if magnitudes < 1e-10:
      pocket_angle = 0.0
    else:
      cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
      pocket_angle = np.rad2deg(np.arccos(cos_angle))
    
    # Calculate angle at red ball vertex (angle between red->pocket and red->white)
    vec_red_to_pocket = pocket_pos - red_pos
    vec_red_to_white = white_pos - red_pos
    
    dot_product = np.dot(vec_red_to_pocket, vec_red_to_white)
    magnitudes = np.linalg.norm(vec_red_to_pocket) * np.linalg.norm(vec_red_to_white)
    if magnitudes < 1e-10:
      red_angle = 0.0
    else:
      cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
      red_angle = np.rad2deg(np.arccos(cos_angle))
    
    # Calculate angle at white ball vertex (angle between white->pocket and white->red)
    vec_white_to_pocket = pocket_pos - white_pos
    vec_white_to_red = red_pos - white_pos
    
    dot_product = np.dot(vec_white_to_pocket, vec_white_to_red)
    magnitudes = np.linalg.norm(vec_white_to_pocket) * np.linalg.norm(vec_white_to_red)
    if magnitudes < 1e-10:
      white_angle = 0.0
    else:
      cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
      white_angle = np.rad2deg(np.arccos(cos_angle))
    
    return {
        'pocket_angle': pocket_angle,
        'red_angle': red_angle,
        'white_angle': white_angle,
        'total': pocket_angle + red_angle + white_angle
    }

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
      
      # Get pocket center (handle middle pockets which aren't predefined in pooltool)
#      if target_pocket_id in ['lc', 'rc']:
#        if target_pocket_id == 'lc':
#          pocket_center = (0.0, self.table.l / 2.0)
#        else:  # 'rc'
#          pocket_center = (self.table.w, self.table.l / 2.0)
#      else:
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
      
      total_difficulty = (angle_difficulty * 0.6) + (distance_difficulty * 0.4)
      
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
#        if self.verbose:
#          print(f"✓ Found suitable position at difficulty {position_difficulty:.1f}/100 (target: {target_difficulty}/100)")
        info["position_difficulty"] = position_difficulty
        info["current_curriculum_difficulty"] = self.current_curriculum_difficulty
        return obs, info
      
      attempts += 1
    
    # If we couldn't find suitable position, log warning and return last attempt
    if self.verbose:
      print(f"⚠ Could not find position with difficulty <= {target_difficulty:.0f}/100 after {max_attempts} attempts")
      print(f"  Returning position with difficulty {position_difficulty:.0f}/100 (closest found)")
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

    if self.linear_velocity:
        # Map raw_velocity [-1, 1] -> [self.min_cue_speed, self.max_cue_speed]
        v0 = self.min_cue_speed + (raw_velocity + 1.0) / 2.0 * (self.max_cue_speed - self.min_cue_speed)
    else:
        # Normalize action raw_velocity from [-1, 1] to [0, 1]
        normalized_action = (raw_velocity + 1.0) / 2.0

        # Apply quadratic power scaling (gamma = 2.0)
        scaled_ratio = normalized_action ** 2.0

        # Map to physical speed range
        v0 = self.min_cue_speed + scaled_ratio * (self.max_cue_speed - self.min_cue_speed)

    
    

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

    # Reward logic: +1 if red in target pocket; -5 if white is pocketed; else 0
    reward = -5.0 if white_pocketed else (1.0 if red_pocketed_target else 0.0)

    norm_speed = (v0 - self.min_cue_speed) / (self.max_cue_speed - self.min_cue_speed)
    r_speed_penalty = -0.1 * (norm_speed ** 2)
    reward += r_speed_penalty

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
    info["cut_side"] = self.cut_side
    info["target_pocket_id"] = self.target_pocket_id

    # print observation, cut side, target pocket
 
#    print(f"Observation: {observation}, Cut side: {self.cut_side}, Target pocket: {self.target_pocket_id}")


    if self.verbose:
      if self.use_curriculum:
        print(f"Shot {self.shot_counter}: red_pocketed_target={red_pocketed_target}, white_pocketed={white_pocketed}, reward={reward}, speed_penalty={r_speed_penalty}, current_curriculum_difficulty={self.current_curriculum_difficulty:.0f}, position_difficulty={self.calculate_position_difficulty():.0f}")
      else:
        print(f"Shot {self.shot_counter}: red_pocketed_target={red_pocketed_target}, white_pocketed={white_pocketed}, reward={reward}, speed_penalty={r_speed_penalty}, position_difficulty={self.calculate_position_difficulty():.0f}")
    if self.use_curriculum:
      self.update_curriculum_difficulty()

    return observation, reward, terminated, truncated, info


 

  def get_obs_pocket_relative_polar(self):
    """
    Convert pocket-relative Cartesian coordinates to polar coordinates.
    
    Angle is always arctan(y/x) in degrees.
    
    Returns:
        dict: Dictionary with polar coordinates:
              {
                'red_radius': float (meters),
                'red_angle': float (degrees),
                'white_radius': float (meters),
                'white_angle': float (degrees)
              }
    """
    pocket_coords = self._get_obs_pocket_relative()
    red_x, red_y, white_x, white_y = pocket_coords[0], pocket_coords[1], pocket_coords[2], pocket_coords[3]
    
    # Calculate polar coordinates: radius and angle = arctan(y/x)
    red_radius = np.sqrt(red_x**2 + red_y**2)
    red_angle = np.rad2deg(np.arctan(red_y / red_x)) if red_x != 0 else 90.0
    
    white_radius = np.sqrt(white_x**2 + white_y**2)
    white_angle = np.rad2deg(np.arctan(white_y / white_x)) if white_x != 0 else 90.0
    
    return {
        'red_radius': red_radius,
        'red_angle': red_angle,
        'white_radius': white_radius,
        'white_angle': white_angle
    }


  def render(self):
    if self.shots:
      for i, shot in enumerate(self.shots):
        if self.verbose:
          print(f"\nDisplaying Shot {i+1}...")
        pt.show(shot)
    elif self.shot is not None:
      pt.show(self.shot)

  def find_valid_shot_angles(self, white_pos, red_pos, visualize=False):
    """
    Find valid shot angle for the top-left target pocket using mirror image geometry.
    
    For top-left pocket, the cue ball is first played to the RIGHT CUSHION.
    Uses the principle that a shot bouncing off a vertical wall is equivalent
    to shooting directly at the mirror image of the red ball reflected across that wall.
    
    Args:
      white_pos: Tuple (x, y) for white ball position in meters
      red_pos: Tuple (x, y) for red ball position in meters
      visualize: Whether to plot the valid rays (default False)
    
    Returns:
      Dictionary with angle for right cushion shot:
      {
        'right': {'phi': angle, 'cushion_point': (x, y), 'distance': d, 'valid': bool}
      }
    """
    table_width = self.table.w
    ball_radius = self.R
    
    print(f"\n{'='*80}")
    print(f"Finding Valid Shot Angle (Top-Left Pocket → Right Cushion)")
    print(f"{'='*80}")
    print(f"White Ball: ({white_pos[0]:.4f}, {white_pos[1]:.4f})")
    print(f"Red Ball:   ({red_pos[0]:.4f}, {red_pos[1]:.4f})")
    
    results = {'right': None}
    
    # --- RIGHT CUSHION (x=w) ---
    # Mirror red ball across right wall: red' = (2*w - red_x, red_y)
    red_mirrored_right = np.array([2 * table_width - red_pos[0], red_pos[1]])
    white_np = np.array(white_pos)
    
    # Line from white to mirrored red intersects right wall at (w, y_intersect)
    # At x=w: w = white_x + t*(red_mirrored_x - white_x)
    if red_mirrored_right[0] != white_np[0]:  # Avoid division by zero
      t_right = (table_width - white_np[0]) / (red_mirrored_right[0] - white_np[0])
      if 0 < t_right < 1:  # Valid intersection
        y_right = white_np[1] + t_right * (red_mirrored_right[1] - white_np[1])
        if 0 <= y_right <= self.table.l:  # Within table bounds
          cushion_point_right = np.array([table_width, y_right])
          angle_right = np.rad2deg(np.arctan2(y_right - white_np[1], table_width - white_np[0]))
          if angle_right < 0:
            angle_right += 360
          distance_right = np.linalg.norm(cushion_point_right - white_np)
          
          results['right'] = {
              'phi': angle_right,
              'cushion_point': tuple(cushion_point_right),
              'distance': distance_right,
              'valid': True,
          }
          print(f"\nRight Cushion Shot:")
          print(f"  Angle (phi)         : {angle_right:.2f}°")
          print(f"  Cushion Point       : ({cushion_point_right[0]:.4f}, {cushion_point_right[1]:.4f})")
          print(f"  Distance to Cushion : {distance_right:.4f} m")
    
    if results['right'] is None:
      results['right'] = {'phi': None, 'cushion_point': None, 'distance': None, 'valid': False}
      print(f"\nRight Cushion Shot: NOT VALID")
    
    print(f"{'='*80}\n")
    
    # Visualization
    if visualize:
      self._visualize_mirror_angles(white_pos, red_pos, results)
    
    return results

  def calculate_cushion_shot_angle(self, white_pos, red_pos):
    """
    Calculate aim angle for a one-cushion shot using mirror geometry.
    
    For top-left target pocket, uses the RIGHT CUSHION via mirror geometry.
    
    Args:
      white_pos: Tuple (x, y) for white ball position in meters
      red_pos: Tuple (x, y) for red ball position in meters
    
    Returns:
      Dictionary with shot information:
      {
        'aim_angle_deg': calculated angle,
        'cushion': 'right',
        'cushion_point': (x, y),
        'distance': distance to cushion,
        'valid': True/False
      }
    """
    print(f"\n{'='*80}")
    print(f"Calculating One-Cushion Shot Angle")
    print(f"{'='*80}")
    print(f"Target Pocket: {self.target_pocket_id.upper()}")
    print(f"White Ball: ({white_pos[0]:.4f}, {white_pos[1]:.4f})")
    print(f"Red Ball:   ({red_pos[0]:.4f}, {red_pos[1]:.4f})")
    
    # Get valid angle for right cushion shot
    results = self.find_valid_shot_angles(white_pos, red_pos, visualize=False)
    
    # Check if right cushion shot is valid
    right_valid = results['right']['valid']
    
    if not right_valid:
      print("ERROR: Right cushion shot is not valid!")
      return {
          'aim_angle_deg': self.aim_angle_deg,
          'cushion': 'right',
          'cushion_point': None,
          'distance': None,
          'valid': False,
      }
    
    # Use right cushion shot
    chosen_cushion = 'right'
    chosen_angle = results[chosen_cushion]['phi']
    cushion_info = results[chosen_cushion]
    
    # Update self.aim_angle_deg
    self.aim_angle_deg = chosen_angle
    
    print(f"\nChosen Cushion: {chosen_cushion.upper()}")
    print(f"  Aim Angle (phi)     : {chosen_angle:.2f}°")
    print(f"  Cushion Point       : ({cushion_info['cushion_point'][0]:.4f}, {cushion_info['cushion_point'][1]:.4f})")
    print(f"  Distance to Cushion : {cushion_info['distance']:.4f} m")
    print(f"{'='*80}\n")
    
    return {
        'aim_angle_deg': chosen_angle,
        'cushion': chosen_cushion,
        'cushion_point': cushion_info['cushion_point'],
        'distance': cushion_info['distance'],
        'valid': True,
    }

  def _visualize_mirror_angles(self, white_pos, red_pos, results):
    """Visualize the mirror geometry and valid shot angles."""
    try:
      import matplotlib.pyplot as plt
      import matplotlib.patches as patches
    except ImportError:
      print("matplotlib not available, skipping visualization")
      return
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    
    # Draw table
    table_rect = patches.Rectangle((0, 0), self.table.w, self.table.l, 
                                   linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.3)
    ax.add_patch(table_rect)
    
    white_np = np.array(white_pos)
    red_np = np.array(red_pos)
    
    # Draw mirrored red balls (faded)
    red_mirrored_left = np.array([-red_pos[0], red_pos[1]])
    red_mirrored_right = np.array([2*self.table.w - red_pos[0], red_pos[1]])
    
    left_circle = patches.Circle(red_mirrored_left, self.R, color='red', ec='red', linewidth=1, alpha=0.2, linestyle='--')
    ax.add_patch(left_circle)
    ax.text(red_mirrored_left[0], red_mirrored_left[1]-0.08, "R'", ha='center', va='center', fontsize=9, alpha=0.5)
    
    right_circle = patches.Circle(red_mirrored_right, self.R, color='red', ec='red', linewidth=1, alpha=0.2, linestyle='--')
    ax.add_patch(right_circle)
    ax.text(red_mirrored_right[0], red_mirrored_right[1]-0.08, "R'", ha='center', va='center', fontsize=9, alpha=0.5)
    
    # Draw lines from white to mirrored red balls
    ax.plot([white_np[0], red_mirrored_left[0]], [white_np[1], red_mirrored_left[1]], 
           'r-', linewidth=1, alpha=0.3, label="Mirror lines")
    ax.plot([white_np[0], red_mirrored_right[0]], [white_np[1], red_mirrored_right[1]], 
           'b-', linewidth=1, alpha=0.3)
    
    # Draw left cushion shot
    if results['left']['valid']:
      cushion_point = np.array(results['left']['cushion_point'])
      ax.plot([white_np[0], cushion_point[0]], [white_np[1], cushion_point[1]], 
             'r-', linewidth=2.5, alpha=0.8, label=f"Left: {results['left']['phi']:.1f}°")
      ax.plot(cushion_point[0], cushion_point[1], 'ro', markersize=6, alpha=0.8)
      
      # Draw reflected path to red ball
      ax.plot([cushion_point[0], red_np[0]], [cushion_point[1], red_np[1]], 
             'r--', linewidth=2, alpha=0.6)
    
    # Draw right cushion shot
    if results['right']['valid']:
      cushion_point = np.array(results['right']['cushion_point'])
      ax.plot([white_np[0], cushion_point[0]], [white_np[1], cushion_point[1]], 
             'b-', linewidth=2.5, alpha=0.8, label=f"Right: {results['right']['phi']:.1f}°")
      ax.plot(cushion_point[0], cushion_point[1], 'bo', markersize=6, alpha=0.8)
      
      # Draw reflected path to red ball
      ax.plot([cushion_point[0], red_np[0]], [cushion_point[1], red_np[1]], 
             'b--', linewidth=2, alpha=0.6)
    
    # Draw white ball
    white_circle = patches.Circle(white_pos, self.R, color='white', ec='black', linewidth=2, zorder=5)
    ax.add_patch(white_circle)
    ax.text(white_pos[0], white_pos[1]-0.08, 'W', ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw red ball
    red_circle = patches.Circle(red_pos, self.R, color='red', ec='darkred', linewidth=2, zorder=5)
    ax.add_patch(red_circle)
    ax.text(red_pos[0], red_pos[1]-0.08, 'R', ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw cushion walls with emphasis
    ax.axvline(x=0, color='black', linewidth=3, alpha=0.5, label='Cushions')
    ax.axvline(x=self.table.w, color='black', linewidth=3, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlim(-0.3, self.table.w + 0.3)
    ax.set_ylim(-0.1, self.table.l + 0.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title(f'Valid Shot Angles (Mirror Geometry)\nLeft & Right Cushion Shots', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('valid_shot_angles.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to 'valid_shot_angles.png'")
    plt.show()

# --- Demo Usage ---
if __name__ == "__main__":
  """Simple demonstration of the RLBilliard_pocket_invariant environment."""
  
  # Create environment
  env = RLBilliard_pocket_invariant(
      use_curriculum=False,
      fixed_difficulty=100,
      target_pocket_id='lc',
      verbose=True
  )

  
  
  # Initialize environment
  print("Resetting environment...")
  obs, info = env.reset()
  print(f"Initial observation shape: {obs.shape}")
  
  # Execute a sample step
  print("\nExecuting sample action...")
  action = env.action_space.sample()
  action[0] = 0.0
  obs, reward, terminated, truncated, info = env.step(action)
  print(f"Action: {action}")
  print(f"Reward: {reward}")
  print(f"Terminated: {terminated}, Truncated: {truncated}")
  print(f"Info: {info}")
  
  # Render the shot
  print("\nRendering shot...")
  env.render()