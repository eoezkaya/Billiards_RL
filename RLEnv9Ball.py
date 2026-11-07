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

class RLBilliard9Ball(gym.Env):
  """Custom Gymnasium Environment for a 9-ball billiards pocketing task using pooltool."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, render_mode=None,
               x_range=red_x_range, y_range=red_y_range,
               target_pocket_id='lt',
               training_corner_pocket=True, training_center_pocket=False, inference=False,
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
    self.x_range = x_range
    self.y_range = y_range
    self.play_cushion = False
    self.train_corner_pocket = training_corner_pocket
    self.train_center_pocket = training_center_pocket
    self.inference = inference

    # Target pocket: can be 'lb', 'lt', 'rb', 'rt', 'ml', or 'mr'
    if target_pocket_id not in ['lb', 'lt', 'rb', 'rt', 'lc', 'rc']:
      raise ValueError(f"target_pocket_id must be 'lb', 'lt', 'rb', 'rt', 'lc', or 'rc', got '{target_pocket_id}'")
    
    self.target_pocket_id = target_pocket_id
    self.target_ball_id = '1'  # Default target ball ID, can be changed as needed
    
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
          'ml': 'Middle-Left', 'mr': 'Middle-Right', 'lc': 'Middle-Left', 'rc': 'Middle-Right'
      }
      print(f"Target pocket changed to: {pocket_names[pocket_id]} ({pocket_id})")

    # update pocket and related coordinates
    self.pocket_coordinates = self._get_obs_pocket_relative()
    self.pocket_polar_coords = self.get_obs_pocket_relative_polar()

    target_ball_pos = np.array(self.balls[self.target_ball_id].state.rvw[0][:2])
    cue_ball_pos = np.array(self.balls["cue"].state.rvw[0][:2])

        
    # Calculate cut side again based on updated target pocket
    self.cut_side, self.cut_angle = self._determine_cut_side(
      (cue_ball_pos[0], cue_ball_pos[1]), (target_ball_pos[0], target_ball_pos[1]))



  def _calculate_aim_angle(self, cue_ball_pos, target_ball_pos):
    """
    Calculates the aim angle (in degrees) from the cue ball to the target ball.
    
    Args:
        cue_ball_pos: Position of cue ball [x, y]
        target_ball_pos: Position of target ball [x, y]
    
    Returns:
        float: Angle in degrees [0, 360), where 0° is to the right, 90° is up, etc.
    """
    # Vector from cue ball to target ball
    v_cue = np.array(target_ball_pos) - np.array(cue_ball_pos)
    
    # Calculate angle using arctan2 (returns angle in radians from -pi to pi)
    angle_rad = np.arctan2(v_cue[1], v_cue[0])
    
    # Convert to degrees and normalize to [0, 360)
    aim_angle_deg = np.rad2deg(angle_rad)
    if aim_angle_deg < 0:
      aim_angle_deg += 360.0
    
    return aim_angle_deg

  def _determine_cut_side(self, cue_ball_pos, target_ball_pos):
    """Calculates cut side and cut angle between cue vector and pocket vector."""
    v_cue = np.array(target_ball_pos) - np.array(cue_ball_pos)
    v_pocket = self.target_pocket_pos - np.array(target_ball_pos)

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
    target_angle_norm = triangle_angles['target_angle'] / 180.0
    cue_angle_norm = triangle_angles['cue_angle'] / 180.0
    
    # Get white ball (cue ball) position
    cue_ball_pos = self.balls["cue"].state.rvw[0][:2]
    
    # Calculate distances to cushions based on target pocket
    # For middle pockets: measure distance to closest relevant cushion
    if self.target_pocket_id in ['lb', 'lt']:
        # Left pockets: distance to left cushion
        dist_to_vertical_cushion = cue_ball_pos[0]
    elif self.target_pocket_id in ['rb', 'rt']:
        # Right pockets: distance to right cushion
        dist_to_vertical_cushion = self.table.w - cue_ball_pos[0]
    elif self.target_pocket_id == 'lc':
        # Middle-left: distance to left cushion
        dist_to_vertical_cushion = cue_ball_pos[0]
    elif self.target_pocket_id == 'rc':
        # Middle-right: distance to right cushion
        dist_to_vertical_cushion = self.table.w - cue_ball_pos[0]
    
    if self.target_pocket_id in ['lt', 'rt']:
        # Top pockets: distance to top cushion
        dist_to_horizontal_cushion = self.table.l - cue_ball_pos[1]
    elif self.target_pocket_id in ['lb', 'rb']:
        # Bottom pockets: distance to bottom cushion
        dist_to_horizontal_cushion = cue_ball_pos[1]
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
            target_angle_norm,
            cue_angle_norm,
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
    cue_ball_pos = self.balls["cue"].state.rvw[0][:2]
    target_ball_pos = np.array(self.balls[self.target_ball_id].state.rvw[0][:2])
    
    # Transform to pocket-relative coordinates based on target pocket
    if self.target_pocket_id == 'lb':
        # Bottom-left: no transformation needed, pocket is at origin
        red_x_pocket = target_ball_pos[0]
        red_y_pocket = target_ball_pos[1]
        white_x_pocket = cue_ball_pos[0]
        white_y_pocket = cue_ball_pos[1]
    
    elif self.target_pocket_id == 'rb':
        # Bottom-right: flip x-coordinates (pocket at right edge)
        red_x_pocket = self.table.w - target_ball_pos[0]
        red_y_pocket = target_ball_pos[1]
        white_x_pocket = self.table.w - cue_ball_pos[0]
        white_y_pocket = cue_ball_pos[1]
    
    elif self.target_pocket_id == 'lt':
        # Top-left: flip y-coordinates (pocket at top edge)
        red_x_pocket = target_ball_pos[0]
        red_y_pocket = self.table.l - target_ball_pos[1]
        white_x_pocket = cue_ball_pos[0]
        white_y_pocket = self.table.l - cue_ball_pos[1]
    
    elif self.target_pocket_id == 'rt':
        # Top-right: flip both x and y coordinates (pocket at top-right corner)
        red_x_pocket = self.table.w - target_ball_pos[0]
        red_y_pocket = self.table.l - target_ball_pos[1]
        white_x_pocket = self.table.w - cue_ball_pos[0]
        white_y_pocket = self.table.l - cue_ball_pos[1]
    
    elif self.target_pocket_id == 'lc':
        # Left-center: pocket at left edge center, no x-flip, y stays as is
        red_x_pocket = target_ball_pos[0]
        red_y_pocket = target_ball_pos[1] - self.table.l / 2.0
        white_x_pocket = cue_ball_pos[0]
        white_y_pocket = cue_ball_pos[1] - self.table.l / 2.0
    
    elif self.target_pocket_id == 'rc':
        # Right-center: pocket at right edge center, flip x, y distance from center
        red_x_pocket = self.table.w - target_ball_pos[0]
        red_y_pocket = target_ball_pos[1] - self.table.l / 2.0
        white_x_pocket = self.table.w - cue_ball_pos[0]
        white_y_pocket = cue_ball_pos[1] - self.table.l / 2.0
    
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

  def _reset_9ball_table(self):
    """
    Internal method to reset the pool table with random ball placement.
    Randomly places all balls (cue and numbered balls 1-9) on the table
    ensuring they don't overlap or touch each other.
    """
    self.balls = pt.get_rack(pt.GameType.NINEBALL, self.table)
    
    # Minimum distance between ball centers (sum of radii)
    min_distance = 2.0 * self.R
    
    # Collect ball positions as they're placed
    placed_positions = []
    
    # Randomly place each ball, ensuring no overlap with previously placed balls
    for ball_id in self.balls.keys():
        max_attempts = 100
        attempt = 0
        placed = False
        
        while attempt < max_attempts and not placed:
            x = self.np_random.uniform(self.x_range[0], self.x_range[1])
            y = self.np_random.uniform(self.y_range[0], self.y_range[1])
            
            # Check if this position overlaps with any previously placed ball
            valid_position = True
            for placed_x, placed_y in placed_positions:
                distance = np.sqrt((x - placed_x)**2 + (y - placed_y)**2)
                if distance < min_distance:
                    valid_position = False
                    break
            
            # If position is valid, place the ball
            if valid_position:
                self.balls[ball_id].state.rvw[0] = [x, y, self.R]
                placed_positions.append((x, y))
                placed = True
            
            attempt += 1
        
        # If we couldn't find a valid position after max_attempts, place anyway
        # (this prevents infinite loops on very constrained tables)
        if not placed:
            x = self.np_random.uniform(self.x_range[0], self.x_range[1])
            y = self.np_random.uniform(self.y_range[0], self.y_range[1])
            self.balls[ball_id].state.rvw[0] = [x, y, self.R]
            placed_positions.append((x, y))


  def reset(self, seed=None, options=None):
    
    super().reset(seed=seed)

    self._reset_9ball_table()
    
    # Update pocket and related coordinates after resetting the table
    self.pocket_coordinates = self._get_obs_pocket_relative()
    self.pocket_polar_coords = self.get_obs_pocket_relative_polar()
    
    # Calculate cut side and aim angle based on current ball positions
    target_ball_pos = np.array(self.balls[self.target_ball_id].state.rvw[0][:2])
    cue_ball_pos = np.array(self.balls["cue"].state.rvw[0][:2])
    self.cut_side, self.cut_angle = self._determine_cut_side(
        (cue_ball_pos[0], cue_ball_pos[1]), (target_ball_pos[0], target_ball_pos[1]))
    self.aim_angle_deg = self._calculate_aim_angle(
        (cue_ball_pos[0], cue_ball_pos[1]), (target_ball_pos[0], target_ball_pos[1]))
    
    info = {}
    obs = None

    return obs, info

  
  def _calculate_max_cut_angle_deg(self, cue_ball_pos, target_ball_pos):
    """Calculates the maximum angle deviation (in degrees) for tangent contact."""
    d = np.hypot(target_ball_pos[0] - cue_ball_pos[0], target_ball_pos[1] - cue_ball_pos[1])

    # Ball contact radius threshold
    ratio = (2.0 * self.R) / d

    # Clip ratio to [0.0, 1.0] to prevent numerical issues with arcsin
    ratio = np.clip(ratio, 0.0, 1.0)

    theta_max_rad = np.arcsin(ratio)
    return np.rad2deg(theta_max_rad) + 0.1

  def _calculate_pocket_angle(self, cue_ball_pos, target_ball_pos, pocket_pos):
    """
    Calculates the angle between pocket-to-cue and pocket-to-target vectors.
    
    Args:
        cue_ball_pos: Position of cue ball [x, y]
        target_ball_pos: Position of target ball [x, y]
        pocket_pos: Position of target pocket [x, y]
    
    Returns:
        float: Angle in degrees between the two vectors
    """
    # Create vectors from pocket to each ball
    vec_to_cue = np.array(cue_ball_pos) - np.array(pocket_pos)
    vec_to_target = np.array(target_ball_pos) - np.array(pocket_pos)
    
    # Calculate angle using dot product
    dot_product = np.dot(vec_to_cue, vec_to_target)
    magnitudes = np.linalg.norm(vec_to_cue) * np.linalg.norm(vec_to_target)
    cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
    angle_between_deg = np.rad2deg(np.arccos(cos_angle))
    
    return angle_between_deg

  def _calculate_target_angle(self, cue_ball_pos, target_ball_pos, pocket_pos):
    """
    Calculates the angle between pocket->target and target->cue vectors.
    This represents how well-aligned the cue path is for hitting the target ball into the pocket.
    
    Args:
        cue_ball_pos: Position of cue ball [x, y]
        target_ball_pos: Position of target ball [x, y]
        pocket_pos: Position of target pocket [x, y]
    
    Returns:
        float: Angle in degrees between the two vectors
                0° = best angle (straight shot to pocket)
                90° = worst angle (perpendicular)
    """
    # Vector from pocket to target ball
    vec_pocket_to_target = np.array(target_ball_pos) - np.array(pocket_pos)
    # Vector from target ball to cue ball (cue approach)
    vec_target_to_cue = np.array(cue_ball_pos) - np.array(target_ball_pos)
    
    # Calculate angle using dot product
    dot_product = np.dot(vec_pocket_to_target, vec_target_to_cue)
    magnitudes = np.linalg.norm(vec_pocket_to_target) * np.linalg.norm(vec_target_to_cue)
    
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
    target_ball_pos = np.array(self.balls[self.target_ball_id].state.rvw[0][:2])
    cue_ball_pos = np.array(self.balls["cue"].state.rvw[0][:2])
    
    # Get pocket center (handle middle pockets which aren't predefined in pooltool)
    pocket_center = self.table.pockets[target_pocket_id].center
    
    pocket_pos = np.array([float(pocket_center[0]), float(pocket_center[1])])
    
    # Calculate angle at pocket vertex (angle between pocket->red and pocket->white)
    vec_pocket_to_target = target_ball_pos - pocket_pos
    vec_pocket_to_cue= cue_ball_pos - pocket_pos
    
    dot_product = np.dot(vec_pocket_to_target, vec_pocket_to_cue)
    magnitudes = np.linalg.norm(vec_pocket_to_target) * np.linalg.norm(vec_pocket_to_cue)
    if magnitudes < 1e-10:
      pocket_angle = 0.0
    else:
      cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
      pocket_angle = np.rad2deg(np.arccos(cos_angle))
    
    # Calculate angle at target ball vertex (angle between target->pocket and target->cue)
    vec_target_to_pocket = pocket_pos - target_ball_pos
    vec_target_to_cue = cue_ball_pos - target_ball_pos
    
    dot_product = np.dot(vec_target_to_pocket, vec_target_to_cue)
    magnitudes = np.linalg.norm(vec_target_to_pocket) * np.linalg.norm(vec_target_to_cue)
    if magnitudes < 1e-10:
      target_ball_angle = 0.0
    else:
      cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
      target_ball_angle = np.rad2deg(np.arccos(cos_angle))
    
    # Calculate angle at white ball vertex (angle between white->pocket and white->red)
    vec_cue_to_pocket = pocket_pos - cue_ball_pos
    vec_cue_to_target = target_ball_pos - cue_ball_pos
    
    dot_product = np.dot(vec_cue_to_pocket, vec_cue_to_target)
    magnitudes = np.linalg.norm(vec_cue_to_pocket) * np.linalg.norm(vec_cue_to_target)
    if magnitudes < 1e-10:
      cue_ball_angle = 0.0
    else:
      cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
      cue_ball_angle = np.rad2deg(np.arccos(cos_angle))
    
    return {
        'pocket_angle': pocket_angle,
        'target_angle': target_ball_angle,
        'cue_angle': cue_ball_angle,
        'total': pocket_angle + target_ball_angle + cue_ball_angle
    }

  def calculate_position_difficulty(self, target_pocket_id=None, visualize=False):
      """
      Calculates the difficulty of the current position on a scale of 0-100.
      
      Difficulty depends on two factors:
      1. Target angle: Angle between pocket->target and target->cue vectors
        0° = best angle (straight shot), 90° = worst angle (perpendicular)
      2. Distance to target ball: Closer distances are easier, farther distances are harder
      
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
      target_ball_pos = np.array(self.balls[self.target_ball_id].state.rvw[0][:2])
      cue_ball_pos = np.array(self.balls["cue"].state.rvw[0][:2])
      
      # Get pocket center (handle middle pockets which aren't predefined in pooltool)
      pocket_center = self.table.pockets[target_pocket_id].center
      
      pocket_center = np.array([float(pocket_center[0]), float(pocket_center[1])])
      
      # Calculate red angle (pocket->target vs target->cue)
      target_ball_angle = self._calculate_target_angle(cue_ball_pos, target_ball_pos, pocket_center)
      
      # Calculate distance from pocket to red ball
      distance_to_target = np.linalg.norm(np.array(target_ball_pos) - pocket_center)
      
      # Normalize angle: 0° = 0 difficulty, 90° = max difficulty
      # Normalize to 0-100 scale
      angle_difficulty = (target_ball_angle / 90.0) * 100.0
      
      # Normalize distance: 
      # Maximum distance on table is diagonal from corner to corner
      max_distance = np.linalg.norm(np.array([self.table.w, self.table.l]))
      # Closer to pocket is easier (lower difficulty), farther is harder
      distance_difficulty = (distance_to_target / max_distance) * 100.0
      
      total_difficulty = (angle_difficulty * 0.6) + (distance_difficulty * 0.4)
      
      # Clip to 0-100 range
      total_difficulty = np.clip(total_difficulty, 0.0, 100.0)
      
      return total_difficulty


  def step(self, action):

    raw_angle = np.clip(action[0], -1.0, 1.0)
    raw_velocity = np.clip(action[1], -1.0, 1.0)

    # Get current ball positions
    
    target_ball_pos = self.balls[self.target_ball_id].state.rvw[0][:2]
    cue_ball_pos = self.balls["cue"].state.rvw[0][:2]

  # 1. Compute dynamic max angle for tangent contact
    theta_max_deg = self._calculate_max_cut_angle_deg(cue_ball_pos, target_ball_pos)

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
    cue = pt.Cue(cue_ball_id="cue")
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
    info["position_difficulty"] = self.calculate_position_difficulty()
    info["cut_side"] = self.cut_side
    info["target_pocket_id"] = self.target_pocket_id

    # print observation, cut side, target pocket
 
#    print(f"Observation: {observation}, Cut side: {self.cut_side}, Target pocket: {self.target_pocket_id}")


    if self.verbose:
      print(f"Shot {self.shot_counter}: red_pocketed_target={red_pocketed_target}, white_pocketed={white_pocketed}, reward={reward:.3f}, speed_penalty={r_speed_penalty:.3f}, position_difficulty={self.calculate_position_difficulty():.0f}")

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

  
# --- Demo Usage ---
if __name__ == "__main__":
  """Simple demonstration of the RLBilliard environment."""
  
  # Create environment
  env = RLBilliard9Ball(
      training_center_pocket=True,
      target_pocket_id='lt',
      verbose=True
  )

  env.reset()
  env.step([0.0, 0.0])  # Example step with zero action
  env.render()  # Display the current shot

  
  