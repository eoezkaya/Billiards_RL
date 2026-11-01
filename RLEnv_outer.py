#! /usr/bin/env python

import copy
import os
from RLEnv import RLBilliard
import gymnasium as gym
import numpy as np

import pooltool as pt
from gymnasium import spaces
from stable_baselines3 import PPO



class RLBilliard_outer(gym.Env):
  """Custom Gymnasium Environment for a 2-ball billiards pocketing task using pooltool."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, render_mode=None,
               red_x_range=(0.1, 0.9), red_y_range=(0.1, 1.9),
               white_x_range=(0.1, 0.9), white_y_range=(0.1, 1.9),
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

    # Action Space: Categorical with 6 actions
    # Action 0: play_top_left_pocket (lt)
    # Action 1: play_top_right_pocket (rt)
    # Action 2: play_bottom_left_pocket (lb)
    # Action 3: play_bottom_right_pocket (rb)
    # Action 4: play_center_right_pocket (rc)
    # Action 5: play_center_left_pocket (lc)
    self.action_space = spaces.Discrete(6)
    # Observation Space: (Matches normalized output in _get_obs)
    # [red_x, red_y, white_x, white_y, difficult_top_left, difficult_top_right, difficult_bottom_left, difficult_bottom_right, difficult_center_left, difficult_center_right]
    self.observation_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    self.shot = None
    self.shots = []  # Store multiple shots for continuous display
    self.shot_counter = 0  # Track total number of shots simulated
    self.observation = None  # Store current observation
    self.current_env = None  # Current environment (lt or rt)

    self.current_env = RLBilliard(red_x_range=red_x_range, red_y_range=red_y_range,
                                                   white_x_range=white_x_range, white_y_range=white_y_range,
                                                   verbose=False, use_curriculum=False, 
                                                   training_corner_pocket=False, training_center_pocket=False, 
                                                   inference = True, fixed_difficulty=100, target_pocket_id="lt")
    self.model_path_corner_pockets = "./models/ppo_billiards/ppo_billiards_final_corner_pockets.zip"
    self.model_path_center_pockets = "./models/ppo_billiards/ppo_billiards_final_center_pockets.zip"

    if not os.path.exists(self.model_path_corner_pockets):
        raise FileNotFoundError(
            f"No trained model found at {self.model_path_corner_pockets} Run train.py first!"
        )
    print(f"Loading trained model from: {self.model_path_corner_pockets}")
    self.model_corner_pockets = PPO.load(self.model_path_corner_pockets)
    if not os.path.exists(self.model_path_center_pockets):
        raise FileNotFoundError(
            f"No trained model found at {self.model_path_center_pockets} Run train.py first!"
        )
    print(f"Loading trained model from: {self.model_path_center_pockets}")
    self.model_center_pockets = PPO.load(self.model_path_center_pockets)
   

  def _get_obs(self):
    red_pos = self.current_env.balls["red"].state.rvw[0][:2]
    white_pos = self.current_env.balls["white"].state.rvw[0][:2]

    difficult_top_left = self.current_env.calculate_position_difficulty("lt")
    difficult_top_right = self.current_env.calculate_position_difficulty("rt")
    difficult_bottom_left = self.current_env.calculate_position_difficulty("lb")
    difficult_bottom_right = self.current_env.calculate_position_difficulty("rb")
    difficult_center_left = self.current_env.calculate_position_difficulty("lc")
    difficult_center_right = self.current_env.calculate_position_difficulty("rc")

    return np.array(
        [
            red_pos[0] / self.table.w,
            red_pos[1] / self.table.l,
            white_pos[0] / self.table.w,
            white_pos[1] / self.table.l,
            difficult_top_left/100.0,
            difficult_top_right/100.0,
            difficult_bottom_left/100.0,
            difficult_bottom_right/100.0,
            difficult_center_left/100.0,
            difficult_center_right/100.0,
        ],
        dtype=np.float32,
    )

  def reset(self, seed=None, options=None):
    """
    Reset the environment to a new episode.
    
    Args:
        seed (int, optional): Random seed for reproducibility
        options (dict, optional): Additional options (unused)
    
    Returns:
        tuple: (observation, info)
    """
    super().reset(seed=seed)

    self.current_env.reset()
    self.observation = self._get_obs()
    info = {}
    return self.observation, info



  def step(self, action):
    """
    Execute a single shot towards the selected pocket.
    
    Args:
        action (int): Categorical action - 0 (top-left), 1 (top-right), 2 (bottom-left), 3 (bottom-right)
    
    Returns:
        tuple: (observation, reward, terminated, truncated, info)
    """
    # Convert numpy array to Python int if needed
    if isinstance(action, np.ndarray):
      action = int(action.item())
    else:
      action = int(action)
    
    # Validate action is in action space
    assert self.action_space.contains(action), f"Invalid action: {action}"
    
    # Map action to pocket
    action_to_pocket = {
        0: 'lt',  # top-left
        1: 'rt',  # top-right
        2: 'lb',  # bottom-left
        3: 'rb',  # bottom-right
        4: 'rc',  # center right
        5: 'lc',  # center left
    }
    
    action_to_name = {
        0: 'top left',
        1: 'top right',
        2: 'bottom left',
        3: 'bottom right',
        4: 'center right',
        5: 'center left',
    }
    
    target_pocket = action_to_name[action]
    pocket_id = action_to_pocket[action]
    
    # Determine which model to use based on pocket type
    # Center pockets: 4 (rc) and 5 (lc)
    is_center_pocket = action in [4, 5]
    model = self.model_center_pockets if is_center_pocket else self.model_corner_pockets
    
    # Set target pocket and get observation for the trained model
    self.current_env.set_target_pocket(pocket_id)
    observation_for_action = self.current_env._get_obs()
    predicted_action, _states = model.predict(observation_for_action, deterministic=True)
    
    # Execute the predicted action
    obs, reward, terminated, truncated, info = self.current_env.step(predicted_action)
    
    # Update observation
    self.observation = self._get_obs()
    
    info["target_pocket"] = target_pocket
    info["pocket_id"] = pocket_id
    info["reward"] = reward
   
    return self.observation, reward, terminated, truncated, info


  def render(self):
    if self.current_env is not None:
      self.current_env.render()


def test_with_selected_action(env, action, num_games=10):
  """
  Test method that plays multiple games with a selected action and renders each game.
  
  Args:
      env (RLBilliard_outer): The environment to test
      action (int): Categorical action - 0 for top-left pocket, 1 for top-right pocket, 2 for bottom-left, 3 for bottom-right, 4 for center-right, 5 for center-left
      num_games (int): Number of games to play (default: 10)
  """
  pocket_name = {
      0: "Top-Left",
      1: "Top-Right",
      2: "Bottom-Left",
      3: "Bottom-Right",
      4: "Center-Right",
      5: "Center-Left"
  }.get(action, "Unknown")
  print(f"\n{'='*70}")
  print(f"Testing {pocket_name} Pocket Selection - Playing {num_games} Games")
  print(f"{'='*70}\n")
  
  successes = 0
  failures = 0
  
  for game_num in range(1, num_games + 1):
    # Reset environment for new game
    obs, info = env.reset()
    
    # Execute single shot with selected action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Track success/failure
    if reward > 0.0:
      successes += 1
      result_str = "✓ SUCCESS"
    else:
      failures += 1
      result_str = "✗ FAILURE"
    
    print(f"Game {game_num:2d}: {result_str} | Red pocketed: {reward > 0.0}")
    
    # Render the shot
    env.render()
  
  # Print summary statistics
  success_rate = (successes / num_games) * 100
  print(f"\n{'='*70}")
  print(f"Summary for {pocket_name} Pocket:")
  print(f"  Total Games: {num_games}")
  print(f"  Successes: {successes}")
  print(f"  Failures: {failures}")
  print(f"  Success Rate: {success_rate:.1f}%")
  print(f"{'='*70}\n")


# --- Test Run ---
if __name__ == "__main__":
  env = RLBilliard_outer(verbose=True)
  
  # Test environment
  obs, info = env.reset()
  print(f"Initial observation: {obs}")
  print(f"Action space: {env.action_space}")
  
  # Test with selected actions (10 games each)
  test_with_selected_action(env, action=0, num_games=10)  # Test Top-Left pocket
  test_with_selected_action(env, action=1, num_games=10)  # Test Top-Right pocket
  test_with_selected_action(env, action=2, num_games=10)  # Test Bottom-Left pocket
  test_with_selected_action(env, action=3, num_games=10)  # Test Bottom-Right pocket
  test_with_selected_action(env, action=4, num_games=10)  # Test Center-Right pocket
  test_with_selected_action(env, action=5, num_games=10)  # Test Center-Left pocket


