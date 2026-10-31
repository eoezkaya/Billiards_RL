#!/usr/bin/env python3

import os
from RLEnv_outer import RLBilliard_outer
from stable_baselines3 import DQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def play_game(env, model, num_episodes=20, render=True):
  """
  Play the pocket selection game using a trained DQN model.
  
  Args:
      env (RLBilliard_outer): The outer environment
      model (DQN): Trained DQN model for pocket selection
      num_episodes (int): Number of episodes to play (default: 20)
      render (bool): Whether to render each game (default: True)
  """
  
  print(f"\n{'='*70}")
  print(f"Playing Billiards Pocket Selection Game")
  print(f"Episodes: {num_episodes}")
  print(f"{'='*70}\n")
  
  total_rewards = 0
  successful_shots = 0
  pocket_selections = {'lt': 0, 'rt': 0, 'lb': 0, 'rb': 0}
  episode_rewards = []
  
  for episode in range(1, num_episodes + 1):
    # Reset environment for new episode
    obs, info = env.reset()
    episode_reward = 0
    
    # Agent predicts which pocket to aim for
    action, _states = model.predict(obs, deterministic=True)
    
    # Execute the shot
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    
    # Track statistics
    total_rewards += episode_reward
    episode_rewards.append(episode_reward)
    
    target_pocket = info.get("target_pocket", "unknown")
    pocket_id = info.get("pocket_id", "unknown")
    
    action_to_pocket = {
        "top left": "lt",
        "top right": "rt",
        "bottom left": "lb",
        "bottom right": "rb",
    }
    
    if target_pocket in action_to_pocket:
      pocket_key = action_to_pocket[target_pocket]
      pocket_selections[pocket_key] += 1
      pocket_str = target_pocket.replace(" ", "-").title()
    else:
      pocket_str = "Unknown"
    
    is_success = episode_reward > 0.0
    if is_success:
      successful_shots += 1
      result_str = "✓ SUCCESS"
    else:
      result_str = "✗ FAILURE"
    
    print(f"Episode {episode:2d}: {result_str} | Pocket: {pocket_str:12s} | Reward: {episode_reward:.1f}")
    
    # Render the game
    if render:
      env.render()
  
  # Calculate statistics
  success_rate = (successful_shots / num_episodes) * 100
  mean_reward = total_rewards / num_episodes
  lt_percentage = (pocket_selections['lt'] / num_episodes) * 100
  rt_percentage = (pocket_selections['rt'] / num_episodes) * 100
  lb_percentage = (pocket_selections['lb'] / num_episodes) * 100
  rb_percentage = (pocket_selections['rb'] / num_episodes) * 100
  
  # Print summary
  print(f"\n{'='*70}")
  print(f"GAME STATISTICS ({num_episodes} Episodes)")
  print(f"{'='*70}")
  print(f"  Total Reward: {total_rewards:.1f}")
  print(f"  Mean Reward per Episode: {mean_reward:.4f}")
  print(f"  Successful Shots: {successful_shots}/{num_episodes}")
  print(f"  Success Rate: {success_rate:.1f}%")
  print(f"\n  Pocket Selection Distribution:")
  print(f"    Top-Left (lt):     {pocket_selections['lt']:2d} ({lt_percentage:5.1f}%)")
  print(f"    Top-Right (rt):    {pocket_selections['rt']:2d} ({rt_percentage:5.1f}%)")
  print(f"    Bottom-Left (lb):  {pocket_selections['lb']:2d} ({lb_percentage:5.1f}%)")
  print(f"    Bottom-Right (rb): {pocket_selections['rb']:2d} ({rb_percentage:5.1f}%)")
  print(f"{'='*70}\n")


def main():
  print(f"\n{'='*70}")
  print(f"Loading Trained DQN Model")
  print(f"{'='*70}")
  
  # Path to trained model
  model_path = "./models/dqn_billiards_outer/dqn_billiards_outer_final.zip"
  
  if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    print(f"Please run train_outer.py first to train the model.")
    return
  
  print(f"Loading model from: {model_path}")
  model = DQN.load(model_path)
  print(f"✓ Model loaded successfully\n")
  
  # Create environment
  print(f"Initializing environment...")
  env = RLBilliard_outer(verbose=False)
  print(f"✓ Environment initialized\n")
  
  # Play the game
  play_game(env, model, num_episodes=20, render=True)
  
  # Close environment
  env.close()
  print(f"✓ Game completed!\n")


if __name__ == "__main__":
  main()
