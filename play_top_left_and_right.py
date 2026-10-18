#!/usr/bin/env python3

import os
import time
import numpy as np
from stable_baselines3 import PPO

from RLEnv import RLBilliard


def transform_obs_for_rt_pocket(obs, env):
    """
    Transform observation by reflecting ball positions across the vertical symmetry axis
    (middle of the table at x = table.w / 2).
    
    This allows using the LT-trained model for RT pocket by transforming the input.
    The cut_angle is recalculated for the new target pocket geometry.
    
    Args:
        obs: Original observation from environment [red_x, red_y, white_x, white_y, cut_angle]
        env: RLBilliard environment instance
        
    Returns:
        Transformed observation with ball positions reflected and cut_angle recalculated
    """
    obs_transformed = obs.copy()
    
    # The observation contains:
    # [0] red_x, [1] red_y, [2] white_x, [3] white_y, [4] cut_angle
    # Transform x-coordinates by reflecting across the vertical axis at x = table.w / 2
    
    # Reflect red ball x-coordinate
    obs_transformed[0] = 1.0 - obs[0]  # Normalized: x_new = 1.0 - x_old
    # y-coordinate stays the same: obs_transformed[1] = obs[1]
    
    # Reflect white ball x-coordinate
    obs_transformed[2] = 1.0 - obs[2]  # Normalized: x_new = 1.0 - x_old
    # y-coordinate stays the same: obs_transformed[3] = obs[3]
    
    # Recalculate cut_angle for the RT pocket
    # Get denormalized positions
    red_x_norm = obs_transformed[0]
    red_y_norm = obs_transformed[1]
    white_x_norm = obs_transformed[2]
    white_y_norm = obs_transformed[3]
    
    red_pos = np.array([red_x_norm * env.table.w, red_y_norm * env.table.l])
    white_pos = np.array([white_x_norm * env.table.w, white_y_norm * env.table.l])
    
    # Get RT pocket position
    rt_pocket = env.table.pockets['rt']
    rt_pocket_pos = np.array([float(rt_pocket.center[0]), float(rt_pocket.center[1])])
    
    # Calculate cut angle for RT pocket
    v_cue = red_pos - white_pos
    v_pocket = rt_pocket_pos - red_pos
    
    unit_cue = v_cue / (np.linalg.norm(v_cue) + 1e-8)
    unit_pocket = v_pocket / (np.linalg.norm(v_pocket) + 1e-8)
    dot_product = np.clip(np.dot(unit_cue, unit_pocket), -1.0, 1.0)
    cut_angle_deg = np.rad2deg(np.arccos(dot_product))
    
    # Normalize cut_angle to [0, 1] range
    obs_transformed[4] = cut_angle_deg / 180.0
    
    return obs_transformed


def main():
    model_path = "./models/ppo_billiards/ppo_billiards_final.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first!"
        )

    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)

    # Initialize environment for inference
    env = RLBilliard(verbose=False, use_curriculum=False, fixed_difficulty=100)

    total_episodes = 10
    successful_shots = 0

    print(f"\n--- Starting Evaluation ({total_episodes} Games) ---")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()

        # Calculate position difficulties BEFORE executing the shot
        print(f"\nEpisode {episode}/{total_episodes}:")
        print(f"  Cut Side: {info.get('cut_side', 'N/A')}")
        print()
        print("  Position Difficulty Analysis:")
        print("  " + "-" * 60)
        
        # Get ball positions
        red_pos = env.balls["red"].state.rvw[0][:2]
        white_pos = env.balls["white"].state.rvw[0][:2]
        print(f"    Red ball   : ({red_pos[0]:.3f}, {red_pos[1]:.3f})")
        print(f"    White ball : ({white_pos[0]:.3f}, {white_pos[1]:.3f})")
        print()
        
        # Calculate difficulty for top-left pocket
        difficulty_lt = env.calculate_position_difficulty(target_pocket_id='lt', visualize=False)
        print(f"    Top-Left (lt) pocket difficulty  : {difficulty_lt:6.2f}/100")
        
        # Calculate difficulty for top-right pocket
        difficulty_rt = env.calculate_position_difficulty(target_pocket_id='rt', visualize=False)
        print(f"    Top-Right (rt) pocket difficulty : {difficulty_rt:6.2f}/100")
        
        # Show difference
        diff = abs(difficulty_lt - difficulty_rt)
        print(f"    Difference                       : {diff:6.2f}/100")
        print()
        
        # Select easier pocket
        if difficulty_lt <= difficulty_rt:
            target_pocket = 'lt'
            obs_for_model = obs
            
            print(f"    → Selected: Top-Left (easier)")
        else:
            target_pocket = 'rt'
            obs_for_model = transform_obs_for_rt_pocket(obs, env)
            env.cut_side_factor = -1.0
            print(f"    → Selected: Top-Right (easier)")
            print(f"    → Observation transformed by vertical symmetry")
            print(f"    → Angle will be negated for opposite cut")
        
        print("  " + "-" * 60)
        print()

        # Predict deterministic action from trained policy
        action, _states = model.predict(obs_for_model, deterministic=True)


        obs, reward, terminated, truncated, info = env.step(action)

        angle_mag, velocity = action[0], action[1]
        is_success = info.get("red_pocketed_target", False) and not info.get(
            "white_pocketed", False
        )

        if is_success:
            successful_shots += 1

        print(f"  Action Chosen       : Angle Mag={angle_mag:.2f}°, Velocity={velocity:.2f} m/s")
        print(f"  Signed Control Angle: {info.get('signed_control_angle', 0.0):.2f}°")
        print(f"  Target Pocket       : {target_pocket.upper()} (lt=left-top, rt=right-top)")
        print(f"  Result              : {'SUCCESS (Pocketed)' if is_success else 'MISS'}")
        print(f"  Reward              : {reward:.2f}")

        # Render 3D visualization window
        env.render()
        
    env.close()

    success_rate = (successful_shots / total_episodes) * 100
    print(f"\n==========================================")
    print(f"Final Success Rate: {successful_shots}/{total_episodes} ({success_rate:.1f}%)")
    print(f"==========================================")


if __name__ == "__main__":
    from billiards_tests import test_transform_obs_for_rt_pocket
    
    # Run the transformation test
#    test_transform_obs_for_rt_pocket(verbose=True)
    
    # Then run the main adaptive pocket selection demo
    main()
