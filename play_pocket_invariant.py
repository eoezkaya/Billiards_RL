#!/usr/bin/env python3

import os
import time
import numpy as np
from stable_baselines3 import PPO

from RLEnv import RLBilliard

TARGET_POCKET_ID = 'lc'
DIFFICULTY = 50

def main():
    model_path = "./models/ppo_billiards/ppo_billiards_final.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train_pocket_invariant.py first!"
        )

    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)
    red_x_range=(0.1, 0.9)
    red_y_range=(0.1, 1.95)
    white_x_range=(0.1, 0.9)
    white_y_range=(0.1, 1.95)


    # Initialize environment for inference
    env = RLBilliard(
        verbose=False,
        red_x_range = red_x_range,
        red_y_range = red_y_range,
        white_x_range = white_x_range,
        white_y_range = white_y_range,
        use_curriculum=False,
        fixed_difficulty=DIFFICULTY,
        target_pocket_id=TARGET_POCKET_ID,
        inference=True

    )

    total_episodes = 20
    successful_shots = 0

    print(f"\n--- Starting Evaluation ({total_episodes} Games) ---")
    print(f"Target Pocket: {TARGET_POCKET_ID.upper()}")
    print(f"Difficulty: {DIFFICULTY}")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()

        # Predict deterministic action from trained policy
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # Get polar coordinates for display
        polar = env.get_obs_pocket_relative_polar()
        
        is_success = info.get("red_pocketed_target", False) and not info.get(
            "white_pocketed", False
        )

        if is_success:
            successful_shots += 1

        print(f"\nEpisode {episode}/{total_episodes}:")
        print(f"  Target Pocket       : {info.get('target_pocket_id', 'N/A').upper()}")
        print(f"  Cut Side            : {info.get('cut_side', 'N/A')}")
        print(f"  Action Chosen       : [{action[0]:.4f}, {action[1]:.4f}]")
        print(f"  Red Ball Polar      : r={polar['red_radius']:.4f}m, θ={polar['red_angle']:.2f}°")
        print(f"  White Ball Polar    : r={polar['white_radius']:.4f}m, θ={polar['white_angle']:.2f}°")
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
    main()
