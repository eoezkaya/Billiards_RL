#!/usr/bin/env python3

import os
import time
import numpy as np
from stable_baselines3 import PPO

from RLEnv import RLBilliard

# ===== TARGET POCKET CONFIGURATION =====
TARGET_POCKET = "top_left"  # Options: "top_left" or "top_right"
# ========================================


def main():
    if(TARGET_POCKET == "top_left"):
        model_path = "ppo_billiards_final_top_left.zip"
    else:
        model_path = "ppo_billiards_final_top_right.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first!"
        )

    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)

    # Initialize environment specifically for inference
    env = RLBilliard(target_pocket=TARGET_POCKET)

    total_episodes = 40
    successful_shots = 0

    print(f"\n--- Starting Evaluation ({total_episodes} Games) ---")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()

        # Predict deterministic action from trained policy
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        angle_mag, velocity = action[0], action[1]
        is_success = info.get("red_pocketed_target", False) and not info.get(
            "white_pocketed", False
        )

        if is_success:
            successful_shots += 1

        print(f"\nEpisode {episode}/{total_episodes}:")
        print(f"  Cut Side            : {info.get('cut_side', 'N/A')}")
        print(f"  Action Chosen       : Angle Mag={angle_mag:.2f}°, Velocity={velocity:.2f} m/s")
        print(f"  Signed Control Angle: {info.get('signed_control_angle', 0.0):.2f}°")
        print(f"  Result              : {'SUCCESS (Pocketed)' if is_success else 'MISS'}")
        print(f"  Reward              : {reward:.2f}")

        # Render 3D visualization window
        env.render()
        
        # Delay after viewing to keep ball positions visible longer

    env.close()

    success_rate = (successful_shots / total_episodes) * 100
    print(f"\n==========================================")
    print(f"Final Success Rate: {successful_shots}/{total_episodes} ({success_rate:.1f}%)")
    print(f"==========================================")


if __name__ == "__main__":
    main()