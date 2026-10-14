#!/usr/bin/env python3

import os
import time
import numpy as np
from stable_baselines3 import PPO

from RLEnv import RLBilliard


def main():
    model_path = "./best_model.zip"

    # Fallback to final model if best_model doesn't exist yet
    if not os.path.exists(model_path):
        model_path = "./ppo_billiards_final.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first!"
        )

    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)

    # Initialize environment specifically for inference
    env = RLBilliard()

    total_episodes = 3
    successful_shots = 0

    print("\n--- Starting Evaluation (3 Games) ---")

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