"""
Test methods for the billiards RL environment.

This module contains standalone test functions for testing various aspects
of the RLBilliard environment including difficulty levels and initial setup.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def test_difficulty(env, target_difficulty=None, action=None, output_dir="difficulty_tests"):
    """
    Tests the environment with a specific curriculum difficulty level.
    
    Configures the environment's curriculum to target a specific difficulty,
    then performs a reset and executes a shot. This tests the curriculum-based
    position generation through the reset() method.
    
    Args:
        env (RLBilliard): The environment instance
        target_difficulty (float, optional): Curriculum difficulty target (0-100).
                                           If None, uses current curriculum difficulty.
        action (np.ndarray, optional): Action to execute. If None, uses [0.0, 0.5]
        output_dir (str): Directory to save visualizations (default: "difficulty_tests")
    
    Returns:
        dict: Information about the test including:
            - 'target_difficulty': requested curriculum difficulty
            - 'actual_difficulty': actual difficulty of generated position
            - 'red_angle': red angle of the position
            - 'action': action executed
            - 'reward': reward from the shot
            - 'success': whether red was pocketed
            - 'white_scratch': whether white scratched
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Default action if not provided
    if action is None:
        action = np.array([0.0, 0.5], dtype=np.float32)
    
    # Use current curriculum difficulty if target not specified
    if target_difficulty is None:
        target_difficulty = env.current_curriculum_difficulty
    else:
        # Set curriculum to target difficulty for this test
        env.use_curriculum = False
        env.fixed_difficulty = target_difficulty
        
    print(f"\n{'='*60}")
    print(f"Testing Curriculum Difficulty Level: {target_difficulty}/100")
    print(f"{'='*60}")
    
    # Generate table using reset() which uses current curriculum difficulty
    obs, info = env.reset()
    actual_difficulty = env.calculate_position_difficulty()
    
    # Get red angle for reference
    red_pos = env.balls["red"].state.rvw[0][:2]
    white_pos = env.balls["white"].state.rvw[0][:2]
    pocket_center = env.table.pockets[env.target_pocket_id].center
    pocket_center = np.array([float(pocket_center[0]), float(pocket_center[1])])
    red_angle = env._calculate_red_angle(white_pos, red_pos, pocket_center)
    
    # Save initial position visualization
    filename_init = os.path.join(output_dir, f"difficulty_{target_difficulty:03.0f}_initial.png")
    env.visualize_balls(filename_init)
    print(f"\n✓ Initial position saved: {filename_init}")
    print(f"  Red angle: {red_angle:.1f}°")
    print(f"  Curriculum target: {target_difficulty:.1f}/100")
    print(f"  Actual position difficulty: {actual_difficulty:.1f}/100")
    
    # Execute the shot
    print(f"\nExecuting shot with action: {action}")
    obs, reward, terminated, truncated, step_info = env.step(action)
    
    # Check results
    red_pocketed = step_info['red_pocketed_target']
    white_pocketed = step_info['white_pocketed']
    
    print(f"  Control angle: {step_info['signed_control_angle']:.2f}°")
    print(f"  Reward: {reward:.1f}")
    
    # Update ball positions to final state after shot
    if env.shot is not None:
        env.balls = env.shot.balls
    
    # Save result visualization (after shot)
    # Note: After step, the balls have moved, so we visualize the result state
    filename_result = os.path.join(output_dir, f"difficulty_{target_difficulty:03.0f}_result.png")
    env.visualize_balls(filename_result)
    print(f"✓ Result position saved: {filename_result}")
    
    # Render the shot
    print(f"\nRendering shot animation...")
    env.render()
    
    # Print results
    print(f"\nResults:")
    print(f"  Red Pocketed (Success): {red_pocketed}")
    print(f"  White Scratched: {white_pocketed}")
    print(f"  Reward: {reward:.1f}")
    
    if red_pocketed and not white_pocketed:
        print(f"  ✓ SUCCESSFUL SHOT!")
    elif white_pocketed:
        print(f"  ✗ White scratched")
    else:
        print(f"  ✗ Red ball not pocketed")
    
    print(f"{'='*60}\n")
    
    # Return comprehensive test results
    return {
        'target_difficulty': target_difficulty,
        'actual_difficulty': actual_difficulty,
        'red_angle': red_angle,
        'action': action,
        'reward': reward,
        'success': red_pocketed and not white_pocketed,
        'white_scratch': white_pocketed,
        'initial_file': filename_init,
        'result_file': filename_result
    }


def test_initial(env):
    """
    Test the initial setup with two consecutive shots:
    Shot 1: action[0] = -1 (left)
    Shot 2: action[0] = 1 (right)
    Visualizes table layout, initial ball positions, and renders both shots together.
    
    Args:
        env (RLBilliard): The environment instance
    """
    print("="*60)
    print("Testing Initial Setup with Two Shots")
    print("="*60)
    
    # Visualize table and ball placement areas
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Draw table (green)
    table_rect = patches.Rectangle(
        (0, 0), env.table.w, env.table.l,
        linewidth=2, edgecolor='darkgreen', facecolor='green', alpha=0.6,
        label='Table'
    )
    ax.add_patch(table_rect)

    # Draw red ball placement area (red shadow)
    red_width = env.red_x_range[1] - env.red_x_range[0]
    red_height = env.red_y_range[1] - env.red_y_range[0]
    red_area = patches.Rectangle(
        (env.red_x_range[0], env.red_y_range[0]), red_width, red_height,
        linewidth=2, edgecolor='darkred', facecolor='red', alpha=0.3,
        label='Red Ball Area'
    )
    ax.add_patch(red_area)

    # Draw white ball placement area (gray shadow)
    white_width = env.white_x_range[1] - env.white_x_range[0]
    white_height = env.white_y_range[1] - env.white_y_range[0]
    white_area = patches.Rectangle(
        (env.white_x_range[0], env.white_y_range[0]), white_width, white_height,
        linewidth=2, edgecolor='darkgray', facecolor='gray', alpha=0.3,
        label='White Ball Area'
    )
    ax.add_patch(white_area)

    # Set axis properties
    ax.set_xlim(-0.2, env.table.w + 0.2)
    ax.set_ylim(-0.2, env.table.l + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('Width (m)', fontsize=11)
    ax.set_ylabel('Length (m)', fontsize=11)
    ax.set_title('Billiards Table with Ball Placement Areas', fontsize=13, fontweight='bold')

    # Save figure
    plt.tight_layout()
    plt.savefig('billiards_table_layout.png', dpi=150, bbox_inches='tight')
    print("✓ Table layout saved to 'billiards_table_layout.png'")
    plt.close()

    obs, info = env.reset()
    print(f"Initial Cut Side: {info['cut_side']}")
    
    # Visualize initial ball positions
    print("\nVisualizing initial ball positions...")
    env.visualize_balls('billiards_initial_balls.png')

    # First shot with action[0] = -1
    print("\n--- Shot 1: action[0] = -1 ---")
    action1 = np.array([-1.0, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action1)
    shot1 = env.shot

    print(f"Executed Control Angle: {info['signed_control_angle']:.2f}°")
    print(f"Reward Received: {reward}")
    print(f"Red Pocketed in Target Pocket (top-left): {info['red_pocketed_target']}")
    print(f"White Scratch: {info['white_pocketed']}")

    # Reset for second shot
    obs, info = env.reset()
    print(f"\n--- Shot 2: action[0] = 1 ---")
    action2 = np.array([1.0, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action2)
    shot2 = env.shot

    print(f"Executed Control Angle: {info['signed_control_angle']:.2f}°")
    print(f"Reward Received: {reward}")
    print(f"Red Pocketed in Target Pocket (top-left): {info['red_pocketed_target']}")
    print(f"White Scratch: {info['white_pocketed']}")

    # Store both shots and render them together
    env.shots = [shot1, shot2]
    env.render()

    print("="*60)
    print("✓ Initial test completed successfully")
    print("="*60 + "\n")


if __name__ == "__main__":
    from RLEnv import RLBilliard
    
    print("\nRunning billiards test suite...\n")
    
    env = RLBilliard()
    test_initial(env)
