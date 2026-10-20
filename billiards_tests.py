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
        env.current_curriculum_difficulty = target_difficulty
        
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

    print(f"Reward Received: {reward}")
    print(f"Red Pocketed in Target Pocket (top-left): {info['red_pocketed_target']}")
    print(f"White Scratch: {info['white_pocketed']}")

    # Reset for second shot
    obs, info = env.reset()
    print(f"\n--- Shot 2: action[0] = 1 ---")
    action2 = np.array([1.0, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action2)
    shot2 = env.shot

    print(f"Reward Received: {reward}")
    print(f"Red Pocketed in Target Pocket (top-left): {info['red_pocketed_target']}")
    print(f"White Scratch: {info['white_pocketed']}")

    # Store both shots and render them together
    env.shots = [shot1, shot2]
    env.render()

    print("="*60)
    print("✓ Initial test completed successfully")
    print("="*60 + "\n")


def test_position_difficulty(env, target_pocket='lt', verbose=True):
    """
    Test method for calculate_position_difficulty().
    
    Generates a single random table configuration and calculates the difficulty
    for the specified target pocket. Displays detailed results.
    
    Args:
        env (RLBilliard): The environment instance
        target_pocket (str): Target pocket ID ('lt', 'rt', 'lc', 'rc', 'lb', 'rb').
                           Default is 'lt' (top-left)
        verbose (bool): Whether to print detailed information (default: True)
    
    Returns:
        dict: Dictionary containing position data and difficulty information
    """
    if verbose:
        print("=" * 70)
        print("Test: Position Difficulty Calculation")
        print("=" * 70)
        print(f"\nGenerating random table position for pocket '{target_pocket}'...\n")
    
    # Reset to generate a random table
    obs, info = env.reset()
    
    # Get ball positions
    red_pos = env.balls["red"].state.rvw[0][:2]
    white_pos = env.balls["white"].state.rvw[0][:2]
    
    # Calculate difficulty for the specified pocket
    difficulty = env.calculate_position_difficulty(target_pocket_id=target_pocket)
    
    # Prepare results
    results = {
        'target_pocket': target_pocket,
        'red_pos': red_pos,
        'white_pos': white_pos,
        'difficulty': difficulty,
        'cut_side': info.get('cut_side', 'N/A')
    }
    
    if verbose:
        print("Position Data:")
        print("-" * 70)
        print(f"  Red ball    : ({red_pos[0]:.3f}, {red_pos[1]:.3f})")
        print(f"  White ball  : ({white_pos[0]:.3f}, {white_pos[1]:.3f})")
        print(f"  Cut side    : {info.get('cut_side', 'N/A')}")
        print()
        print("Difficulty Results:")
        print("-" * 70)
        print(f"  Target pocket : {target_pocket}")
        print(f"  Difficulty    : {difficulty:6.2f}/100")
        print()
        print("=" * 70 + "\n")
    
    return results


def test_calculate_position_difficulty_with_visualization(env, verbose=True):
    """
    Test calculate_position_difficulty() with visualization for multiple pockets.
    
    Generates a single random table position and visualizes the difficulty calculation
    for both top-left (lt) and top-right (rt) pockets using the visualize parameter.
    
    Args:
        env (RLBilliard): The environment instance
        verbose (bool): Whether to print detailed information (default: True)
    
    Returns:
        dict: Dictionary containing difficulty results for both pockets
    """
    if verbose:
        print("=" * 70)
        print("Test: calculate_position_difficulty() with Visualization")
        print("=" * 70)
        print("\nGenerating random table position...\n")
    
    # Generate a single random table position
    obs, info = env.reset()
    
    # Get ball positions
    red_pos = env.balls["red"].state.rvw[0][:2]
    white_pos = env.balls["white"].state.rvw[0][:2]
    cut_side = info.get('cut_side', 'N/A')
    
    if verbose:
        print("Position Data:")
        print("-" * 70)
        print(f"  Red ball    : ({red_pos[0]:.3f}, {red_pos[1]:.3f})")
        print(f"  White ball  : ({white_pos[0]:.3f}, {white_pos[1]:.3f})")
        print(f"  Cut side    : {cut_side}")
        print()
    
    # Test Top-Left Pocket (lt) with visualization
    if verbose:
        print("Calculating difficulty for Top-Left (lt) pocket with visualization...")
    difficulty_lt = env.calculate_position_difficulty(target_pocket_id='lt', visualize=True)
    
    if verbose:
        print(f"  ✓ Top-Left difficulty: {difficulty_lt:.2f}/100")
        print()
    
    # Test Top-Right Pocket (rt) with visualization
    if verbose:
        print("Calculating difficulty for Top-Right (rt) pocket with visualization...")
    difficulty_rt = env.calculate_position_difficulty(target_pocket_id='rt', visualize=True)
    
    if verbose:
        print(f"  ✓ Top-Right difficulty: {difficulty_rt:.2f}/100")
        print()
    
    # Prepare results
    results = {
        'red_pos': red_pos,
        'white_pos': white_pos,
        'cut_side': cut_side,
        'difficulty_lt': difficulty_lt,
        'difficulty_rt': difficulty_rt,
        'difference': abs(difficulty_lt - difficulty_rt)
    }
    
    if verbose:
        print("=" * 70)
        print("Results Summary:")
        print("=" * 70)
        print(f"  Top-Left (lt) difficulty   : {difficulty_lt:6.2f}/100")
        print(f"  Top-Right (rt) difficulty  : {difficulty_rt:6.2f}/100")
        print(f"  Difference                 : {results['difference']:6.2f}/100")
        print()
        print("✓ Test completed successfully!")
        print("=" * 70 + "\n")
    
    return results


def test_transform_obs_for_rt_pocket(verbose=True):
    """
    Test the observation transformation for right-top pocket.
    
    Generates a random table position and visualizes:
    1. Original table with ball positions (LT orientation)
    2. Transformed table with reflected ball positions (RT orientation)
    
    This demonstrates how the symmetry transformation converts an RT shot
    into an LT-equivalent shot that the model can understand.
    
    Args:
        verbose (bool): If True, prints detailed output. Default is True.
    
    Returns:
        dict: Test results including original obs, transformed obs, and comparison
    """
    from RLEnv import RLBilliard
    from play_top_left_and_right import transform_obs_for_rt_pocket
    
    print("\n" + "=" * 70)
    print("Test: Observation Transformation for Right-Top Pocket")
    print("=" * 70)
    
    # Create environment and generate random position
    env = RLBilliard(verbose=False)
    obs, _ = env.reset()
    print(obs)
    
    # Get ball positions from observation
    red_x_orig = obs[0]
    red_y_orig = obs[1]
    white_x_orig = obs[2]
    white_y_orig = obs[3]
    
    if verbose:
        print("\nOriginal Table (LT Orientation):")
        print("-" * 70)
        print(f"  Red ball   : ({red_x_orig:.3f}, {red_y_orig:.3f})")
        print(f"  White ball : ({white_x_orig:.3f}, {white_y_orig:.3f})")
    
    # Transform observation for RT pocket
    obs_transformed = transform_obs_for_rt_pocket(obs, env)
    print(obs_transformed)
    
    red_x_trans = obs_transformed[0]
    red_y_trans = obs_transformed[1]
    white_x_trans = obs_transformed[2]
    white_y_trans = obs_transformed[3]
    
    if verbose:
        print("\nTransformed Table (RT Orientation via Symmetry):")
        print("-" * 70)
        print(f"  Red ball   : ({red_x_trans:.3f}, {red_y_trans:.3f})")
        print(f"  White ball : ({white_x_trans:.3f}, {white_y_trans:.3f})")
    
    # Verify transformation properties
    table_width = env.table.w
    
    # Check vertical symmetry: x_new = table.w - x_old
    red_x_expected = table_width - red_x_orig
    white_x_expected = table_width - white_x_orig
    
    red_x_matches = np.isclose(red_x_trans, red_x_expected, atol=1e-6)
    white_x_matches = np.isclose(white_x_trans, white_x_expected, atol=1e-6)
    red_y_matches = np.isclose(red_y_trans, red_y_orig, atol=1e-6)
    white_y_matches = np.isclose(white_y_trans, white_y_orig, atol=1e-6)
    
    if verbose:
        print("\nTransformation Verification:")
        print("-" * 70)
        print(f"  Red X:   {red_x_orig:.3f} → {red_x_trans:.3f} (expected: {red_x_expected:.3f}) {'✓' if red_x_matches else '✗'}")
        print(f"  Red Y:   {red_y_orig:.3f} → {red_y_trans:.3f} (expected: {red_y_orig:.3f}) {'✓' if red_y_matches else '✗'}")
        print(f"  White X: {white_x_orig:.3f} → {white_x_trans:.3f} (expected: {white_x_expected:.3f}) {'✓' if white_x_matches else '✗'}")
        print(f"  White Y: {white_y_orig:.3f} → {white_y_trans:.3f} (expected: {white_y_orig:.3f}) {'✓' if white_y_matches else '✗'}")
    
    all_correct = red_x_matches and white_x_matches and red_y_matches and white_y_matches
    
    # Create side-by-side visualization
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === LEFT PLOT: Original Table (LT Orientation) ===
    ax_left.set_title('Original Table (LT Orientation)', fontsize=14, fontweight='bold')
    
    # Draw table
    table_rect_left = patches.Rectangle(
        (0, 0), env.table.w, env.table.l,
        linewidth=2, edgecolor='darkgreen', facecolor='green', alpha=0.6
    )
    ax_left.add_patch(table_rect_left)
    
    # Draw balls
    white_circle_left = patches.Circle(
        (white_x_orig, white_y_orig), env.R,
        edgecolor='white', facecolor='white', linewidth=2
    )
    ax_left.add_patch(white_circle_left)
    ax_left.plot(white_x_orig, white_y_orig, 'k+', markersize=10, markeredgewidth=2)
    
    red_circle_left = patches.Circle(
        (red_x_orig, red_y_orig), env.R,
        edgecolor='darkred', facecolor='red', linewidth=2
    )
    ax_left.add_patch(red_circle_left)
    ax_left.plot(red_x_orig, red_y_orig, 'k+', markersize=10, markeredgewidth=2)
    
    # Mark LT pocket
    lt_pocket = env.table.pockets['lt']
    lt_center = np.array([float(lt_pocket.center[0]), float(lt_pocket.center[1])])
    ax_left.plot(lt_center[0], lt_center[1], 'b*', markersize=20, label='LT Pocket')
    
    # Draw shot lines
    ax_left.plot([lt_center[0], red_x_orig], [lt_center[1], red_y_orig], 'b-', linewidth=2, alpha=0.8)
    ax_left.plot([red_x_orig, white_x_orig], [red_y_orig, white_y_orig], 'b-', linewidth=2, alpha=0.8)
    
    ax_left.set_xlim(-0.2, env.table.w + 0.2)
    ax_left.set_ylim(-0.2, env.table.l + 0.2)
    ax_left.set_aspect('equal')
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=10, loc='upper right')
    ax_left.set_xlabel('X (meters)', fontsize=11)
    ax_left.set_ylabel('Y (meters)', fontsize=11)
    
    # === RIGHT PLOT: Transformed Table (RT Orientation) ===
    ax_right.set_title('Transformed Table (RT Orientation via Symmetry)', fontsize=14, fontweight='bold')
    
    # Draw table
    table_rect_right = patches.Rectangle(
        (0, 0), env.table.w, env.table.l,
        linewidth=2, edgecolor='darkgreen', facecolor='green', alpha=0.6
    )
    ax_right.add_patch(table_rect_right)
    
    # Draw balls (transformed positions)
    white_circle_right = patches.Circle(
        (white_x_trans, white_y_trans), env.R,
        edgecolor='white', facecolor='white', linewidth=2
    )
    ax_right.add_patch(white_circle_right)
    ax_right.plot(white_x_trans, white_y_trans, 'k+', markersize=10, markeredgewidth=2)
    
    red_circle_right = patches.Circle(
        (red_x_trans, red_y_trans), env.R,
        edgecolor='darkred', facecolor='red', linewidth=2
    )
    ax_right.add_patch(red_circle_right)
    ax_right.plot(red_x_trans, red_y_trans, 'k+', markersize=10, markeredgewidth=2)
    
    # Mark RT pocket (which appears as LT position after transformation)
    rt_pocket = env.table.pockets['rt']
    rt_center = np.array([float(rt_pocket.center[0]), float(rt_pocket.center[1])])
    ax_right.plot(rt_center[0], rt_center[1], 'r*', markersize=20, label='RT Pocket')
    
    # Draw shot lines
    ax_right.plot([rt_center[0], red_x_trans], [rt_center[1], red_y_trans], 'r-', linewidth=2, alpha=0.8)
    ax_right.plot([red_x_trans, white_x_trans], [red_y_trans, white_y_trans], 'r-', linewidth=2, alpha=0.8)
    
    ax_right.set_xlim(-0.2, env.table.w + 0.2)
    ax_right.set_ylim(-0.2, env.table.l + 0.2)
    ax_right.set_aspect('equal')
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=10, loc='upper right')
    ax_right.set_xlabel('X (meters)', fontsize=11)
    ax_right.set_ylabel('Y (meters)', fontsize=11)
    
    # Add info text
    info_text = (
        f"Symmetry Axis: x = {table_width/2:.3f}m\n"
        f"Transformation: x_new = {table_width:.3f} - x_old\n"
        f"Y-coordinates: unchanged\n"
        f"{'✓ All coordinates transformed correctly' if all_correct else '✗ Transformation error detected'}"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    
    if verbose:
        print("\n" + "=" * 70)
        if all_correct:
            print("✓ Test PASSED: All coordinates transformed correctly!")
        else:
            print("✗ Test FAILED: Transformation errors detected!")
        print("=" * 70 + "\n")
    
    return {
        'obs_original': obs,
        'obs_transformed': obs_transformed,
        'red_original': (red_x_orig, red_y_orig),
        'red_transformed': (red_x_trans, red_y_trans),
        'white_original': (white_x_orig, white_y_orig),
        'white_transformed': (white_x_trans, white_y_trans),
        'transformation_correct': all_correct
    }


def test_reset_with_positions(red_pos=(0.5, 1.5), white_pos=(0.5, 0.9), action=None, verbose=True):
    """
    Test the reset_with_positions method.
    
    Places balls at specific positions, visualizes the table, and executes a shot.
    This tests the ability to initialize the environment with predetermined ball
    configurations for testing specific scenarios.
    
    Args:
        red_pos (tuple): Red ball position [x, y] in meters. Default: (0.5, 1.5)
        white_pos (tuple): White ball position [x, y] in meters. Default: (0.5, 0.9)
        action (np.ndarray, optional): Action to execute. If None, uses [0.0, 0.5]
        verbose (bool): If True, prints detailed output. Default is True.
    
    Returns:
        dict: Test results including positions, observation, execution results
    """
    from RLEnv import RLBilliard
    
    print("\n" + "=" * 70)
    print("Test: Reset with Specific Ball Positions")
    print("=" * 70)
    
    if action is None:
        action = np.array([0.0, 0.5], dtype=np.float32)
    
    # Create environment
    env = RLBilliard(verbose=False, use_curriculum=False, fixed_difficulty=100)
    
    if verbose:
        print("\nInitializing environment with specific ball positions...")
        print(f"  Red ball   : {red_pos}")
        print(f"  White ball : {white_pos}")
    
    try:
        # Reset with specific positions
        obs, info = env.reset_with_positions(red_pos=red_pos, white_pos=white_pos)
        
        if verbose:
            print("\n✓ Reset successful!")
            print("\nInitialized Position Data:")
            print("-" * 70)
            print(f"  Red ball position   : ({info['red_pos'][0]:.3f}, {info['red_pos'][1]:.3f})")
            print(f"  White ball position : ({info['white_pos'][0]:.3f}, {info['white_pos'][1]:.3f})")
            print(f"  Cut side            : {info['cut_side']}")
            print(f"  Cut angle           : {info['cut_angle']:.2f}°")
            print(f"  Observation (norm)  : {obs}")
        
        # Visualize the table before shot
        if verbose:
            print("\nVisualizing table before shot...")
        env.visualize_balls()
        
        # Execute action
        if verbose:
            print("\nExecuting shot...")
            print(f"  Action (angle, velocity): [{action[0]:.2f}, {action[1]:.2f}]")
        
        obs_after, reward, terminated, truncated, info_after = env.step(action)
        
        # Extract result info
        is_success = info_after.get("red_pocketed_target", False)
        white_scratch = info_after.get("white_scratch", False)
        
        if verbose:
            print("\nShot Results:")
            print("-" * 70)
            print(f"  Reward              : {reward:.2f}")
            print(f"  Red Pocketed        : {is_success}")
            print(f"  White Scratch       : {white_scratch}")
            print(f"  Episode Terminated  : {terminated}")
            print(f"  Observation (norm)  : {obs_after}")
        
        # Render the shot (3D visualization)
        if verbose:
            print("\nRendering shot animation...")
        env.render()
        
        results = {
            'red_pos': red_pos,
            'white_pos': white_pos,
            'action': action,
            'obs_initial': obs,
            'obs_after': obs_after,
            'reward': reward,
            'success': is_success,
            'white_scratch': white_scratch,
            'terminated': terminated,
            'cut_side': info['cut_side'],
            'cut_angle': info['cut_angle']
        }
        
        if verbose:
            print("\n" + "=" * 70)
            print("✓ Test COMPLETED successfully!")
            print("=" * 70 + "\n")
        
        return results
        
    except ValueError as e:
        if verbose:
            print(f"\n✗ Test FAILED with validation error:")
            print(f"  {str(e)}")
            print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    from RLEnv import RLBilliard
    
    print("\nRunning billiards test suite...\n")
    
    env = RLBilliard(verbose=False)
