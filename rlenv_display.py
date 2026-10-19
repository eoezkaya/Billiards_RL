"""
Display utilities for RLBilliard environment.

This module provides formatting functions to display detailed information
about the RLBilliard environment state.
"""

import numpy as np


def format_env_state(env):
    """
    Format a detailed string representation of the RLBilliard environment state.
    
    Displays all important class information including table dimensions,
    ball positions, target pocket, difficulty settings, and geometry info.
    
    Args:
        env: RLBilliard environment instance
    
    Returns:
        str: Formatted string with all environment information
    """
    pocket_names = {
        'lt': 'Top-Left',
        'rt': 'Top-Right',
        'lc': 'Left-Center',
        'rc': 'Right-Center',
        'lb': 'Left-Bottom',
        'rb': 'Right-Bottom'
    }
    
    # Check if balls have been initialized
    balls_initialized = hasattr(env, 'balls') and env.balls is not None
    
    info = []
    info.append("=" * 70)
    info.append("RLBilliard Environment State")
    info.append("=" * 70)
    
    # Table Information
    info.append("\n[Table Information]")
    info.append(f"  Table Width         : {env.table.w:.3f} m")
    info.append(f"  Table Length        : {env.table.l:.3f} m")
    info.append(f"  Ball Radius         : {env.R:.6f} m")
    
    # Target Pocket
    info.append("\n[Target Pocket]")
    info.append(f"  Pocket ID           : {env.target_pocket_id} ({pocket_names.get(env.target_pocket_id, 'Unknown')})")
    info.append(f"  Pocket Position     : ({env.target_pocket_pos[0]:.3f}, {env.target_pocket_pos[1]:.3f}) m")
    
    # Ball Positions
    info.append("\n[Ball Positions]")
    if balls_initialized:
        red_pos = env.balls["red"].state.rvw[0][:2]
        white_pos = env.balls["white"].state.rvw[0][:2]
        info.append(f"  Red Ball            : ({red_pos[0]:.3f}, {red_pos[1]:.3f}) m")
        info.append(f"  White Ball          : ({white_pos[0]:.3f}, {white_pos[1]:.3f}) m")
        distance = np.hypot(red_pos[0] - white_pos[0], red_pos[1] - white_pos[1])
        info.append(f"  Distance            : {distance:.3f} m")
    else:
        info.append(f"  Red Ball            : Not initialized")
        info.append(f"  White Ball          : Not initialized")
    
    # Ball Placement Ranges
    info.append("\n[Ball Placement Ranges]")
    info.append(f"  Red X Range         : ({env.red_x_range[0]:.3f}, {env.red_x_range[1]:.3f}) m")
    info.append(f"  Red Y Range         : ({env.red_y_range[0]:.3f}, {env.red_y_range[1]:.3f}) m")
    info.append(f"  White X Range       : ({env.white_x_range[0]:.3f}, {env.white_x_range[1]:.3f}) m")
    info.append(f"  White Y Range       : ({env.white_y_range[0]:.3f}, {env.white_y_range[1]:.3f}) m")
    
    # Geometry Information
    info.append("\n[Geometry Information]")
    if balls_initialized:
        info.append(f"  Cut Side            : {env.cut_side}")
        info.append(f"  Cut Angle           : {env.cut_angle:.2f}°")
        info.append(f"  Aim Angle           : {env.aim_angle_deg:.2f}°")
    else:
        info.append(f"  Cut Side            : Not calculated")
        info.append(f"  Cut Angle           : N/A")
        info.append(f"  Aim Angle           : N/A")
    
    # Difficulty Settings
    info.append("\n[Difficulty Settings]")
    if env.use_curriculum:
        info.append(f"  Mode                : Curriculum Learning")
        info.append(f"  Current Difficulty  : {env.current_curriculum_difficulty:.2f}/100")
        info.append(f"  Initial Difficulty  : {env.curriculum_initial_difficulty:.2f}/100")
        info.append(f"  Final Difficulty    : {env.curriculum_final_difficulty:.2f}/100")
        info.append(f"  Counter 1 (to final): {env.curriculum_counter_1}")
        info.append(f"  Counter 2 (ongoing) : {env.curriculum_counter_2}")
    else:
        info.append(f"  Mode                : Fixed Difficulty")
        info.append(f"  Fixed Difficulty    : {env.fixed_difficulty:.2f}/100")
    
    # Action/Observation Spaces
    info.append("\n[Action & Observation Spaces]")
    info.append(f"  Action Space        : {env.action_space}")
    info.append(f"  Observation Space   : {env.observation_space}")
    
    # Episode Statistics
    info.append("\n[Episode Statistics]")
    info.append(f"  Total Shots         : {env.shot_counter}")
    info.append(f"  Stored Shots        : {len(env.shots)}")
    
    # Render Mode
    info.append("\n[Rendering]")
    info.append(f"  Render Mode         : {env.render_mode if env.render_mode else 'None'}")
    info.append(f"  Verbose             : {env.verbose}")
    
    info.append("\n" + "=" * 70)
    
    return "\n".join(info)
