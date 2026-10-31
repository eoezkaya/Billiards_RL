#!/usr/bin/env python3
"""
Table visualization module for billiards environment.

Provides visualization of the billiards table with balls, target pocket,
coordinates, and placement areas.

Usage:
    from visualize_table import visualize_table
    
    visualize_table(env, filename="table.png")
"""

import numpy as np


def _get_pocket_center(env, pocket_id):
    """
    Get pocket center coordinates.
    
    For corner pockets, retrieves from env.table.pockets.
    For middle pockets (ml, mr), calculates manually since they're not predefined in pooltool.
    
    Args:
        env: Environment with table property
        pocket_id: Pocket identifier ('lb', 'lt', 'rb', 'rt', 'ml', 'mr')
    
    Returns:
        tuple: (x, y) coordinates of pocket center
    """
#    if pocket_id in ['ml', 'mr']:
#        if pocket_id == 'ml':
#            # Middle-Left: x=0, y=table_center
#            return (0.0, env.table.l / 2.0)
#        else:  # 'mr'
#            # Middle-Right: x=table_width, y=table_center
#            return (env.table.w, env.table.l / 2.0)
#       else:
        # Corner pockets: retrieve from table
    pocket_center = env.table.pockets[pocket_id].center
    return (float(pocket_center[0]), float(pocket_center[1]))


def visualize_table(env, filename=None, show_pocket_coords=True, show_absolute_coords=True):
    """
    Visualize the billiards table with balls, target pocket, coordinates, and placement areas.
    
    Shows:
      - Table as green rectangle
      - Red and white balls at current positions
      - Target pocket highlighted in blue
      - Valid placement areas for red and white balls as shaded regions
      - Coordinate information (absolute and pocket-relative)
      - Aim ray (direct or via cushion)
    
    Args:
      env: RLBilliard environment instance
      filename: If provided, save figure to this path instead of displaying
      show_pocket_coords: If True, display pocket-relative coordinates
      show_absolute_coords: If True, display absolute table coordinates
    
    Returns:
      matplotlib.figure.Figure: The figure object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return None
    
    # Create figure and axis with highly optimized size
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get ball positions
    red_pos = env.balls["red"].state.rvw[0][:2]
    white_pos = env.balls["white"].state.rvw[0][:2]
    
    # Draw table as green rectangle
    table_rect = patches.Rectangle(
        (0, 0), 
        env.table.w, 
        env.table.l,
        linewidth=3,
        edgecolor='darkgreen',
        facecolor='lightgreen',
        alpha=0.3,
        zorder=1
    )
    ax.add_patch(table_rect)
    
    # Draw valid placement areas as shaded rectangles
    # Red ball placement area
    red_area = patches.Rectangle(
        (env.red_x_range[0], env.red_y_range[0]),
        env.red_x_range[1] - env.red_x_range[0],
        env.red_y_range[1] - env.red_y_range[0],
        linewidth=1.5,
        edgecolor='red',
        facecolor='red',
        alpha=0.1,
        linestyle='--',
        zorder=2,
        label='Red placement area'
    )
    ax.add_patch(red_area)
    
    # White ball placement area
    white_area = patches.Rectangle(
        (env.white_x_range[0], env.white_y_range[0]),
        env.white_x_range[1] - env.white_x_range[0],
        env.white_y_range[1] - env.white_y_range[0],
        linewidth=1.5,
        edgecolor='black',
        facecolor='gray',
        alpha=0.08,
        linestyle='--',
        zorder=2,
        label='White placement area'
    )
    ax.add_patch(white_area)
    
    # Draw target pocket only
    pocket_radius = 0.05
    
    pocket_center = _get_pocket_center(env, env.target_pocket_id)
    pocket_circle = patches.Circle(
        (pocket_center[0], pocket_center[1]),
        pocket_radius,
        facecolor='darkblue',
        edgecolor='darkblue',
        alpha=0.8,
        linewidth=2,
        zorder=5
    )
    ax.add_patch(pocket_circle)
    
    # Draw balls
    red_circle = patches.Circle(
        (red_pos[0], red_pos[1]),
        env.R,
        facecolor='red',
        edgecolor='darkred',
        alpha=0.9,
        linewidth=2,
        zorder=10,
        label='Red ball'
    )
    ax.add_patch(red_circle)
    
    white_circle = patches.Circle(
        (white_pos[0], white_pos[1]),
        env.R,
        facecolor='white',
        edgecolor='black',
        alpha=0.9,
        linewidth=2,
        zorder=10,
        label='White ball'
    )
    ax.add_patch(white_circle)
    
    # Draw aim ray (direct or via cushion based on play_cushion flag)
    if hasattr(env, 'play_cushion') and env.play_cushion:
        # Draw ray with cushion bounce: white -> cushion_point -> red
        if hasattr(env, 'cushion_info') and env.cushion_info and env.cushion_info['cushion_point']:
            cushion_point = env.cushion_info['cushion_point']
            
            # Draw line from white ball to cushion point
            ax.plot([white_pos[0], cushion_point[0]], [white_pos[1], cushion_point[1]],
                    'r--', linewidth=2, alpha=0.6, zorder=8, label='Cushion shot ray')
            
            # Draw line from cushion point to red ball
            ax.plot([cushion_point[0], red_pos[0]], [cushion_point[1], red_pos[1]],
                    'r--', linewidth=2, alpha=0.6, zorder=8)
            
            # Mark the cushion point
            ax.plot(cushion_point[0], cushion_point[1], 'ro', markersize=6, alpha=0.8, zorder=9)
        else:
            # Fallback: draw ray based on aim angle if cushion_info not available
            import math
            angle_rad = math.radians(env.aim_angle_deg)
            ray_length = max(env.table.w, env.table.l) * 2
            ray_end_x = white_pos[0] + ray_length * math.cos(angle_rad)
            ray_end_y = white_pos[1] + ray_length * math.sin(angle_rad)
            
            ax.plot([white_pos[0], ray_end_x], [white_pos[1], ray_end_y],
                    'r--', linewidth=2, alpha=0.6, zorder=8, label='Cushion shot ray')
    else:
        # Direct ray to red ball
        ax.plot([white_pos[0], red_pos[0]], [white_pos[1], red_pos[1]],
                'b--', linewidth=2, alpha=0.6, zorder=8, label='Direct shot ray')
    
    # Add target pocket information - compact
    target_pocket_names = {
        'lb': 'LB',
        'rb': 'RB',
        'lt': 'LT',
        'rt': 'RT',
        'lc': 'ML',
        'rc': 'MR'
    }
    
    pocket_center = _get_pocket_center(env, env.target_pocket_id)
    ax.text(pocket_center[0], pocket_center[1] + 0.08,
            f'{target_pocket_names[env.target_pocket_id]}',
            fontsize=7, color='red', weight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='red', alpha=0.3, edgecolor='red', linewidth=1.5),
            zorder=12)
    
    # Add coordinate system origin indicator
    ax.plot(0, 0, 'ko', markersize=6, zorder=12)
    ax.text(0.02, -0.06, 'O', fontsize=7, color='black')
    
    # Add grid and labels - minimal
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.3)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_title(f'Billiards ( target pocket = {env.target_pocket_id})', fontsize=9, weight='bold')
    
    # Set axis limits with minimal padding
    padding = 0.05
    ax.set_xlim(-padding, env.table.w + padding)
    ax.set_ylim(-padding, env.table.l + padding)
    
    # Set aspect ratio to be equal (so circles appear as circles)
    ax.set_aspect('equal')
    
    # Create coordinates table as text beside the figure
    table_text = "BALL COORDINATES\n" + "=" * 35 + "\n\n"
    
    table_text += "RED BALL\n" + "-" * 35 + "\n"
    if show_absolute_coords:
        table_text += f"  Absolute:  ({red_pos[0]:.4f}, {red_pos[1]:.4f}) m\n"
    
    if show_pocket_coords:
        obs_pocket_rel = env.pocket_coordinates
        red_pocket_x, red_pocket_y = obs_pocket_rel[0], obs_pocket_rel[1]
        table_text += f"  Pocket-rel: ({red_pocket_x:.4f}, {red_pocket_y:.4f}) m\n"
        
        polar = env.get_obs_pocket_relative_polar()
        table_text += f"  Polar:      r={polar['red_radius']:.4f}m, θ={polar['red_angle']:.2f}°\n"
    
    table_text += "\nWHITE BALL\n" + "-" * 35 + "\n"
    if show_absolute_coords:
        table_text += f"  Absolute:  ({white_pos[0]:.4f}, {white_pos[1]:.4f}) m\n"
    
    if show_pocket_coords:
        white_pocket_x, white_pocket_y = obs_pocket_rel[2], obs_pocket_rel[3]
        table_text += f"  Pocket-rel: ({white_pocket_x:.4f}, {white_pocket_y:.4f}) m\n"
        
        table_text += f"  Polar:      r={polar['white_radius']:.4f}m, θ={polar['white_angle']:.2f}°\n"
    
    table_text += "\nTARGET POCKET\n" + "-" * 35 + "\n"
    table_text += f"  Pocket ID:  {env.target_pocket_id.upper()}\n"
    table_text += f"  Position:   ({pocket_center[0]:.4f}, {pocket_center[1]:.4f}) m\n"
    
    table_text += "\nAIM ANGLE\n" + "-" * 35 + "\n"
    table_text += f"  Angle:      {env.aim_angle_deg:.2f}°\n"
    
    # Adjust subplots to make room for table outside
    import matplotlib.pyplot as plt
    plt.subplots_adjust(right=0.65)
    
    # Add text box completely outside the plot rectangle
    fig.text(0.68, 0.5, table_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(pad=1.0, rect=[0, 0, 0.65, 1])
    
    # Save or show
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Table visualization saved to: {filename}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    from RLEnv_pocket_invariant import RLBilliard
    
    print("Initializing billiards environment...")
    env = RLBilliard(use_curriculum=False, verbose=False, target_pocket_id='lc', fixed_difficulty=100)
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print("Visualizing table...")
    visualize_table(env, filename="billiards_table.png")
    print("Done!")
