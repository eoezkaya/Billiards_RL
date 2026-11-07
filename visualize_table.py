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


def visualize_table(env, filename=None, show_absolute_coords=True):
    """
    Visualize the billiards table with all balls at their current positions.
    
    Shows:
      - Table as green rectangle
      - All balls from the current pool rack
      - Coordinate information (absolute table coordinates)
    
    Args:
      env: Environment instance with balls and table
      filename: If provided, save figure to this path instead of displaying
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
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    
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
    
    # Draw all pockets
    pocket_radius = 0.05
    pockets = env.table.pockets
    for pocket_id, pocket in pockets.items():
        pocket_center = (float(pocket.center[0]), float(pocket.center[1]))
        pocket_circle = patches.Circle(
            pocket_center,
            pocket_radius,
            facecolor='darkgray',
            edgecolor='black',
            alpha=0.7,
            linewidth=1.5,
            zorder=5
        )
        ax.add_patch(pocket_circle)
        ax.text(pocket_center[0], pocket_center[1] - 0.1, pocket_id, 
                ha='center', va='top', fontsize=8, zorder=6)
    
    # Draw all balls with different colors
    ball_colors = {
        'cue': ('white', 'black'),      # (fill, edge)
        '1': ('yellow', 'black'),
        '2': ('blue', 'black'),
        '3': ('red', 'darkred'),
        '4': ('purple', 'black'),
        '5': ('orange', 'black'),
        '6': ('green', 'darkgreen'),
        '7': ('brown', 'black'),
        '8': ('black', 'white'),
        '9': ('yellow', 'orange'),
    }
    
    for ball_id, ball in env.balls.items():
        pos = ball.state.rvw[0][:2]
        
        # Get color for this ball
        if ball_id in ball_colors:
            fill_color, edge_color = ball_colors[ball_id]
        else:
            fill_color, edge_color = ('cyan', 'black')
        
        # Draw ball circle
        circle = patches.Circle(
            (pos[0], pos[1]),
            env.R,
            facecolor=fill_color,
            edgecolor=edge_color,
            alpha=0.9,
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Add ball ID label
        ax.text(pos[0], pos[1], ball_id, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='black', zorder=11)
    
    # Set axis properties
    ax.set_xlim(-0.2, env.table.w + 0.2)
    ax.set_ylim(-0.2, env.table.l + 0.2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_title('Billiards Table Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add coordinate information
    if show_absolute_coords:
        info_text = f"Table: {env.table.w:.2f}m × {env.table.l:.2f}m\nBalls: {len(env.balls)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save or display
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    from RLEnv9Ball import RLBilliard9Ball
    
    print("Initializing billiards environment...")
    env = RLBilliard9Ball()
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print("Visualizing table...")
    visualize_table(env, filename="billiards_table.png")
    print("Done!")
