"""Visualization utilities for billiards RL environment."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_balls(env, filename=None):
  """
  Draws the ball positions on the table as an image.
  Shows the green table, white and red balls, and the triangle (shaded blue).
  Calculates and displays the red angle (pocket->red vs red->white).
  Also displays the position difficulty score (0-100).
  
  Args:
      env: RLBilliard environment instance
      filename (str, optional): If provided, saves the image to this file.
                                Otherwise displays the plot.
  """
  # Get current ball positions
  red_pos = env.balls["red"].state.rvw[0][:2]
  white_pos = env.balls["white"].state.rvw[0][:2]
  
  # Get target pocket center
  pocket_center = env.table.pockets[env.target_pocket_id].center
  target_pocket_center = np.array([float(pocket_center[0]), float(pocket_center[1])])
  
  # Calculate red angle (pocket->red vs red->white)
  red_angle_deg = env._calculate_red_angle(white_pos, red_pos, target_pocket_center)
  
  # Calculate position difficulty
  position_difficulty = env.calculate_position_difficulty()
  
  # Use the main visualization function
  visualize_billiards_position(env, red_angle_deg, position_difficulty, filename)


def calculate_angle_between_vectors(pos1, pos2, pos3):
  """
  Calculates the angle between two vectors sharing a common point.
  
  Args:
      pos1: First point [x, y]
      pos2: Common vertex [x, y]
      pos3: Second point [x, y]
  
  Returns:
      float: Angle in degrees between the two vectors
  """
  vec1 = np.array(pos1) - np.array(pos2)
  vec2 = np.array(pos3) - np.array(pos2)
  
  dot_product = np.dot(vec1, vec2)
  magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)
  cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
  angle_between_deg = np.rad2deg(np.arccos(cos_angle))
  
  return angle_between_deg


def visualize_billiards_position(env, red_angle_deg, position_difficulty, filename=None, target_pocket_id=None):
  """
  Draws the ball positions on the table as an image.
  Shows the green table, white and red balls, and two triangles:
  - Triangle 1: pocket->red->white (target pocket, shaded blue)
  - Triangle 2: pocket->red->white (top-right pocket, shaded orange)
  Displays the red angles and position difficulty score.
  
  Args:
      env: RLBilliard environment instance
      red_angle_deg: Red angle in degrees (pocket->red vs red->white)
      position_difficulty: Difficulty score (0-100)
      filename (str, optional): If provided, saves the image to this file.
                                Otherwise displays the plot.
      target_pocket_id (str, optional): Pocket ID to visualize for. If None, uses env.target_pocket_id.
  """
  if target_pocket_id is None:
    target_pocket_id = env.target_pocket_id
  
  fig, ax = plt.subplots(1, 1, figsize=(14, 9))
  
  # Draw table (green)
  table_rect = patches.Rectangle(
      (0, 0), env.table.w, env.table.l,
      linewidth=2, edgecolor='darkgreen', facecolor='green', alpha=0.6
  )
  ax.add_patch(table_rect)
  
  # Get current ball positions
  red_pos = env.balls["red"].state.rvw[0][:2]
  white_pos = env.balls["white"].state.rvw[0][:2]
  
  # Draw white ball
  white_circle = patches.Circle(
      white_pos, env.R,
      edgecolor='white', facecolor='white', linewidth=2
  )
  ax.add_patch(white_circle)
  ax.plot(white_pos[0], white_pos[1], 'k+', markersize=10, markeredgewidth=2)
  
  # Draw red ball
  red_circle = patches.Circle(
      red_pos, env.R,
      edgecolor='darkred', facecolor='red', linewidth=2
  )
  ax.add_patch(red_circle)
  ax.plot(red_pos[0], red_pos[1], 'k+', markersize=10, markeredgewidth=2)
  
  # Get target pocket center (specified target pocket)
  pocket_center = env.table.pockets[target_pocket_id].center
  target_pocket_center = np.array([float(pocket_center[0]), float(pocket_center[1])])
  
  # Get top-right pocket center for comparison
  # Find the pocket closest to the top-right corner (max_x, max_y)
  top_right_corner = np.array([env.table.w, env.table.l])
  top_right_pocket = min(env.table.pockets.values(), 
                         key=lambda p: np.linalg.norm(np.array([p.center[0], p.center[1]]) - top_right_corner))
  top_right_pocket_center = np.array([float(top_right_pocket.center[0]), float(top_right_pocket.center[1])])
  
  # Calculate angle for top-right pocket
  top_right_angle_deg = env._calculate_red_angle(white_pos, red_pos, top_right_pocket_center)
  
  # === DRAW TRIANGLE 1 (Target Pocket) ===
  triangle_points_1 = [
      (float(red_pos[0]), float(red_pos[1])),
      (float(white_pos[0]), float(white_pos[1])),
      (float(target_pocket_center[0]), float(target_pocket_center[1]))
  ]
  triangle_1 = patches.Polygon(
      triangle_points_1,
      closed=True, edgecolor='darkblue', facecolor='blue', alpha=0.3,
      linewidth=2
  )
  ax.add_patch(triangle_1)
  
  # Mark the target pocket center
  ax.plot(target_pocket_center[0], target_pocket_center[1], 'b*', 
              markersize=20, label=f'Target Pocket ({target_pocket_id})')
  
  # Draw lines for target pocket
  ax.plot([target_pocket_center[0], red_pos[0]], [target_pocket_center[1], red_pos[1]], 
              'b-', linewidth=2.5, alpha=0.8)
  ax.plot([red_pos[0], white_pos[0]], [red_pos[1], white_pos[1]], 
              'b-', linewidth=2.5, alpha=0.8)
  
  # === DRAW TRIANGLE 2 (Top-Right Pocket - for comparison) ===
  triangle_points_2 = [
      (float(red_pos[0]), float(red_pos[1])),
      (float(white_pos[0]), float(white_pos[1])),
      (float(top_right_pocket_center[0]), float(top_right_pocket_center[1]))
  ]
  triangle_2 = patches.Polygon(
      triangle_points_2,
      closed=True, edgecolor='darkorange', facecolor='orange', alpha=0.3,
      linewidth=2
  )
  ax.add_patch(triangle_2)
  
  # Mark the top-right pocket center
  ax.plot(top_right_pocket_center[0], top_right_pocket_center[1], 'o', 
              color='darkorange', markersize=15, label='Top-Right Pocket')
  
  # Draw lines for top-right pocket
  ax.plot([top_right_pocket_center[0], red_pos[0]], [top_right_pocket_center[1], red_pos[1]], 
              color='orange', linewidth=2.5, alpha=0.8, linestyle='--')
  ax.plot([red_pos[0], white_pos[0]], [red_pos[1], white_pos[1]], 
              color='orange', linewidth=2.5, alpha=0.8, linestyle='--')
  
  # Add angle text for target pocket
  mid_x_1 = red_pos[0] + 0.12 * np.cos(np.radians((np.rad2deg(np.arctan2(target_pocket_center[1] - red_pos[1], target_pocket_center[0] - red_pos[0])) + np.rad2deg(np.arctan2(white_pos[1] - red_pos[1], white_pos[0] - red_pos[0]))) / 2))
  mid_y_1 = red_pos[1] + 0.12 * np.sin(np.radians((np.rad2deg(np.arctan2(target_pocket_center[1] - red_pos[1], target_pocket_center[0] - red_pos[0])) + np.rad2deg(np.arctan2(white_pos[1] - red_pos[1], white_pos[0] - red_pos[0]))) / 2))
  ax.text(mid_x_1 - 0.15, mid_y_1, f'Target\n{red_angle_deg:.1f}°', fontsize=12, fontweight='bold', 
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
  
  # Add angle text for top-right pocket
  mid_x_2 = red_pos[0] + 0.12 * np.cos(np.radians((np.rad2deg(np.arctan2(top_right_pocket_center[1] - red_pos[1], top_right_pocket_center[0] - red_pos[0])) + np.rad2deg(np.arctan2(white_pos[1] - red_pos[1], white_pos[0] - red_pos[0]))) / 2))
  mid_y_2 = red_pos[1] + 0.12 * np.sin(np.radians((np.rad2deg(np.arctan2(top_right_pocket_center[1] - red_pos[1], top_right_pocket_center[0] - red_pos[0])) + np.rad2deg(np.arctan2(white_pos[1] - red_pos[1], white_pos[0] - red_pos[0]))) / 2))
  ax.text(mid_x_2 + 0.15, mid_y_2, f'Top-Right\n{top_right_angle_deg:.1f}°', fontsize=12, fontweight='bold', 
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
  
  # Add difficulty text box in upper right corner
  ax.text(0.98, 0.98, f'Difficulty: {position_difficulty:.1f}/100', 
          transform=ax.transAxes, fontsize=13, fontweight='bold',
          verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.8))
  
  # Set axis properties
  ax.set_xlim(-0.1, env.table.w + 0.1)
  ax.set_ylim(-0.1, env.table.l + 0.1)
  ax.set_aspect('equal')
  ax.set_xlabel('Width (m)', fontsize=11)
  ax.set_ylabel('Length (m)', fontsize=11)
  ax.set_title(f'Billiards Table - Multiple Pocket Views\nTarget ({target_pocket_id}): {red_angle_deg:.1f}° | Alternative (Top-Right): {top_right_angle_deg:.1f}° | Difficulty: {position_difficulty:.1f}/100', 
               fontsize=12, fontweight='bold')
  ax.grid(True, alpha=0.3)
  ax.legend(loc='lower left', fontsize=10)
  
  if filename:
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
  else:
    plt.show()
  
  plt.close()
