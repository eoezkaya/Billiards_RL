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


def visualize_billiards_position(env, red_angle_deg, position_difficulty, filename=None):
  """
  Draws the ball positions on the table as an image.
  Shows the green table, white and red balls, and the triangle (shaded blue).
  Displays the red angle and position difficulty score.
  
  Args:
      env: RLBilliard environment instance
      red_angle_deg: Red angle in degrees (pocket->red vs red->white)
      position_difficulty: Difficulty score (0-100)
      filename (str, optional): If provided, saves the image to this file.
                                Otherwise displays the plot.
  """
  fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  
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
  
  # Get target pocket center
  pocket_center = env.table.pockets[env.target_pocket_id].center
  target_pocket_center = np.array([float(pocket_center[0]), float(pocket_center[1])])
  
  # Draw triangle formed by: red ball, white ball, and target pocket (shaded blue)
  triangle_points = [
      (float(red_pos[0]), float(red_pos[1])),
      (float(white_pos[0]), float(white_pos[1])),
      (float(target_pocket_center[0]), float(target_pocket_center[1]))
  ]
  triangle = patches.Polygon(
      triangle_points,
      closed=True, edgecolor='darkblue', facecolor='blue', alpha=0.3,
      linewidth=2
  )
  ax.add_patch(triangle)
  
  # Mark the target pocket center
  ax.plot(target_pocket_center[0], target_pocket_center[1], 'b*', 
              markersize=20)
  
  # Draw line from pocket to red ball
  ax.plot([target_pocket_center[0], red_pos[0]], [target_pocket_center[1], red_pos[1]], 
              'r-', linewidth=2.5, alpha=0.8, label='Pocket→Red')
  
  # Draw line from red ball to white ball
  ax.plot([red_pos[0], white_pos[0]], [red_pos[1], white_pos[1]], 
              'b-', linewidth=2.5, alpha=0.8, label='Red→White')
  
  # Add red angle text in the figure near the red ball
  mid_x = red_pos[0] + 0.12 * np.cos(np.radians((np.rad2deg(np.arctan2(target_pocket_center[1] - red_pos[1], target_pocket_center[0] - red_pos[0])) + np.rad2deg(np.arctan2(white_pos[1] - red_pos[1], white_pos[0] - red_pos[0]))) / 2))
  mid_y = red_pos[1] + 0.12 * np.sin(np.radians((np.rad2deg(np.arctan2(target_pocket_center[1] - red_pos[1], target_pocket_center[0] - red_pos[0])) + np.rad2deg(np.arctan2(white_pos[1] - red_pos[1], white_pos[0] - red_pos[0]))) / 2))
  ax.text(mid_x, mid_y, f'{red_angle_deg:.1f}°', fontsize=14, fontweight='bold', 
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
  
  # Add difficulty text box in upper right corner
  ax.text(0.98, 0.98, f'Difficulty: {position_difficulty:.1f}/100', 
          transform=ax.transAxes, fontsize=13, fontweight='bold',
          verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.8))
  
  # Set axis properties
  ax.set_xlim(-0.1, env.table.w + 0.1)
  ax.set_ylim(-0.1, env.table.l + 0.1)
  ax.set_aspect('equal')
  ax.set_xlabel('Width (m)')
  ax.set_ylabel('Length (m)')
  ax.set_title(f'Billiards Table - Red Angle: {red_angle_deg:.1f}° | Difficulty: {position_difficulty:.1f}/100\n(0° = easiest, 90° = hardest)')
  ax.grid(True, alpha=0.3)
  
  if filename:
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
  else:
    plt.show()
  
  plt.close()
