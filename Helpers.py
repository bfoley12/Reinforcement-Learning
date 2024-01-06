import math
import random
import numpy as np

def did_crash(old_position, new_position, track):
   """Finds the point cross just before crashing
   Uses Besenham's algorithm to find all matrix points crossed in the line taken
   
   Args:
      old_position (array): the previous position [y, x]
      new_position (array): the new position [y, x]
      track (Racetrack): the track that you are currently using
   Returns:
      the last valid point before encountering the wall. If no wall encountered, returns (None, None)
   """
   # Bresenham's algorithm
   dx = abs(new_position[1] - old_position[1])
   dy = abs(new_position[0] - old_position[0])
   sx = -1 if old_position[1] > new_position[1] else 1
   sy = -1 if old_position[0] > new_position[0] else 1
   err = dx - dy
    
   line_points = []
   
   # Calculate all points on line
   while old_position[0] != new_position[0] or old_position[1] != new_position[1]:
      line_points.append([old_position[0], old_position[1]])
      e2 = 2 * err
        
      if e2 > -dy:
         err -= dy
         old_position[1] += sx
        
      if e2 < dx:
         err += dx
         old_position[0] += sy
            
   line_points.append([new_position[0], new_position[1]])
   
   # If we run into the wall at any point on the line, return the point just before contact was made
   # This will simulate returning us to the nearest point on the track, in the case of "soft collisions"
   # If using "hard collisions", we can simply check if the value returned is not (None, None), and if it is not then we have to hard reset
   prev_point = line_points[0]
   if prev_point in track.wall:
         return prev_point
   for point in line_points[1:]:
      if point in track.finish_line:
         return point
      if point in track.wall:
         return prev_point
      prev_point = point
    
   return (None, None)

def crossed_finish(old_position, new_position, track):
   """Determines whether a finishin point is encountered
   Uses Besenham's algorithm to find all matrix points crossed in the line taken
   
   Args:
      old_position (array): the previous position [y, x]
      new_position (array): the new position [y, x]
      track (Racetrack): the track that you are currently using
   Returns:
      True if a finishing point is encountered
      False otherwise
   """
   # Bresenham's algorithm
   dx = abs(new_position[1] - old_position[1])
   dy = abs(new_position[0] - old_position[0])
   sx = -1 if old_position[1] > new_position[1] else 1
   sy = -1 if old_position[0] > new_position[0] else 1
   err = dx - dy
    
   line_points = []
   
   # Calculate all points on line
   while old_position[0] != new_position[0] or old_position[1] != new_position[1]:
      line_points.append([old_position[0], old_position[1]])
      e2 = 2 * err
        
      if e2 > -dy:
         err -= dy
         old_position[1] += sx
        
      if e2 < dx:
         err += dx
         old_position[0] += sy
            
   line_points.append([new_position[0], new_position[1]])
   
   # If we run into the wall at any point on the line, return the point just before contact was made
   # This will simulate returning us to the nearest point on the track, in the case of "soft collisions"
   # If using "hard collisions", we can simply check if the value returned is not (-1, -1), and if it is not then we have to hard reset
   for point in line_points:
      if point in track.wall:
         return False
      if point in track.finish_line:
         return True
    
   return False
def get_nearest_start(track, position):
   """Finds the nearest starting position to current position
   Distance is determined by Euclidean distance
   
   Args:
      track (Racetrack): the race track currently being used
      position (array): the current position [y, x]
      
   Returns:
      nearest_point (array): the starting point nearest to position [y, x]
   """
   nearest_dist = 100000
   nearest_point = [10000, 10000]
   for point in track.start_line:
      distance = math.dist(point, position)
      if distance < nearest_dist:
         nearest_dist = distance
         nearest_point = position
   return nearest_point

def epsilon_greedy(q, epsilon):
   if random.random() <= epsilon:
      return np.random.randint(9)
   else:
      return np.unravel_index(q.argmax(), q.shape)[0]