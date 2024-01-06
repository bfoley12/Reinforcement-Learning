import random
class Car:
   def __init__(self, y, x):
      """Initializes a car
      
      Args:
         x (int): the x-position of the car
         y (int): the y-position of the car"""
      self.x = x
      self.y = y
      self.vx = 0
      self.vy = 0
      self.ax = 0
      self.ay = 0

   def step(self, ay, ax):
      """Updates the state of the car
      
      Velocities are clipped to range [-5, 5]

      Args:
         ay (int): acceleration in the y-direction
         ax (int): acceleration in the x-direction
      """
      self.ax = ax
      self.ay = ay

      self.vx += ax
      if self.vx > 5:
         self.vx = 5
      elif self.vx <-5:
         self.vx = -5
      
      self.vy += ay
      if self.vy > 5:
         self.vy = 5
      elif self.vy <-5:
         self.vy = -5
      
      self.x += self.vx
      self.y += self.vy
   
   def crash_reset(self, position):
      """Handles reset after crash
      Velocities get set to 0 on crash

      Args:
         position (array): the y and x values to reset to [y, x]"""
      self.y = position[0]
      self.x = position[1]
      self.vx = 0
      self.vy = 0