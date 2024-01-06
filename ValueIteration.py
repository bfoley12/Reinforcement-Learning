import numpy as np
from Racetrack import Racetrack
from Car import Car
import random
import copy
import Helpers

class ValueIteration:
   def __init__(self, filename):
      """ Initializes the class
      Args:
         filename (string): the name of the file containing the Racetrack
      """
      self.track = Racetrack(filename)
      self.actions = [[-1, -1], [-1, 0], [0, -1], [0, 0], [0, 1], [1, 0], [1, 1], [-1, 1], [1, -1]]
      self.n_states = (len(self.track.track), len(self.track.track[0]), 11, 11)
      self.value_table = np.zeros(self.n_states)
      self.q_table = np.zeros((self.n_states[0], self.n_states[1], self.n_states[2], self.n_states[3], len(self.actions)))
      self.policy_table = np.zeros((self.n_states[0], self.n_states[1], self.n_states[2], self.n_states[3], 2))
      self.num_train_iter = 0
      self.num_test_iter = 0

   def train(self, discount = .9, threshold = .1, crash_type = 0, max_iter = 100):
      """Trains Value Iteration
      
      Args:
         discount (float): the amount of discount to be applied
         threshold (float): the limit at which to stop training
         crash_type (bool): whether the crash should reset to the nearest point (0), or the nearest starting point (1)
         max_iter (int): the max number of iterations through each state to allow
      
      Returns:
         past_value_difference (array): the sequence of differences between deltas in each iteration
         self.num_train_iter (array): the number of iterations of each episode it took to reach the finish line
      """
      start_pos = random.choice(self.track.start_line)
      car = Car(start_pos[0], start_pos[1])
      past_value_difference = []
      converged = False
      # Number of steps
      while self.num_train_iter < max_iter and not converged:
         old_value = copy.deepcopy(self.value_table)
         delta = 0
         # For every state
         # X coordinate
         for y in range(self.n_states[0]):
            # Y coordinate
            for x in range(self.n_states[1]):
               # X velocity
               for vy in range(self.n_states[2]):
                  # Y velocity
                  for vx in range(self.n_states[3]):
                     # If we are in a wall, don't consider actions
                     if self.track.track[y][x] == "#":
                        self.value_table[y][x][vy][vx] = -1
                        continue
                     max_action_value = float('-inf')
                     policy = [0, 0]
                     action_index = 0
                     # For every accelaration possible
                     for action in self.actions:
                        # Set car's state as specified by for loops
                        car.x = x
                        car.y = y
                        car.vx = vx - 5
                        car.vy = vy - 5

                        old_position = [car.y, car.x]
                        old_v = self.value_table[car.y][car.x][car.vy + 5][car.vx + 5]

                        # Advance the car
                        car.step(action[0], action[1])
                        
                        # Check if we finished and/or crashed (both can happen)
                        finished = Helpers.crossed_finish([old_position[0], old_position[1]], [car.y, car.x], self.track)
                        crash = Helpers.did_crash([old_position[0], old_position[1]], [car.y, car.x], self.track)
                        reward = -1

                        # Handle crash if it occurs
                        if crash[0] != None:
                           if crash_type:
                              car.crash_reset(Helpers.get_nearest_start(self.track, crash))
                           else:
                              car.crash_reset(crash)
                        
                        new_v = 0
                        
                        if finished:
                           reward = 0
                        else:
                           new_v = old_value[car.y][car.x][car.vy + 5][car.vx + 5]

                        expected_value = new_v * 0.8 + old_v * 0.2
                        new_q = reward + discount * expected_value
                        self.q_table[y][x][vy][vx][action_index] = new_q
                        
                        if new_q > max_action_value:
                           policy = action
                           max_action_value = new_q
                        
                        action_index += 1

                     # Update Value and policy tables
                     old_q = self.value_table[y][x][vy][vx]
                     self.value_table[y][x][vy][vx] = max_action_value
                     self.policy_table[y][x][vy][vx] = policy

                     # Calculate the maximum value difference
                     delta_q = old_q - max_action_value
                     if delta_q > delta:
                        delta = delta_q
                        
         # Convergence criteria
         if delta < threshold:
            converged = True

         self.num_train_iter += 1
         past_value_difference.append(delta)
      return past_value_difference, self.num_train_iter
   
   def test(self, crash_type = 0):
      """Runs a car from the start to the finish
      
      Args:
         crash_type (bool): whether the crash should reset to the nearest point (0), or the nearest starting point (1)
         
      Returns:
         self.num_test_iter (int): the number of steps required to get to the finish
         steps (array): the list of steps taken
      """
      start_pos = random.choice(self.track.start_line)
      car = Car(start_pos[0], start_pos[1])

      finished = False
      steps = [start_pos]

      while not finished and self.num_test_iter < 1000:
         old_position = [car.y, car.x]
         action = self.policy_table[int(car.y)][int(car.x)][int(car.vy + 5)][int(car.vx + 5)]
         self.num_test_iter += 1
         
         # Nondeterministic step
         if random.random() <= .8:
            car.step(action[0], action[1])
         else:
            car.step(0, 0)
         finished = Helpers.crossed_finish([old_position[0], old_position[1]], [car.y, car.x], self.track)
         crash = Helpers.did_crash([old_position[0], old_position[1]], [car.y, car.x], self.track)
         if crash[0] != None:
            if crash_type:
               car.crash_reset(Helpers.get_nearest_start(self.track, crash))
            else:
               car.crash_reset(crash)
         steps.append([car.y, car.x])
      return self.num_test_iter, steps
