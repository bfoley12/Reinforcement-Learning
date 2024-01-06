import numpy as np
from Racetrack import Racetrack
from Car import Car
import random
import Helpers

# --------- Q-Learning ---------
class QLearning:
   def __init__(self, filename):
      """ Initializes the class
      Args:
         filename (string): the name of the file containing the Racetrack
      """
      self.track = Racetrack(filename)

      self.n_states = (len(self.track.track), len(self.track.track[0]), 11, 11)
      self.actions = [[-1, -1], [-1, 0], [0, -1], [0, 0], [0, 1], [1, 0], [1, 1], [-1, 1], [1, -1]]
      self.q_table = np.zeros((self.n_states[0], self.n_states[1], self.n_states[2], self.n_states[3], len(self.actions)))
   
   def train(self, discount = .9, epsilon = .4, decay = .99, learning_rate = .8, num_iter = 1000, crash_type = 0):
      """Trains Q-Learning
      
      Args:
         discount (float): the discount factor to be applied
         epsilon (float): the percentage of exploration to take
         decay (float): the amount of decay to apply
         learning_rate(float): how fast the algorithm should learn
         num_iter (int): the number of episodes/iterations to take
         crash_type (bool): whether the crash should reset to the nearest point (0), or the nearest starting point (1)
      
      Returns:
         self.num_train_iter (array): the number of iterations of each episode it took to reach the finish line
      """
      self.num_train_iter = []
      #sas = []
      for i in range(num_iter):
         start_pos = random.choice(self.track.start_line)
         car = Car(start_pos[0], start_pos[1])

         step = 0
         finished = False
         while not finished:
            reward = -1
            old_position = [car.y, car.x]
            q_vals = self.q_table[car.y][car.x][car.vy + 5][car.vx + 5]
            #s1 = (car.y, car.x, car.vy, car.vx)
            index = Helpers.epsilon_greedy(q_vals, epsilon)

            q_val = q_vals[3]
            action = [0, 0]

            # Nondeterministic Step
            if random.random() <= .8:
               q_val = q_vals[index]
               action = self.actions[index]
            #a = (action[0], action[1])
            
            car.step(action[0], action[1])
            #s2 = (car.y, car.x, car.vy, car.vx)

            # Adjust reward if we crash or finish
            finished = Helpers.crossed_finish([old_position[0], old_position[1]], [car.y, car.x], self.track)
            crash = Helpers.did_crash([old_position[0], old_position[1]], [car.y, car.x], self.track)
            if crash[0] != None:
               reward = -10
               if crash_type:
                  car.crash_reset(Helpers.get_nearest_start(self.track, crash))
               else:
                  car.crash_reset(crash)
            if finished:
               reward = 0

            # Update q-table values
            q_values_prime = self.q_table[car.y][car.x][car.vy + 5][car.vx + 5]
            max_q_value_prime = np.max(q_values_prime)
            q_vals[index] += learning_rate * (reward + discount * max_q_value_prime - q_val)

            step += 1
            #sas.append([s1, a, s2])
         # Handle decay
         epsilon *= decay
         if learning_rate > 0.01:
            learning_rate *= decay

         self.num_train_iter.append(step)
      return self.num_train_iter

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

      self.num_test_iter = 0
      finished = False
      steps = [start_pos]
      epsilon = 0

      while not finished and self.num_test_iter < 1000:
         old_position = [car.y, car.x]

         q_vals = self.q_table[car.y][car.x][car.vy + 5][car.vx + 5]

         index = Helpers.epsilon_greedy(q_vals, epsilon)
         action = self.actions[index]
         
         self.num_test_iter += 1

         # Nondeterministic Step
         if random.random() <= .8:
            car.step(action[0], action[1])
         else:
            car.step(0, 0)

         # Check if we crashed or finished
         finished = Helpers.crossed_finish([old_position[0], old_position[1]], [car.y, car.x], self.track)
         crash = Helpers.did_crash([old_position[0], old_position[1]], [car.y, car.x], self.track)
         if crash[0] != None:
            if crash_type:
               car.crash_reset(Helpers.get_nearest_start(self.track, crash))
            else:
               car.crash_reset(crash)
         steps.append([car.y, car.x])
      return self.num_test_iter, steps