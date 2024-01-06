import numpy as np
from Racetrack import Racetrack
from Car import Car
import random
import Helpers

class SARSA:
   def __init__(self, filename):
      """ Initializes the class
      Args:
         filename (string): the name of the file containing the Racetrack
      """
      self.track = Racetrack(filename)

      self.n_states = (len(self.track.track), len(self.track.track[0]), 11, 11)
      self.actions = [[-1, -1], [-1, 0], [0, -1], [0, 0], [0, 1], [1, 0], [1, 1], [-1, 1], [1, -1]]
      self.q_table = np.random.rand(self.n_states[0], self.n_states[1], self.n_states[2], self.n_states[3], len(self.actions))

   def train(self, num_episodes = 10000, iter_per_episode = 100, discount = .9, epsilon = .1, decay = .99, learning_rate = .1, crash_type = 0):
      """Trains Q-Learning
      
      Args:
         num_episodes (int): the number of episodes to iterate through
         iter_per_episode (int): the number of iterations to take within each episode
         discount (float): the discount factor to be applied
         epsilon (float): the percentage of exploration to take
         decay (float): the amount of decay to apply
         learning_rate(float): how fast the algorithm should learn
         crash_type (bool): whether the crash should reset to the nearest point (0), or the nearest starting point (1)
      
      Returns:
         num_episodes * iter_per_episode: the total number of steps taken in the training
         episode_rewards (array): the cumulative reward from each episode
      """
      start_pos = random.choice(self.track.start_line)
      car = Car(start_pos[0], start_pos[1])

      reward = -1
      episode_rewards = []
      # Iterate through all episodes
      for episode in range(num_episodes):
         for point in self.track.finish_line:
            self.q_table[point[0]][point[1]][:][:][:] = 0
         
         # Get initial state
         start_pos = random.choice(self.track.start_line)
         y = start_pos[0]
         x = start_pos[1]
         vy = 0
         vx = 0

         # Get action, epsilon greedy
         index = Helpers.epsilon_greedy(self.q_table[y][x][vy][vx], epsilon)
         action = self.actions[index]
         
         # Instantiate car
         car.y = y
         car.x = x
         car.vx = vx
         car.vy = vy

         episode_reward = 0

         # Iterate through episode iterations
         for i in range(iter_per_episode):
            # If we are in the wall of finished, do not consider
            if self.track.track[y][x] == "F" or self.track.track[y][x] == "#":
               break
            old_position = [y, x]

            # Nondeterministic step
            if random.random() <= .8:
               car.step(action[0], action[1])
            else:
               car.step(0, 0)

            # Check whether we crashed or finished
            finished = Helpers.crossed_finish([old_position[0], old_position[1]], [car.y, car.x], self.track)
            crash = Helpers.did_crash([old_position[0], old_position[1]], [car.y, car.x], self.track)
            
            # Handle crash
            if crash[0] != None:
               if crash_type:
                  car.crash_reset(Helpers.get_nearest_start(self.track, crash))
               else:
                  car.crash_reset(crash)

            # Get next action
            index_prime = Helpers.epsilon_greedy(self.q_table[car.y][car.x][car.vy + 5][car.vx + 5], epsilon)
            
            # Update Q-table
            self.q_table[y][x][vy + 5][vx + 5][index] += learning_rate * (reward + decay * self.q_table[car.y][car.x][car.vy + 5][car.vx + 5][index_prime] - self.q_table[y][x][vy + 5][vx + 5][index])
            
            # Set original state, s, to our new state, s`
            y = car.y
            x = car.x
            vy = car.vy
            vx = car.vx
            index = index_prime
            action = self.actions[index_prime]

            episode_reward += reward
         episode_rewards.append(episode_reward)
      return num_episodes * iter_per_episode, episode_rewards
      
## Simulation
   def test(self, crash_type):
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
         finished = Helpers.crossed_finish([old_position[0], old_position[1]], [car.y, car.x], self.track)
         crash = Helpers.did_crash([old_position[0], old_position[1]], [car.y, car.x], self.track)
         if crash[0] != None:
            if crash_type:
               car.crash_reset(Helpers.get_nearest_start(self.track, crash))
            else:
               car.crash_reset(crash)
         steps.append([car.y, car.x])
      return self.num_test_iter, steps
