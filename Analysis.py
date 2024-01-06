from ValueIteration import ValueIteration
from QLearning import QLearning
from SARSA import SARSA
import csv
import time
import os

def write_csv(filename, rows, header):
   """ Writes to a CSV
   
   Args:
      filename (string): the filename to write to
      rows (matrix): the data to write
      header(array): the header row
   """
   with open(filename, 'w', newline="", encoding='UTF8') as f:
      writer = csv.writer(f)

      # write the header
      writer.writerow(header)

      # write the data
      for row in rows:
         writer.writerow(row)

csv_rows = []
def test_ValueIteration():
   """Tests various values of hyperparameters for Value Iteration"""
   tracks = ["./L-track-1.txt", "O-track-1.txt", "R-track-1.txt"]

   discount_values = [.75, .85, .9, .95, .99]
   threshold_values = [.05, .1, .2]

   for track in tracks:
      for discount in discount_values:
         for threshold in threshold_values:
            for crash_type in [0, 1]:
               if crash_type == 1 and track != "R-track-1.txt":
                  continue
               os.system("cls")
               print("Track: " + track)
               print("Discount: " + str(discount))
               print("Threshold: " + str(threshold))
               print("Crash Type " + str(crash_type))
               model = ValueIteration(track)
               train_start = time.time()
               past_values, num_train_iters = model.train(discount = discount, threshold = threshold, crash_type = crash_type)
               train_end = time.time()
               num_test_iters, steps = model.test(crash_type = crash_type)
               test_end = time.time()
               training_time = train_end - train_start
               test_time = test_end - train_end
               csv_rows.append([track, crash_type, discount, threshold, num_train_iters, num_test_iters, training_time, test_time])

   header = ["track", "crash_type", "discount", "threshold", "num_train_iters", "num_test_iters", "training_time", "test_time"]

   write_csv('ValueIterationEvaluation.csv', csv_rows, header)

def experiment_ValueIteration():
   """ Performs the data collection for 10 runs of Value Iteration"""
   csv_rows = []
   optimal_discount = .9
   optimal_threshold = .1

   tracks = ["L-track-1.txt", "O-track-1.txt", "R-track-1.txt"]
   for track in tracks:
      for i in range(10):
         for crash_type in [0, 1]:
            max_iters = 1000
            if crash_type == 1 and track != "R-track-1.txt":
               continue
            print("Track: " + track)
            print("Iteration: " + str(i))
            model = ValueIteration(track)
            past_values, num_train_iters = model.train(discount = optimal_discount, threshold = optimal_threshold, crash_type = crash_type, max_iter=max_iters)
            num_test_iters, steps = model.test(crash_type = crash_type)
            csv_rows.append([track, i, crash_type, num_train_iters, num_test_iters, past_values, steps])
   header = ["track", "iteration", "crash_type", "num_train_iters", "num_test_iters", "past_values", "steps"]
   write_csv('ValueIterationExperiment.csv', csv_rows, header)

def experiment_QLearning():
   """ Performs the data collection for 10 runs of Q-Learning"""
   csv_rows = []
   discount = .9
   epsilon = .6
   decay = .9999
   learning_rate = .8
   max_iters = 10000
   tracks = ["L-track-1.txt", "O-track-1.txt", "R-track-1.txt"]
   for track in tracks:
      for i in range(10):
         for crash_type in [0, 1]:
            max_iters = 1000
            discount = .8
            if crash_type == 1 and track != "R-track-1.txt":
               continue
            print("Track: " + track)
            print("Iteration: " + str(i))
            model = QLearning(track)
            num_train_iters = model.train(discount = discount, epsilon = epsilon, decay = decay, learning_rate = learning_rate, num_iter = max_iters, crash_type = crash_type)
            num_test_iters, steps = model.test(crash_type = crash_type)
            csv_rows.append([track, i, crash_type, num_train_iters, num_test_iters, steps])
   header = ["track", "iteration", "crash_type", "num_train_iters", "num_test_iters", "steps"]
   write_csv('QLearningExperiment-2.csv', csv_rows, header)

def experiment_SARSA():
   """ Performs the data collection for 10 runs of SARSA"""
   csv_rows = []
   
   tracks = ["L-track-1.txt", "O-track-1.txt", "R-track-1.txt"]
   for track in tracks:
      if track == "R-track-1.txt":
         ipe = 10000
      else:
         ipe = 1000
      for i in range(10):
         for crash_type in [0, 1]:
            if crash_type == 1 and track != "R-track-1.txt":
               continue
            
            print("Track: " + track)
            print("Iteration: " + str(i))
            model = SARSA(track)
            num_train_iters, episode_rewards = model.train(num_episodes = 10000, iter_per_episode = ipe, discount = .9, epsilon = .1, decay = .99, learning_rate = .1, crash_type = crash_type)
            num_test_iters, steps = model.test(crash_type = crash_type)
            csv_rows.append([track, i, crash_type, num_train_iters, num_test_iters, steps, episode_rewards])
   header = ["track", "iteration", "crash_type", "num_train_iters", "num_test_iters", "steps", "episode_rewards"]
   write_csv('SARSAExperiment.csv', csv_rows, header)

experiment_ValueIteration()
experiment_QLearning()
experiment_SARSA()
