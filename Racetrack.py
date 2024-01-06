import numpy as np
class Racetrack:
   def __init__(self, filename):
      """Initializes a race track
      
      Args:
         filename (string): the name of the file containing the racetrack
      """
      file = open(filename, "r")
      file.readline()
      
      self.track = []
      self.start_line = []
      self.finish_line = []
      self.wall = []
      i = 0
      for line in file:
         line = line.strip()
         self.track.append(line)
         for j in range(len(line)):
            if line[j] == "S":
               self.start_line.append([i, j])
            elif line[j] == "F":
               self.finish_line.append([i, j])
            elif line[j] == "#":
               self.wall.append([i, j])
         i += 1
      file.close()

   def print_track(self):
      """Prints a string representation of the track"""
      track_string = ""
      for x in range(len(self.track[0])):
         for y in range(len(self.track)):
            track_string += self.track[y][x]
         track_string += "\n"

      print(track_string)