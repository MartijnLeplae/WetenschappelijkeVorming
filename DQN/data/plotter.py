import os
import sys
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import csv

from scipy.interpolate import make_interp_spline

y_values = {}
directory = '../data/TreasureMap-v0/ReprExperiment/'
print(os.listdir(directory))
for filename in os.listdir(directory):
    with open(directory + filename, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            # y_values.append([float(x) for x in row])
            y_values[filename] = [float(x) for x in row]

# plt.figure(1)
# x = np.arange(0, len(y_values[0]), 5)
# y = [float(y_values[0][i]) for i in range(0, len(y_values[0]), 5)]
# plt.plot(x, y)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
flattened = list(chain.from_iterable(y_values.values()))
flattened = [float(x) for x in flattened]
ax.set_ylim(min(flattened), max(flattened))
step = 10
x_values = np.arange(0, max(map(len, y_values.values())), step)

first = y_values['(500)Episode-Length:75-Toggle:0-States:5-BoW:0-StepSize:1-14:55 16-04-2021.csv']
first_y_values = [first[i] for i in range(0, len(first), step)]
second = y_values['(500)Episode-Length:75-Toggle:0-States:1-BoW:4-StepSize:1-15:16 16-04-2021.csv']
second_y_values = [second[i] for i in range(0, len(second), step)]
# ax.plot(x_values, first_y_values)
# ax.plot(x_values, second_y_values)
continuous = np.linspace(x_values.min(), x_values.max(), 500)
ax.plot(continuous, make_interp_spline(x_values, first_y_values)(continuous))
ax.plot(continuous, make_interp_spline(x_values, second_y_values)(continuous))

# for ys in y_values:
#     ys_ = [ys[i] for i in range(0, len(ys), step)]
#     ax.plot(x_values, ys_)
plt.show()

