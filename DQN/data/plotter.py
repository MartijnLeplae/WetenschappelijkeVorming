import math
import os
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import BSpline

from scipy.interpolate import make_interp_spline

LIGHT_BLUE = "#8497f4"  # "#9aaaf7"
DARK_BLUE = "#0000e0"
LIGHT_GREEN = "#30eab2"
PALE_BLUE = "#d6ecfa"
LIGHT_PINK = "#ff8ac4"  # "#ff1fff"
DARK_PINK = "#a300a3"
DARK_PURPLE = "#EF00FF"
LIGHT_PURPLE = "#F557FF"
BLACK = "#000000"
GRAY = "#7D7D7D"
YELLOW = "#F7FF00"
LIGHT_YELLOW = "#b7ba10ff"
DARK_GREEN = "#006633"


colors = [[LIGHT_BLUE, DARK_BLUE], [LIGHT_GREEN, LIGHT_GREEN],
          [LIGHT_PINK, DARK_PINK], [LIGHT_PURPLE, DARK_PURPLE],
          [GRAY, BLACK], [LIGHT_YELLOW, LIGHT_YELLOW], [DARK_GREEN, DARK_GREEN]]


def read_data_to_array(directory, step=1):
    y_values = []
    for filename in os.listdir(directory):
        with open(directory + os.sep + filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                y_values.append([float(row[x]) for x in range(0, len(row), step)])
    return y_values


def read_data_to_dict(directory):
    y_values = {}
    for filename in os.listdir(directory):
        with open(directory + os.sep + filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                y_values[filename] = [float(x) for x in row]
    return y_values


def plot_all(directory, step=5):
    y_values = read_data_to_array(directory)
    plt.figure(1)
    data = os.listdir(directory)
    x_values = np.arange(0, max(map(len, y_values)), step)
    for index, ys in enumerate(y_values):
        ys_stepped = [ys[i] for i in range(0, len(ys), step)]
        # x_values = np.arange(0, len(ys), step)
        plt.plot(x_values, ys_stepped, label=data[index].replace('_', '-'))
    plt.legend(title="Legenda", loc="lower right")
    plt.show()


def plot_all_with_err(directory, step=10):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = os.listdir(directory)
    for index, dir in enumerate(data):
        y_values = read_data_to_array(f"{directory}/{dir}")
        x_values = np.arange(0, max(map(len, y_values)), step)
        y_err = np.nanstd(y_values, axis=0)
        y_mean = np.nanmean(y_values, axis=0)
        y_err_stepped = y_err[::step]
        y_mean_stepped = y_mean[::step]
        # overall_mean = np.nanmean(y_mean)
        x_new = np.linspace(x_values.min(), x_values.max(), 200)
        spl1 = make_interp_spline(x_values, y_err_stepped)
        spl2 = make_interp_spline(x_values, y_mean_stepped)
        y_err_new = spl1(x_new)
        y_mean_new = spl2(x_new)
        # ax.fill_between(x_values, np.subtract(y_mean_stepped, y_err_stepped),
        #                 np.add(y_mean_stepped, y_err_stepped),
        #                 color=colors[index][0], alpha=0.2) #label="Standaardafwijking"
        plt.plot(x_values, y_mean_stepped, color=colors[index][1], label=f"{dir}")
    plt.legend(title="Legenda", loc="lower right")
    plt.show()


if __name__ == '__main__':
    # Set default save directory
    matplotlib.rcParams["savefig.directory"] = "../../finishedGraphs"
    # Enable LaTeX for plot titles and text
    plt.rc('text', usetex=True)
    # Set LaTeX font
    plt.rc('font', family='serif')
    # Set window size
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    # Set font size
    matplotlib.rcParams.update({'font.size': 35})

    # Directory that contains data to be plotted
    data_directory = "./ButtonsWorld-v0/SequenceLength/3/75"
    baseline_directory = "./ButtonsWorld-v0/SequenceLength/3"
    # plot_title = r'Trainen m.b.v. \textit{uitgebreid} geheugen $(N = 10)$:' \
    #              + '\n' \
    #              + r'\#Observaties $= 5$ + Geschiedenis som'
    plot_title = "TestPlot"

    # plot_mean_and_error(data_directory, baseline_directory, plot_title)
    plot_all_with_err(data_directory, step=25)

