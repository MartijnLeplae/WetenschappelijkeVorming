import math
import os
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

from scipy.interpolate import make_interp_spline


def read_data_to_array(directory):
    y_values = []
    print(os.listdir(directory))
    for filename in os.listdir(directory):
        with open(directory + filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                y_values.append([float(x) for x in row])
    return y_values


def read_data_to_dict(directory):
    y_values = {}
    print(os.listdir(directory))
    for filename in os.listdir(directory):
        with open(directory + filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                y_values[filename] = [float(x) for x in row]
    return y_values


def plot_specific_graphs():
    directory = '../data/TreasureMap-v0/ReprExperiment/'
    y_values = read_data_to_dict(directory)

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

    plt.show()


def get_min_max(y_values):
    if type(y_values) == dict:
        flattened = list(chain.from_iterable(y_values.values()))
    else:
        flattened = list(chain.from_iterable(y_values))
    flattened = [float(x) for x in flattened]
    return min(flattened), max(flattened)


def plot_mean_and_error(directory, title, step=5):
    y_values = read_data_to_array(directory=directory)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.xlabel('Aantal episodes')
    plt.ylabel('Verkregen beloning')
    x_values_stepped = np.arange(0, max(map(len, y_values)), step)

    y_values_stepped = []
    for ys in y_values:
        ys_ = [ys[i] for i in range(0, len(ys), step)]
        # ax.plot(x_values_stepped, ys_)
        y_values_stepped.append(ys_)

    # Calculating variables
    y_err = np.nanstd(y_values, axis=0)
    y_mean = np.nanmean(y_values, axis=0)
    y_err_stepped = y_err[::step]
    y_mean_stepped = y_mean[::step]
    overall_mean = np.nanmean(y_mean)

    # Plotting
    # ax.errorbar(x_values_stepped, y_mean_stepped, y_err_stepped)
    ax.fill_between(x_values_stepped, np.subtract(y_mean_stepped, y_err_stepped), np.add(y_mean_stepped, y_err_stepped),
                    color="#9aaaf7", label="Standaardafwijking")
    plt.plot(x_values_stepped, y_mean_stepped, color='white', label="Gemiddelde per episode")
    plt.axhline(y=overall_mean, color='red', linestyle='-', label=f"Totaal gemiddelde: {int(overall_mean)}")
    # plt.text(plt.gca().get_xlim()[0]+0.5, overall_mean+0.1, str(round(overall_mean, 1)))
    # plt.gca().annotate('Totaal gemiddelde',
    #                    xy=(plt.gca().get_xlim()[0], overall_mean),
    #                    xytext=(plt.gca().get_xlim()[0], overall_mean+0.3),
    #                    ha='left',
    #                    arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.yticks(list(plt.yticks()[0]) + [overall_mean])
    legend = plt.legend(title="Legenda", loc="lower right", frameon=1)
    legend.get_frame().set_facecolor('#d6ecfa')
    # legend.get_frame().set_edgecolor('black')
    plt.show()


if __name__ == '__main__':
    # Set default save directory
    matplotlib.rcParams["savefig.directory"] = "../../finishedGraphs"
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    data_directory = "./TreasureMap-v0/VGL1/InPlaceMemory/"
    plot_title = r'Trainen met behulp van \textit{vast} geheugen $(N = 10)$:' \
                 "\n" \
                 r'\#Observaties $= 6$'
    plot_mean_and_error(data_directory, plot_title)
