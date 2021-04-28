import math
import os
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

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
LIGHT_YELLOW = "#FDFFAC"

colors = [[LIGHT_BLUE, DARK_BLUE], [LIGHT_GREEN, LIGHT_GREEN],
          [LIGHT_PINK, DARK_PINK], [LIGHT_PURPLE, DARK_PURPLE],
          [GRAY, BLACK], [LIGHT_YELLOW, YELLOW]]


def read_data_to_array(directory):
    y_values = []
    for filename in os.listdir(directory):
        with open(directory + os.sep + filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                y_values.append([float(x) for x in row])
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


def plot_all_with_err(directory):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    step = 15
    data = os.listdir(directory)
    for index, dir in enumerate(data):
        y_values = read_data_to_array(f"{directory}/{dir}")
        x_values = np.arange(0, max(map(len, y_values)), step)
        #ys = [ys[i] for i in range(0, len(ys))]
        # x_values = np.arange(0, len(ys), step)
        y_err = np.nanstd(y_values, axis=0)
        y_mean = np.nanmean(y_values, axis=0)
        y_err_stepped = y_err[::step]
        y_mean_stepped = y_mean[::step]
        overall_mean = np.nanmean(y_mean)
        ax.fill_between(x_values, np.subtract(y_mean_stepped, y_err_stepped),
                        np.add(y_mean_stepped, y_err_stepped),
                        color=colors[index][0], alpha=0.2) #label="Standaardafwijking"
        plt.plot(x_values, y_mean_stepped, color=colors[index][1], label=f"{dir}")
        # plt.plot(x_values, ys_stepped, label=data[index].replace('_', '-'))
    plt.legend(title="Legenda", loc="lower right")
    plt.show()


def plot_specific_graphs(directory):
    y_values = read_data_to_dict(directory)

    # plt.figure(1)
    # x = np.arange(0, len(y_values[0]), 5)
    # y = [float(y_values[0][i]) for i in range(0, len(y_values[0]), 5)]
    # plt.plot(x, y)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    step = 5
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


def plot_baseline():
    pass


def plot_mean_and_error(directory, baseline_directory, title, step=5):
    y_values = read_data_to_array(directory=directory)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.xlabel('Aantal episodes')
    plt.ylabel('Verkregen beloning')
    x_values_stepped = np.arange(0, max(map(len, y_values)), step)

    # y_values_stepped = []
    # for ys in y_values:
    #     ys_ = [ys[i] for i in range(0, len(ys), step)]
    #     # ax.plot(x_values_stepped, ys_)
    #     y_values_stepped.append(ys_)

    # Calculating variables
    y_err = np.nanstd(y_values, axis=0)
    y_mean = np.nanmean(y_values, axis=0)
    y_err_stepped = y_err[::step]
    y_mean_stepped = y_mean[::step]
    overall_mean = np.nanmean(y_mean)

    # Plotting
    # ax.errorbar(x_values_stepped, y_mean_stepped, y_err_stepped)
    ax.fill_between(x_values_stepped, np.subtract(y_mean_stepped, y_err_stepped), np.add(y_mean_stepped, y_err_stepped),
                    color=LIGHT_BLUE, label="Standaardafwijking", alpha=0.8)
    plt.plot(x_values_stepped, y_mean_stepped, color=DARK_BLUE, label="Gemiddelde per episode")
    plt.axhline(y=overall_mean, color='red', linestyle='-', label=f"Totaal gemiddelde: {int(overall_mean)}")
    # plt.text(plt.gca().get_xlim()[0]+0.5, overall_mean+0.1, str(round(overall_mean, 1)))
    # plt.yticks(list(plt.yticks()[0]) + [overall_mean])

    # Plotting baseline
    step = 5
    x_values_stepped = np.arange(0, max(map(len, y_values)), step)
    y_values = read_data_to_array(baseline_directory)

    y_err = np.nanstd(y_values, axis=0)
    y_mean = np.nanmean(y_values, axis=0)
    y_err_stepped = y_err[::step]
    y_mean_stepped = y_mean[::step]

    ax.fill_between(x_values_stepped, np.subtract(y_mean_stepped, y_err_stepped), np.add(y_mean_stepped, y_err_stepped),
                    color=LIGHT_GREEN, label="Standaardafwijking (baseline)", alpha=0.2)
    plt.plot(x_values_stepped, y_mean_stepped, color=LIGHT_GREEN, label="Gemiddelde per episode (baseline)")

    legend = plt.legend(title="Legenda", loc="lower right", frameon=1)
    legend.get_frame().set_facecolor(PALE_BLUE)
    # legend.get_frame().set_edgecolor('black')

    plot_baseline()

    plt.show()


def plot_all_in_place_combinations300():
    all_ys = read_data_to_array("./TreasureMap-v0/Baseline:AllInPlaceCombinations300")
    y_values = [ys for ys in all_ys if np.nanmean(ys) < 100]
    step = 5
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(r"Trainen m.b.v. \textit{vast} geheugen." + "\n" + r"Vergelijking van alle Observatie/BoW-combinaties")
    plt.xlabel('Aantal episodes')
    plt.ylabel('Verkregen beloning')
    x_values_stepped = np.arange(0, max(map(len, y_values)), step)

    # --- Plotting Lower y-values --- #
    # Calculating variables
    y_err = np.nanstd(y_values, axis=0)
    y_mean = np.nanmean(y_values, axis=0)
    y_err_stepped = y_err[::step]
    y_mean_stepped = y_mean[::step]
    overall_mean = np.nanmean(y_mean)

    # Plotting
    # ax.errorbar(x_values_stepped, y_mean_stepped, y_err_stepped)
    ax.fill_between(x_values_stepped, np.subtract(y_mean_stepped, y_err_stepped), np.add(y_mean_stepped, y_err_stepped),
                    color=LIGHT_BLUE, label=r"(Groep $b$): Standaardafwijking", alpha=0.2)
    plt.plot(x_values_stepped, y_mean_stepped, color=DARK_BLUE, label=r"(Groep $b$): Gemiddelde per episode")
    plt.axhline(y=overall_mean, color='gray', linestyle='dotted', label=fr"(Groep $b$): Totaal Gemiddelde: {int(overall_mean)}")
    # plt.text(plt.gca().get_xlim()[0]+0.5, overall_mean+0.1, str(round(overall_mean, 1)))
    # plt.yticks(list(plt.yticks()[0]) + [overall_mean])

    # --- Plotting Higher y-values --- #
    # Calculating variables
    y_values = [ys for ys in all_ys if np.nanmean(ys) > 100]
    y_err = np.nanstd(y_values, axis=0)
    y_mean = np.nanmean(y_values, axis=0)
    y_err_stepped = y_err[::step]
    y_mean_stepped = y_mean[::step]
    overall_mean = np.nanmean(y_mean)

    # Plotting
    # ax.errorbar(x_values_stepped, y_mean_stepped, y_err_stepped)
    ax.fill_between(x_values_stepped, np.subtract(y_mean_stepped, y_err_stepped), np.add(y_mean_stepped, y_err_stepped),
                    color=LIGHT_PINK, label=r"(Groep $a$): Standaardafwijking", alpha=0.1)
    plt.plot(x_values_stepped, y_mean_stepped, color="#ff0f87", label=r"(Groep $a$): Gemiddelde per episode")
    plt.axhline(y=overall_mean, color='red', linestyle='dotted', label=fr"(Groep $a$): Totaal Gemiddelde: {int(overall_mean)}")
    # plt.text(plt.gca().get_xlim()[0]+0.5, overall_mean+0.1, str(round(overall_mean, 1)))
    # plt.yticks(list(plt.yticks()[0]) + [overall_mean])

    # Plotting baseline
    y_values = read_data_to_array("./TreasureMap-v0/Baseline:StatesOnly300")

    y_err = np.nanstd(y_values, axis=0)
    y_mean = np.nanmean(y_values, axis=0)
    y_err_stepped = y_err[::step]
    y_mean_stepped = y_mean[::step]

    ax.fill_between(x_values_stepped, np.subtract(y_mean_stepped, y_err_stepped), np.add(y_mean_stepped, y_err_stepped),
                    color=LIGHT_GREEN, label="(Baseline): Standaardafwijking", alpha=0.2)
    plt.plot(x_values_stepped, y_mean_stepped, color=LIGHT_GREEN, label="(Baseline): Gemiddelde per episode")

    # Sort legend entries
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    legend = ax.legend(handles, labels, title="Legenda", loc="lower right", frameon=1, ncol=2)

    # legend = plt.legend(title="Legenda", loc="lower right", frameon=1)
    # legend.get_frame().set_facecolor(PALE_BLUE)
    # legend.get_frame().set_edgecolor('black')

    plot_baseline()

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
    data_directory = "./ButtonsWorld-v0/ORGates/3"
    baseline_directory = "./ButtonsWorld-v0/SequenceLength/3"
    # plot_title = r'Trainen m.b.v. \textit{uitgebreid} geheugen $(N = 10)$:' \
    #              + '\n' \
    #              + r'\#Observaties $= 5$ + Geschiedenis som'
    plot_title = "TestPlot"

    # plot_mean_and_error(data_directory, baseline_directory, plot_title)
    plot_all_with_err(data_directory)

