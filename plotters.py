import numpy
from matplotlib.pyplot import bar
from matplotlib import pyplot


def plot_intervals(time_interval_cardinalities, is_grid=False):
    x = range(len(time_interval_cardinalities))
    y = time_interval_cardinalities.copy()
    pyplot.grid(is_grid)
    pyplot.figure()
    bar(x=x, height=y)
    pyplot.show()


def plot_with_keys_for_multi_value_line_chart(title, y_label, x_label, x_range, x_labels, x_value_dict, is_grid=False):
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)
    pyplot.title(title)
    x = x_range
    pyplot.xticks(x, x_labels)
    pyplot.grid(is_grid)
    for x_value in x_value_dict:
        pyplot.plot(x, x_value["values"], label=x_value["label"], marker=x_value["mark"])
    pyplot.legend()
    pyplot.show()


def plot_with_keys_for_multi_value_bar_chart(title, y_label, x_label, x_value_dict, bar_labels, is_grid=False):
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)
    pyplot.title(title)
    x = numpy.arange(len(bar_labels))
    width = 0.35
    fig, ax = pyplot.subplots()
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.legend()
    for x_value in x_value_dict:
        ax.bar(x - width / 2, x_value["values"], label=x_value["label"])
    fig.tight_layout()
    pyplot.grid(is_grid)
    pyplot.legend()
    pyplot.show()


def plot_line_chart(title, y_label, x_label, y_values, x_values, is_grid=False):
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)
    pyplot.title(title)
    pyplot.grid(is_grid)
    pyplot.plot(y_values, x_values)
    pyplot.show()
