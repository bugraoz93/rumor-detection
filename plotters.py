from matplotlib.pyplot import bar
from matplotlib import pyplot


def plot_intervals(time_interval_cardinalities):
    x = range(len(time_interval_cardinalities))
    y = time_interval_cardinalities.copy()
    pyplot.figure()
    bar(x=x, height=y)
    pyplot.show()
