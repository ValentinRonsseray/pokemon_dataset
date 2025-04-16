from matplotlib.axes import Axes  # Import du type Axis

def add_count_labels(ax: Axes, fontsize=10):
    """
    Adds the counts above the bars of a seaborn countplot.

    Parameters:
    - ax: matplotlib axes object returned by sns.countplot
    - fontsize: font size for the count labels
    """
    for p in ax.patches:
        height = int(round(p.get_height()))
        if height > 0:
            ax.annotate(f'{height}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=fontsize)