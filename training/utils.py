from typing import Sequence
import matplotlib

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm,
                          target_names: Sequence[str],
                          title='Confusion matrix',
                          cmap=None):
    """
    given a confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph


    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    figure = plt.figure()
    ax = figure.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)

    plt.colorbar(im)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels=target_names)
        ax.set_yticklabels(labels=target_names)

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            text = ax.text(j, i,  f"{cm[i, j]:.4f}", ha="center", va="center", color="black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    plt.tight_layout()

    return figure


def plot_roc(fpr, tpr, label: str):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_label(label)
    ax.grid(True)

    return figure
