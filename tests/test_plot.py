import numpy as np
import pytest

from matplotlib import pyplot as plt
from training.utils import plot_confusion_matrix, plot_roc


@pytest.mark.parametrize("labels", [None, ["1", "2"]])
def test_plot_conf_matrix(labels):
    gen = np.random.default_rng(123)
    matrix = gen.integers(0, 10, size=(2, 2))

    fig = plot_confusion_matrix(matrix, labels)

    assert len(fig.get_axes()) == 2

    plt.close(fig)


def test_plot_roc():
    gen = np.random.default_rng(123)
    size = 5
    tpr = gen.random(size)
    fpr = gen.random(size)

    fig = plot_roc(tpr, fpr, "test")

    assert len(fig.get_axes()) == 1

    plt.close(fig)
