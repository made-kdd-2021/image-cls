from training.utils import plot_confusion_matrix
from model import PneumoniaMobileNetV3
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


if __name__ == "__main__":
    # model = PneumoniaMobileNetV3()

    # dummy_input = torch.rand((1, 3, 256, 256))

    # torch.onnx.export(model, dummy_input, "test.onnx", verbose=True,
    #                   input_names=["image"], output_names=["class_proba"], opset_version=13)

    cm = np.array([[1, 2], [3, 4]])
    fig = plot_confusion_matrix(cm, target_names=["a", "b"])

    plt.show()
