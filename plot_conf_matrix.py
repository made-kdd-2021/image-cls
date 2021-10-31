import pathlib

import hydra
import pandas as pd
from omegaconf import OmegaConf

from training.utils import plot_confusion_matrix
from utils import load_json
from sklearn import metrics


@hydra.main(config_path="configs", config_name="plot_conf_matrix")
def main(config):
    exp_dir = pathlib.Path(config.exp_dir)
    pred_file = exp_dir / "test_metrics" / "prediction.csv"
    class_mapping_file = exp_dir / "class_mapping.json"
    out_path = pathlib.Path(config.image_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    train_config = OmegaConf.load(exp_dir / "train_config.yaml")

    class_mapping = load_json(class_mapping_file)

    prediction_data = pd.read_csv(pred_file, encoding="utf-8",
                                  usecols=[config.true_col, config.pred_col],
                                  engine="c")

    conf_matrix = metrics.confusion_matrix(prediction_data[config.true_col],
                                           prediction_data[config.pred_col],
                                           normalize=config.normalize)

    figure = plot_confusion_matrix(conf_matrix, target_names=tuple(
        map(lambda x: x[0], sorted(class_mapping.items(), key=lambda x: x[1]))),
        title=config.title +
        f"\nProbability threshold at {train_config.model_trainer.proba_threshold:.2f}",
        figsize=config.figsize)

    figure.savefig(out_path, dpi=config.dpi, pil_kwargs={
                   "optimize": True, "progressive": True, "quality": 90})


if __name__ == "__main__":
    main()
