import pathlib
import json

import hydra
from omegaconf.omegaconf import OmegaConf
import pandas as pd
import torch
from hydra import utils
import numpy as np
from training import PneumoniaClsTrainer
from sklearn import metrics
from torch.utils import data

from utils import load_json, load_dump


def save_metrics(metric_dir: pathlib.Path, test_result: dict, proba_threshold: float):
    predcited_proba = torch.cat(tuple(pred["predicted_proba"] for pred in test_result)).numpy()
    true_labels = torch.cat(tuple(pred["true_labels"] for pred in test_result)).numpy()
    image_paths = []

    for pred in test_result:
        image_paths.extend(pred["image_paths"])

    auc_roc = metrics.roc_auc_score(true_labels, predcited_proba)

    with open(metric_dir / "auc.json", "w", encoding="utf-8") as file:
        json.dump({"auc_roc": auc_roc}, file)

    predicted_labels = (predcited_proba > proba_threshold).astype(np.int32)
    predicted_data = pd.DataFrame(
        {"true_labels": true_labels, "predicted_labels": predicted_labels,
         "predcited_proba": predcited_proba, "image_path": image_paths})
    predicted_data.to_csv(metric_dir / "prediction.csv", encoding="utf-8", index=False)


@hydra.main(config_name="test", config_path="configs")
def main(config):
    exp_dir = pathlib.Path(config.exp_dir)
    train_config = OmegaConf.load(exp_dir / "train_config.yaml")

    class_mapping_file = exp_dir / "class_mapping.json"
    class_mapping = load_json(class_mapping_file)
    test_tranform = load_dump(exp_dir / "transforms" / "test_transform.pickle")
    metric_dir = exp_dir / "test_metrics"
    metric_dir.mkdir(exist_ok=True, parents=True)

    model = utils.instantiate(train_config.model)

    model_trainer = PneumoniaClsTrainer.load_from_checkpoint(config.checkpoint_path,
                                                             loss_module=None,
                                                             scheduler_config=None,
                                                             optimizer_config=None,
                                                             class_labels=None,
                                                             model=model,
                                                             proba_threshold=train_config.model_trainer.proba_threshold,
                                                             strict=False)
    del train_config
    model_trainer.to(config.device)
    model_trainer.eval()
    model_trainer.freeze()

    dataset = utils.instantiate(config.dataset, transform=test_tranform,
                                class_mapping=class_mapping)

    loader = data.DataLoader(dataset, batch_size=config.batch_size,
                             shuffle=False, drop_last=False, pin_memory=True,
                             num_workers=config.num_workers)

    result = []
    for batch in loader:
        result.append(model_trainer.predict_step(batch, None))

    save_metrics(metric_dir, result, model_trainer.proba_threshold)


if __name__ == "__main__":
    main()
