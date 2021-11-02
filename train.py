import pathlib
import json
import os

import hydra
import pandas as pd
import torch
from hydra import utils
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from training import ChestXrayDataModule, PneumoniaClsTrainer
from sklearn import metrics
from omegaconf import OmegaConf

from utils import dump_object, dump_json


def save_metrics(metric_dir: pathlib.Path, test_result: dict):
    predcited_proba = torch.cat(tuple(pred["predicted_proba"] for pred in test_result)).numpy()
    true_labels = torch.cat(tuple(pred["true_labels"] for pred in test_result)).numpy()
    image_paths = []

    for pred in test_result:
        image_paths.extend(pred["image_paths"])

    auc_roc = metrics.roc_auc_score(true_labels, predcited_proba)

    with open(metric_dir / "auc.json", "w", encoding="utf-8") as file:
        json.dump({"auc_roc": auc_roc}, file)

    predicted_labels = (predcited_proba > 0.5).astype(np.int32)
    predicted_data = pd.DataFrame(
        {"true_labels": true_labels, "predicted_labels": predicted_labels,
         "predcited_proba": predcited_proba, "image_path": image_paths})
    predicted_data.to_csv(metric_dir / "prediction.csv", encoding="utf-8", index=False)


@hydra.main(config_name="training", config_path="configs")
def main(config):
    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    log_dir = pathlib.Path(config.trainer.logger.save_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    metric_dir = exp_dir / "metrics"
    metric_dir.mkdir(exist_ok=True, parents=True)
    transform_dir = exp_dir / "transforms"
    transform_dir.mkdir(exist_ok=True, parents=True)

    model = utils.instantiate(config.model)
    model_trainer: PneumoniaClsTrainer = utils.instantiate(config.model_trainer,
                                                           loss_module=utils.instantiate(
                                                               config.loss),
                                                           scheduler_config=config.model_trainer.get(
                                                               "scheduler_config", None),
                                                           model=model,
                                                           _recursive_=False)

    trainer: Trainer = utils.instantiate(config.trainer)

    train_transform = utils.instantiate(config.transforms.train_transform)
    test_tranform = utils.instantiate(config.transforms.test_transform)

    dump_object(transform_dir / "train_transform.pickle", train_transform)
    dump_object(transform_dir / "test_transform.pickle", test_tranform)
    dump_json(exp_dir / "class_mapping.json", dict(config.datamodule.class_mapping))

    with open(exp_dir / "train_config.yaml", "w", encoding="utf-8") as dump_config:
        dump_config.write(OmegaConf.to_yaml(config))

    datamodule: ChestXrayDataModule = utils.instantiate(config.datamodule, train_transforms=train_transform,
                                                        test_transforms=test_tranform, val_transforms=test_tranform)

    trainer.fit(model_trainer, datamodule=datamodule)

    if config.trainer.fast_dev_run:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                trainer.save_checkpoint(os.path.join(callback.dirpath, "last.ckpt"))


if __name__ == "__main__":
    main()
