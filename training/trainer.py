import logging
from typing import Sequence, Dict

from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from torch import nn
from torch import Tensor
import torch
from sklearn import metrics
import numpy as np

from model import PneumoniaMobileNetV3

from utils import plot_confusion_matrix, plot_roc

InputType = Dict[str, Tensor]


class PneumoniaClsTrainer(LightningModule):
    def __init__(self, *, model: PneumoniaMobileNetV3, optimizer_config,
                 loss_module: nn.Module,
                 class_labels: Sequence[str],
                 proba_threshold: float,
                 scheduler_config=None):
        super().__init__()
        self._logger = logging.getLogger("trainer")
        self._loss = loss_module
        self._model = model
        self._scheduler_config = scheduler_config
        self._optimizer_config = optimizer_config
        self._class_labels = class_labels
        self._train_prefix = "Train"
        self._valid_prefix = "Valid"
        self._test_prefix = "Test"
        self.proba_threshold = proba_threshold

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

    def _compute_pre_metrics(self, predicted_logits: Tensor, true_labels: Tensor):
        predicted_logits = predicted_logits.detach()
        true_labels = true_labels.detach()
        pos_class_proba = self._model.predict_proba(predicted_logits)

        return {"class_proba": pos_class_proba, "true_label": true_labels}

    def _compute_loss(self, image_batch_info: InputType):
        labels = image_batch_info["label"]
        images = image_batch_info["image"]

        predicted_logits = self._model.predict_logits(images)
        loss_value = self._loss(predicted_logits, labels.float())

        return predicted_logits, labels, loss_value

    def training_step(self, image_batch_info: InputType, dataset_idx):
        predicted_logits, labels, loss_value = self._compute_loss(image_batch_info)
        pre_metrics = self._compute_pre_metrics(predicted_logits, labels)
        self.log(f"{self._train_prefix}/NLL loss", loss_value, on_epoch=True)
        pre_metrics.update({"loss": loss_value})
        return pre_metrics

    def _compute_metrics(self, prediction_res: dict):
        true_labels = torch.cat(tuple(value["true_label"]
                                for value in prediction_res)).cpu().numpy()
        class_proba = torch.cat(tuple(value["class_proba"]
                                for value in prediction_res)).cpu().numpy()
        conf_matrix = metrics.confusion_matrix(
            true_labels, (class_proba > self.proba_threshold).astype(np.int32), normalize="all")
        auc_roc = metrics.roc_auc_score(true_labels, class_proba)
        fpr, tpr, threholds = metrics.roc_curve(true_labels, class_proba)

        return conf_matrix, auc_roc, (fpr, tpr, threholds)

    def _log_metrics(self, prefix_stage: str, prediction_res: dict):
        conf_matrix, auc_roc, (fpr, tpr, _) = self._compute_metrics(prediction_res)

        if self.logger is not None:
            fig = plot_confusion_matrix(conf_matrix, self._class_labels,
                                        f"{prefix_stage} confusion matrix at threshold {self.proba_threshold:.2f}")

            self.logger.experiment.add_figure(
                f"{prefix_stage}/confusion_matrix", fig, global_step=self.current_epoch)

            fig = plot_roc(fpr, tpr, "Train confusion matrix")

            self.logger.experiment.add_figure(
                f"{prefix_stage}/ROC_curve", fig, global_step=self.current_epoch)

        self.log(f"{prefix_stage}/AUC_ROC", auc_roc)

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self._log_metrics(self._train_prefix, outputs)

    def validation_step(self, image_batch_info: InputType, dataset_idx):
        predicted_logits, labels, loss_value = self._compute_loss(image_batch_info)
        pre_metrics = self._compute_pre_metrics(predicted_logits, labels)
        self.log(f"{self._valid_prefix}/NLL loss", loss_value, on_epoch=True)
        pre_metrics.update({"loss": loss_value})
        return pre_metrics

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self._log_metrics(self._valid_prefix, outputs)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()

    def test_step(self, image_batch_info: InputType, dataset_idx=None):
        predicted_logits, labels, _ = self._compute_loss(image_batch_info)
        pre_metrics = self._compute_pre_metrics(predicted_logits, labels)
        return pre_metrics

    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)
        conf_matrix, auc_roc, (fpr, tpr, _) = self._compute_metrics(outputs)
        self.log_dict({"auc_roc": auc_roc})

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        labels = batch["label"].to(self.device)
        images = batch["image"].to(self.device)
        predicted_logits = self._model.predict_logits(images)
        predcited_proba = self._model.predict_proba(predicted_logits)
        return {"predicted_proba": predcited_proba.cpu(), "true_labels": labels.cpu(), "image_paths": batch["image_path"]}

    def configure_optimizers(self):
        optimizer = instantiate(self._optimizer_config, self._model.parameters())

        if self._scheduler_config is not None:
            scheduler = instantiate(self._scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            self._logger.info("Scheduler is None. Use constant learning rate")

        return optimizer
