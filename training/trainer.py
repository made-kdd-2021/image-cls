import logging
from typing import Sequence, Dict

from pytorch_lightning import LightningModule
import torch
from torchmetrics import ConfusionMatrix, AUROC, ROC
from hydra.utils import instantiate
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from model import PneumoniaMobileNetV3

from .utils import plot_confusion_matrix, plot_roc

InputType = Dict[str, Tensor]


class PneumoniaClsTrainer(LightningModule):
    def __init__(self, *, model: PneumoniaMobileNetV3, optimizer_config,
                 loss_module: nn.Module,
                 class_labels: Sequence[str],
                 scheduler_config=None,
                 unfreeze_after_epoch: int = 0):
        super().__init__()
        self._logger = logging.getLogger("trainer")
        self._loss = loss_module
        self._model = model
        self._scheduler_config = scheduler_config
        self._optimizer_config = optimizer_config
        self._logger.info("Freeze all model")
        self._model.freeze()
        self._logger.info("Unfreeze classification head")
        self._model.unfreeze_cls()
        self._unfreeze_after_epoch = unfreeze_after_epoch
        self._class_labels = class_labels
        self._confusion_metric = None
        self._roc_metric = None
        self._auc_roc = None
        self._train_prefix = "Train"
        self._valid_prefix = "Valid"
        self._test_prefix = "Test"
        self._reset_metrics()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        if self.current_epoch > self._unfreeze_after_epoch:
            self._logger.info("Unfreeze model")
            self._model.unfreeze_model()

        self._reset_metrics()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._reset_metrics()

    def _reset_metrics(self):
        num_classes = len(self._class_labels)
        self._confusion_metric = ConfusionMatrix(
            num_classes, normalize="all", compute_on_step=False)
        self._confusion_metric.to(self.device)
        self._auc_roc = AUROC(None if num_classes == 2 else num_classes, compute_on_step=False)
        self._auc_roc.to(self.device)
        self._roc = ROC(None if num_classes == 2 else num_classes, compute_on_step=False)
        self._roc.to(self.device)

    def _update_metrics(self, predicted_logits: Tensor, true_labels: Tensor):
        predicted_logits = predicted_logits.detach()
        true_labels = true_labels.detach()
        self._confusion_metric.update(predicted_logits.to(), true_labels)
        pos_class_proba = F.softmax(predicted_logits, dim=-1)[:, 1]
        self._auc_roc.update(pos_class_proba, true_labels)
        self._roc.update(pos_class_proba, true_labels)

    def _compute_loss(self, image_batch_info: InputType):
        labels = image_batch_info["label"]
        images = image_batch_info["image"]

        predicted_logits = self._model.predict_logits(images)
        loss_value = self._loss(predicted_logits, labels)

        return predicted_logits, labels, loss_value

    def training_step(self, image_batch_info: InputType, dataset_idx):
        predicted_logits, labels, loss_value = self._compute_loss(image_batch_info)
        self._update_metrics(predicted_logits, labels)
        self.log(f"{self._train_prefix}/NLL loss", loss_value, on_epoch=True)

        return loss_value

    def _compute_metrics(self):
        conf_matrix = self._confusion_metric.compute().cpu()
        auc_roc = self._auc_roc.compute().cpu()
        fpr, tpr, threholds = self._roc.compute()

        return conf_matrix, auc_roc, (fpr.cpu(), tpr.cpu(), threholds.cpu())

    def _log_metrics(self, prefix_stage: str):
        conf_matrix, auc_roc, (fpr, tpr, _) = self._compute_metrics()

        fig = plot_confusion_matrix(conf_matrix, self._class_labels,
                                    f"{prefix_stage} confusion matrix")

        if self.logger is not None:
            self.logger.experiment.add_figure(
                f"{prefix_stage}/confusion_matrix", fig, global_step=self.current_epoch)

            fig = plot_roc(fpr, tpr, "Train confusion matrix")

            self.logger.experiment.add_figure(
                f"{prefix_stage}/ROC_curve", fig, global_step=self.current_epoch)

        self.log(f"{prefix_stage}/AUC_ROC", auc_roc)

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self._log_metrics(self._train_prefix)

    def validation_step(self, image_batch_info: InputType, dataset_idx):
        predicted_logits, labels, loss_value = self._compute_loss(image_batch_info)
        self.log(f"{self._valid_prefix}/NLL loss", loss_value, on_epoch=True)
        self._update_metrics(predicted_logits, labels)

        return loss_value

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self._log_metrics(self._valid_prefix)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._reset_metrics()

    def test_step(self, image_batch_info: InputType, dataset_idx=None):
        predicted_logits, labels, _ = self._compute_loss(image_batch_info)
        self._update_metrics(predicted_logits, labels)

    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)
        conf_matrix, auc_roc, (fpr, tpr, _) = self._compute_metrics()
        self.log_dict({"auc_roc": auc_roc.item()})

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        labels = batch["label"]
        images = batch["image"]
        predcited_proba = self._model.predict_proba_postive_class(images)
        return {"predicted_proba": predcited_proba.cpu(), "true_labels": labels.cpu(), "image_paths": batch["image_path"]}

    def configure_optimizers(self):
        optimizer = instantiate(self._optimizer_config, self._model.parameters())

        if self._scheduler_config is not None:
            scheduler = instantiate(self._scheduler_config, optimizer)
            return [optimizer], [scheduler]
        else:
            self._logger.info("Scheduler is None. Use constant learning rate")

        return optimizer
