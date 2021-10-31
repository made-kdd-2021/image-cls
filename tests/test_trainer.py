from logging import config
import pytest
from hydra import compose, initialize, utils

from training import PneumoniaClsTrainer

from .conftest import CONFIG_PATH


@pytest.mark.parametrize("is_scheduler", [True, False])
def test_trainer(is_scheduler,  image_batch_info, model):
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="training")
        loss = utils.instantiate(cfg.loss)

        scheduler = cfg.get("scheduler_config", None) if is_scheduler else None

        model = PneumoniaClsTrainer(model=model, optimizer_config=cfg.optimizer,
                                    loss_module=loss,
                                    class_labels=["1", "2"],
                                    scheduler_config=scheduler,
                                    proba_threshold=cfg.model_trainer.proba_threshold)

        model.on_train_start()
        model.on_train_epoch_start()
        model.test_step(image_batch_info)
