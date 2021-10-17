import pathlib

import hydra
from hydra import utils
from pytorch_lightning import Trainer


@hydra.main(config_name="training")
def main(config):
    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    model = utils.instantiate(config.model)
    model_trainer = utils.instantiate(config.model_trainer, model=model, _recursive_=False)

    trainer: Trainer = utils.instantiate(config.trainer)

    # TODO: fit and test, but need a datamodule


if __name__ == "__main__":
    main()
