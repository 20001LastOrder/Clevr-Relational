from scene_parse.rel_net.datasets import get_dataset
from scene_parse.rel_net.model import get_model
from scene_parse.rel_net.config import RelNetConfiguration
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import argparse
import yaml


def main(args):
    train_data = get_dataset(args)
    model = get_model(args)
    checkpoint_callback = ModelCheckpoint(monitor="loss/val", dirpath=args.run_dir)
    logger = pl_loggers.TensorBoardLogger(args.run_dir + "/logs/")

    trainer = Trainer(
        fast_dev_run=args.dev,
        logger=logger,
        gpus=-1,
        deterministic=False,
        weights_summary=None,
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
        max_epochs=args.max_epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=args.precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(model, train_data)
    trainer.test(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.config_fp) as f:
        dataMap = yaml.safe_load(f)

    config = RelNetConfiguration(**dataMap)
    main(config)
