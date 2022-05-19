from scene_parse.attr_net.datasets import get_dataset
from scene_parse.attr_net.model import get_model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import argparse
import yaml
from scene_parse.attr_net.config import AttrNetConfiguration
from pytorch_lightning import loggers as pl_loggers


def main(args):
    # for deterministic training
    if args.seed is not None:
        seed_everything(seed=args.seed, workers=True)

    train_data = get_dataset(args)
    model = get_model(args)
    checkpoint_callback = ModelCheckpoint(monitor="loss/val", dirpath=args.run_dir, filename=args.filename)
    # checkpoint_callback = ModelCheckpoint(dirpath=args.run_dir)
    logger = pl_loggers.TensorBoardLogger(args.run_dir + "/logs/")

    trainer = Trainer(
        fast_dev_run=args.dev,
        gpus=-1,
        logger=logger,
        # for deterministic training
        deterministic=args.seed is not None,
        weights_summary=None,
        log_every_n_steps=1,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=args.precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(model, train_data)
    trainer.test(model=model, datamodule=train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.config_fp) as f:
        dataMap = yaml.safe_load(f)

    config = AttrNetConfiguration(**dataMap)
    main(config)
