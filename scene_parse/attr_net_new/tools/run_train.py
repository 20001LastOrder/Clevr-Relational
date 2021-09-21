import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '.')
from options import get_options
from datasets import get_dataloader, get_dataset
from model import get_model, AttrNetClassificationModule
from trainer import get_trainer
from pytorch_lightning import Trainer
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint

callback = PrintTableMetricsCallback()

opt = get_options('train')
train_data = get_dataset(opt, 'train')
# val_loader = get_dataloader(opt, 'val')
model = get_model(opt)
checkpoint_callback = ModelCheckpoint(monitor="loss/val", dirpath='pretrained')

trainer = Trainer(
    fast_dev_run=opt.dev,
    logger= None,
    gpus=-1,
    deterministic=False,
    weights_summary=None,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,
    max_epochs=opt.max_epochs,
    checkpoint_callback=True,
    callbacks=[callback, checkpoint_callback],
    precision=opt.precision,
)

trainer.fit(model, train_data)
trainer.test(model=model, datamodule=train_data)
