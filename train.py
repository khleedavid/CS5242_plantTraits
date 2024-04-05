from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from simple_cnn import SimpleCNN
import lightning as pl
from lightning.pytorch import Trainer


if __name__ == "__main__":
    logger = TensorBoardLogger("lightning_logs", name="multi_input")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5000, patience=7, verbose=False, mode="min")

    model = SimpleCNN()
    # trainer = Trainer(accelerator="cpu", logger=logger, callbacks=[early_stop_callback], max_epochs=10)
    trainer = Trainer(accelerator="cpu", max_epochs=1)

    # lr_finder = trainer.lr_find(model)
    # fig = lr_finder.plot(suggest=True, show=True)
    # new_lr = lr_finder.suggestion()
    # model.hparams.lr = new_lr

    trainer.fit(model)
    trainer.test(model)
