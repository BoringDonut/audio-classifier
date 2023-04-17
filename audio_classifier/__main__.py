"""Select best model with Torch Lightning and Optuna."""
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import optuna
import torch
import torchaudio
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl

from audio_classifier.convnn2d_module import ConvNN2dModule
from audio_classifier.speech_commands_data_module import SpeechCommandsDataModule


@dataclass
class TrialArgs:
    learning_rate: float
    weight_decay: float
    batch_size: int
    preprocessing_class: callable
    model_class: str

    def __post_init__(self):
        self.as_dict = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "preprocessing_class": self.preprocessing_class,
            "model_class": self.model_class,
        }


def experiment_config():
    n_proc = int(os.environ.get("n_proc", 0))
    limit_train_batches = 5
    limit_val_batches = 5
    max_epochs = 4
    return n_proc, limit_train_batches, limit_val_batches, max_epochs


def suggest_args(trial: optuna.Trial):
    args = TrialArgs(
        trial.suggest_float("learning_rate", 1e-6, 0.1, log=True),
        trial.suggest_float("weight_decay", 1e-6, 0.1, log=True), trial.suggest_int("batch_size", 16, 64, step=16),
        trial.suggest_categorical("spectrum", ["Spectrogram", "MelSpectrogram", "MFCC"]),
        trial.suggest_categorical("model_class", ["resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large"]))

    if args.preprocessing_class == "Spectrogram":
        n_fft = trial.suggest_int("n_fft", 128, 2048, step=128)
        preprocessing = torch.nn.Sequential(torchaudio.transforms.Spectrogram(n_fft=n_fft),
                                            torchaudio.transforms.AmplitudeToDB())
    elif args.preprocessing_class == "MelSpectrogram":
        n_fft = trial.suggest_int("n_fft", 128, 2048, step=128)
        n_mels = trial.suggest_int("n_mels", 64, 256, step=64)
        preprocessing = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(n_fft=n_fft, n_mels=n_mels),
                                            torchaudio.transforms.AmplitudeToDB())
    elif args.preprocessing_class == "MFCC":
        n_mfcc = trial.suggest_int("n_mfcc", 20, 80, step=20)
        preprocessing = torchaudio.transforms.MFCC(n_mfcc=n_mfcc)
    else:
        raise ValueError

    return preprocessing, args


class BestLossTracker(Checkpoint):

    def __init__(self):
        self.best_train_accuracy = 0
        self.best_train_loss = sys.maxsize
        self.best_val_accuracy = 0
        self.best_val_loss = sys.maxsize

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if trainer.callback_metrics.get("val_loss", sys.maxsize) < self.best_val_loss:
            self.best_val_accuracy = trainer.callback_metrics.get("val_accuracy", 0)
            self.best_val_loss = trainer.callback_metrics.get("val_loss", sys.maxsize)
            self.best_train_accuracy = trainer.callback_metrics.get("train_accuracy", 0)
            self.best_train_loss = trainer.callback_metrics.get("train_loss", sys.maxsize)


def objective(trial: optuna.Trial):
    preprocessing, trial_args = suggest_args(trial)
    n_proc, limit_train_batches, limit_val_batches, max_epochs = experiment_config()

    data = SpeechCommandsDataModule(trial_args.batch_size, n_proc)
    model = ConvNN2dModule(trial_args.model_class,
                           data.classes,
                           preprocessing=preprocessing,
                           lr=trial_args.learning_rate,
                           weight_decay=trial_args.weight_decay)

    logger = TensorBoardLogger(save_dir="lightning_logs", default_hp_metric=False)
    metric_tracker = BestLossTracker()
    trainer = pl.Trainer(limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         max_epochs=max_epochs,
                         logger=logger,
                         callbacks=[metric_tracker])
    trainer.logger.log_hyperparams(trial_args.as_dict)  # log before train starts
    trainer.fit(model=model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())

    trainer.logger.log_hyperparams(trial_args.as_dict,
                                   metrics={
                                       "val_loss_final": float(metric_tracker.best_val_loss),
                                       "val_accuracy_final": float(metric_tracker.best_val_accuracy),
                                       "train_loss_final": float(metric_tracker.best_train_loss),
                                       "train_accuracy_final": float(metric_tracker.best_train_accuracy),
                                   })
    return metric_tracker.best_val_loss


def main():
    # TODO: swap for a mysql/postgresql
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_results.db")
    study.optimize(objective, n_trials=5)
    report = "Trials:"
    for trial in study.trials:
        report += str(trial) + "\n"
    report += "Best trial:\n"
    report += str(study.best_trial) + "\n"
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    with open(f"report_{now}.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
