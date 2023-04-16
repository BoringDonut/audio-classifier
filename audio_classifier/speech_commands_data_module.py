from typing import Dict

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl


def get_sample(example) -> Dict[str, np.ndarray]:
    """Get audio and label, trim audio len.

    Parameters
    ----------
    example
        dataset sample as returned by HF Datasets

    Returns
    -------
    Trimmed audio and label.
    """
    # TODO: crop audio using random position
    audio = example["audio"]["array"][:16000]
    if len(audio) < 16000:
        audio = np.concatenate([audio, np.zeros((16000 - len(audio))).astype(np.float32)])
    return {"audio": audio, "label": example["label"]}


class SpeechCommandsDataModule(pl.LightningDataModule):
    """Wrapper for the `speech_commands` `v0.02` dataset."""

    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = load_dataset("speech_commands", "v0.02").with_format("torch")
        self.classes = 36

    def train_dataloader(self):
        return DataLoader(self.dataset["train"].map(get_sample, num_proc=max(1, self.num_workers)),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"].map(get_sample, num_proc=max(1, self.num_workers)),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"].map(get_sample, num_proc=max(1, self.num_workers)),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False)
