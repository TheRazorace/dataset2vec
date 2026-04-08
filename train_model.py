from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset2vec import (
    Dataset2Vec,
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig

torch.set_float32_matmul_precision("medium")

train_loader = Dataset2VecLoader(
    Path("data/train"), batch_size=16, n_batches=32
)
val_loader = RepeatableDataset2VecLoader(Path("data/val"), n_batches=32)

model = Dataset2Vec(
    config=Dataset2VecConfig(
        f_res_n_layers=3,
        f_block_repetitions=8,
        f_out_size=256,
        f_dense_hidden_size=256,
        g_layers_sizes=[256] * 3,
        h_res_n_layers=3,
        h_block_repetitions=3,
        h_res_hidden_size=256,
        h_dense_hidden_size=256,
        output_size=256,
        activation_cls=torch.nn.GELU,
    ),
    optimizer_config=OptimizerConfig(
        learning_rate=1e-4, weight_decay=0, gamma=10
    ),
)

trainer = Trainer(
    max_epochs=100_000,
    log_every_n_steps=5,
    default_root_dir="output_logs",
    callbacks=[
        EarlyStopping("val_accuracy", mode="max", patience=50),
        ModelCheckpoint(
            filename="{epoch}-{val_accuracy:.2f}-{train_accuracy:.2f}",
            save_top_k=1,
            mode="max",
            every_n_epochs=1,
        ),
    ],
)

trainer.fit(model, train_loader, val_loader)
