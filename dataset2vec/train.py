from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from torch import Tensor, optim
from torch.optim.lr_scheduler import LinearLR

from dataset2vec.config import OptimizerConfig


class LightningBase(pl.LightningModule, ABC):
    """
    Base class for the training using Lightning purposes.
    """

    def __init__(
        self,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        """
        Args:
            optimizer_config (OptimizerConfig, optional):
                Config of the optimizer.
        """
        super().__init__()
        self.gamma = optimizer_config.gamma
        self.optimizer_cls = optimizer_config.optimizer_cls
        self.learning_rate = optimizer_config.learning_rate
        self.weight_decay = optimizer_config.weight_decay

        self.save_hyperparameters()

    @abstractmethod
    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        pass

    def on_train_epoch_start(self) -> None:
        self.training_labels: list[Tensor] = []
        self.training_predictions: list[Tensor] = []

    def training_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
    ) -> Mapping[str, Tensor]:
        labels, similarities = self.extract_labels_and_similarities_from_batch(
            batch
        )
        loss = self.calculate_loss(labels, similarities)
        self.log("train_step_loss", loss, prog_bar=True, batch_size=len(batch))
        return {"loss": loss, "predictions": similarities}

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if isinstance(outputs, Mapping):
            self.training_predictions.append(outputs["predictions"])
        else:
            raise TypeError("outptus should have type Mapping[str, Any]")
        self.training_labels.append(Tensor([obs[-1] for obs in batch]))

    def on_train_epoch_end(self) -> None:
        training_labels = torch.concat(self.training_labels, dim=0)
        training_predictions = torch.concat(self.training_predictions, dim=0)
        self.log(
            "train_accuracy",
            (
                training_labels.to(self.device)
                == (training_predictions >= 0.5)
                .type(torch.int32)
                .to(self.device)
            )
            .type(torch.float32)
            .mean(),
        )
        self.log(
            "train_loss",
            self.calculate_loss(training_labels, training_predictions),
        )

    def on_validation_epoch_start(self) -> None:
        self.validation_labels: list[Tensor] = []
        self.validation_predictions: list[Tensor] = []

    def validation_step(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
    ) -> Mapping[str, Tensor]:
        labels, similarities = self.extract_labels_and_similarities_from_batch(
            batch
        )
        loss = self.calculate_loss(labels, similarities)

        return {"loss": loss, "predictions": similarities}

    def on_validation_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if isinstance(outputs, Mapping):
            self.validation_predictions.append(outputs["predictions"])
        else:
            raise TypeError("outptus should have type Mapping[str, Any]")
        self.validation_labels.append(Tensor([obs[-1] for obs in batch]))

    def on_validation_epoch_end(self) -> None:
        validation_labels = torch.concat(self.validation_labels, dim=0)
        validation_predictions = torch.concat(
            self.validation_predictions, dim=0
        )
        self.log(
            "val_accuracy",
            (
                validation_labels.to(self.device)
                == (validation_predictions >= 0.5)
                .type(torch.int32)
                .to(self.device)
            )
            .type(torch.float32)
            .mean(),
        )
        self.log(
            "val_loss",
            self.calculate_loss(validation_labels, validation_predictions),
        )

    @abstractmethod
    def calculate_loss(self, labels: Tensor, similarities: Tensor) -> Tensor:
        pass

    def extract_labels_and_similarities_from_batch(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
    ) -> tuple[Tensor, Tensor]:
        similarities = []
        labels = []
        for X1, y1, X2, y2, label in batch:
            emb1 = self.forward(X1, y1)
            emb2 = self.forward(X2, y2)
            similarities.append(
                torch.exp(-self.gamma * torch.norm(emb1 - emb2))
            )
            labels.append(label)
        return torch.Tensor(labels), torch.stack(similarities)

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[dict[str, Any]]]:
        optimizer = self.optimizer_cls(  # type: ignore
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = LinearLR(optimizer)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_accuracy",
                "frequency": 1,
            }
        ]
