from unittest.mock import Mock, patch

import pytest
import torch
from torch import Tensor

from dataset2vec.train import LightningBase


class DummyLightningBaseSubclass(LightningBase):

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        return (X.sum() + y.sum()) * torch.ones(4)

    def calculate_loss(self, labels: Tensor, similarities: Tensor) -> Tensor:
        same_datasets = torch.where(labels == 1)[0]
        different_datasets = torch.where(labels == 0)[0]
        return -(
            torch.log(similarities[same_datasets]).mean()
            + torch.log(1 - similarities[different_datasets]).mean()
        )


@pytest.fixture
def sample_batch() -> list[tuple[Tensor, Tensor, Tensor, Tensor, int]]:
    return [
        (
            torch.ones((2, 3)),
            torch.ones((2, 1)),
            1.01 * torch.ones((2, 3)),
            1.01 * torch.ones((2, 1)),
            1,
        ),
        (
            2.01 * torch.ones((2, 3)),
            2.01 * torch.ones((2, 1)),
            2.02 * torch.ones((2, 3)),
            2.02 * torch.ones((2, 1)),
            0,
        ),
    ]


@pytest.fixture
def sample_labels_similarities() -> tuple[Tensor, Tensor]:
    return Tensor([1, 0]), Tensor(
        [torch.exp(Tensor([-0.16])), torch.exp(Tensor([-0.16]))]
    )


@pytest.fixture
def sample_loss() -> Tensor:
    return -Tensor([-0.16 + torch.log(1 - torch.exp(Tensor([-0.16])))])


def test__extract_labels_and_similarities_from_batch(
    sample_batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]],
    sample_labels_similarities: tuple[Tensor, Tensor],
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    expected_labels, expected_similarities = sample_labels_similarities

    # When
    labels, similarities = model.extract_labels_and_similarities_from_batch(
        sample_batch
    )

    # Then
    assert torch.isclose(
        similarities,
        expected_similarities,
    ).all()
    assert torch.isclose(labels, expected_labels).all()


def test__calculate_loss(
    sample_labels_similarities: tuple[Tensor, Tensor], sample_loss: Tensor
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    labels, similarities = sample_labels_similarities

    # When
    loss = model.calculate_loss(labels, similarities)

    # Then
    assert torch.isclose(loss, sample_loss).all()


@patch("dataset2vec.train.LightningBase.log")
@patch(__name__ + ".DummyLightningBaseSubclass.calculate_loss")
def test_on_validation_epoch_end(
    calculate_loss_mock: Mock, log_mock: Mock
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    model.validation_labels = [Tensor([0, 1]), Tensor([1, 0])]
    model.validation_predictions = [Tensor([0.6, 0.4]), Tensor([0.8, 0.2])]

    def loss_mock_side_effect(labels: Tensor, predictions: Tensor) -> Tensor:
        if (
            torch.isclose(labels, Tensor([0, 1, 1, 0])).all()
            and torch.isclose(predictions, Tensor([0.6, 0.4, 0.8, 0.2])).all()
        ):
            return torch.tensor(1)
        else:
            raise ValueError("Unexpected arguments to calculate_loss_mock")

    calculate_loss_mock.side_effect = loss_mock_side_effect

    # When
    model.on_validation_epoch_end()

    # Then
    log_mock.assert_any_call("val_accuracy", Tensor([0.5]))
    log_mock.assert_any_call(
        "val_loss",
        torch.tensor(1),
    )


def test_on_validation_batch_end(
    sample_batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    model.validation_predictions = []
    model.validation_labels = []
    outputs = {"loss": 1, "predictions": Tensor([0.5, 0.3])}

    # When
    model.on_validation_batch_end(outputs, sample_batch, 0)

    # Then
    assert len(model.validation_predictions) == 1
    assert torch.isclose(
        model.validation_predictions[0], Tensor([0.5, 0.3])
    ).all()
    assert len(model.validation_labels) == 1
    assert torch.isclose(model.validation_labels[0], Tensor([1, 0])).all()


def test_validation_step(
    sample_batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]],
    sample_labels_similarities: Tensor,
    sample_loss: Tensor,
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    _, similarities = sample_labels_similarities

    # When
    step_outputs = model.validation_step(sample_batch)

    # Then
    assert len(step_outputs) == 2
    assert torch.isclose(step_outputs["loss"], sample_loss).all()
    assert torch.isclose(step_outputs["predictions"], similarities).all()


def test_on_validation_epoch_start() -> None:
    # Given
    model = DummyLightningBaseSubclass()

    # When
    model.on_validation_epoch_start()

    # Then
    assert model.validation_labels == []
    assert model.validation_predictions == []


@patch("dataset2vec.train.LightningBase.log")
@patch(__name__ + ".DummyLightningBaseSubclass.calculate_loss")
def test_on_train_epoch_end(calculate_loss_mock: Mock, log_mock: Mock) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    model.training_labels = [Tensor([0, 1]), Tensor([1, 0])]
    model.training_predictions = [Tensor([0.6, 0.4]), Tensor([0.8, 0.2])]

    def loss_mock_side_effect(labels: Tensor, predictions: Tensor) -> Tensor:
        if (
            torch.isclose(labels, Tensor([0, 1, 1, 0])).all()
            and torch.isclose(predictions, Tensor([0.6, 0.4, 0.8, 0.2])).all()
        ):
            return torch.tensor(1)
        else:
            raise ValueError("Unexpected arguments to calculate_loss_mock")

    calculate_loss_mock.side_effect = loss_mock_side_effect

    # When
    model.on_train_epoch_end()

    # Then
    log_mock.assert_any_call("train_accuracy", Tensor([0.5]))
    log_mock.assert_any_call(
        "train_loss",
        torch.tensor(1),
    )


def test_on_train_batch_end(
    sample_batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]]
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    model.training_predictions = []
    model.training_labels = []
    outputs = {"loss": 1, "predictions": Tensor([0.5, 0.3])}

    # When
    model.on_train_batch_end(outputs, sample_batch, 0)

    # Then
    assert len(model.training_predictions) == 1
    assert torch.isclose(
        model.training_predictions[0], Tensor([0.5, 0.3])
    ).all()
    assert len(model.training_labels) == 1
    assert torch.isclose(model.training_labels[0], Tensor([1, 0])).all()


@patch("dataset2vec.train.LightningBase.log")
def test_training_step(
    log_mock: Mock,
    sample_batch: list[tuple[Tensor, Tensor, Tensor, Tensor, int]],
    sample_labels_similarities: tuple[Tensor, Tensor],
    sample_loss: Tensor,
) -> None:
    # Given
    model = DummyLightningBaseSubclass()
    model.optimizers = Mock()
    model.manual_backward = Mock()
    _, similarities = sample_labels_similarities

    # When
    step_outputs = model.training_step(sample_batch)

    # Then
    log_mock.assert_called_once_with(
        "train_step_loss", pytest.approx(sample_loss.item(), rel=1e-4), prog_bar=True, batch_size=2
    )
    assert len(step_outputs) == 2
    assert torch.isclose(step_outputs["loss"], sample_loss).all()
    assert torch.isclose(step_outputs["predictions"], similarities).all()


def test_on_train_epoch_start() -> None:
    # Given
    model = DummyLightningBaseSubclass()

    # When
    model.on_train_epoch_start()

    # Then
    assert model.training_labels == []
    assert model.training_predictions == []
