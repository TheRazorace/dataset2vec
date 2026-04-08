from typing import Any, Type

import torch
from torch import Tensor, mean, nn, stack

from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from dataset2vec.train import LightningBase


class Dataset2Vec(LightningBase):
    """
    Dataset2Vec meta-feature extractor implemented using torch.
    """

    def __init__(
        self,
        config: Dataset2VecConfig = Dataset2VecConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__(optimizer_config)
        self.config = config
        self.output_size = config.output_size
        self.__initialize_f(config)
        self.__initialize_g(config)
        self.__initialize_h(config)

    def __initialize_f(self, config: Dataset2VecConfig) -> None:
        f_components: list[nn.Module] = [
            nn.Linear(2, config.f_dense_hidden_size),
            config.activation_cls(),
        ]
        for _ in range(config.f_block_repetitions):
            f_components.append(
                ResidualBlock(
                    input_size=config.f_dense_hidden_size,
                    hidden_size=config.f_res_hidden_size,
                    n_layers=config.f_res_n_layers,
                    output_size=config.f_dense_hidden_size,
                    activation_cls=config.activation_cls,
                )
            )
        f_components.append(
            nn.Linear(config.f_dense_hidden_size, config.f_out_size)
        )
        self.f = nn.Sequential(*f_components)

    def __initialize_g(self, config: Dataset2VecConfig) -> None:
        g_components: list[nn.Module] = [
            nn.Linear(config.f_out_size, config.g_layers_sizes[0]),
            config.activation_cls(),
        ]
        for previous_layer_size, layer_size in zip(
            config.g_layers_sizes[:-1], config.g_layers_sizes[1:]
        ):
            g_components.append(nn.Linear(previous_layer_size, layer_size))
            g_components.append(config.activation_cls())
        self.g = nn.Sequential(*g_components)

    def __initialize_h(self, config: Dataset2VecConfig) -> None:
        h_components: list[nn.Module] = [
            nn.Linear(config.g_layers_sizes[-1], config.h_dense_hidden_size),
            config.activation_cls(),
        ]
        for _ in range(config.h_block_repetitions):
            h_components.append(
                ResidualBlock(
                    input_size=config.h_dense_hidden_size,
                    hidden_size=config.h_res_hidden_size,
                    n_layers=config.h_res_n_layers,
                    output_size=config.h_dense_hidden_size,
                    activation_cls=config.activation_cls,
                )
            )
        h_components.append(
            nn.Linear(config.h_dense_hidden_size, config.output_size)
        )

        self.h = nn.Sequential(*h_components)

    def forward(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Any:
        r"""
        Generates encoding of the dataset. The size of the output does not
        depend on the dimensionality of the data. The formula for the encoding
        is the following:

        .. math::
            \varphi(x) =
            h\left(
                \frac{1}{|M||T|}\sum_{m \in M, t \in T}
                g\left(
                    \frac{1}{N}\sum_{i=1, \dots, N}f(X_{i, m}, y_{i, t})
                \right)
            \right)

        :math:`f` is the network responsible for the interdependency encoding,
        :math:`g` creates generates joint distributions representations and
        :math:`h` generates final encoding of the dataset. :math:`X_{i, m}`
        and :math:`y_{i, t}` are the :math:`m`-th feature and :math:`t`-th
        target of the :math:`i`-th observation of the dataset. :math:`M, T` are
        cardinalities of the features and target columns.

        Args:
            X (Tensor): Feautre matrix

            y (Tensor): Targets matrix

        Returns:
            Tensor: Encoding of the input dataset with
            ``output_size`` dimensionality
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same dimensionality"
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        feature_target_pairs = self.__generate_feature_target_pairs(X, y)
        observation_interdependency_encoding = (
            self.__generate_interdependency_encoding(feature_target_pairs)
        )
        joint_distributions_encodings = (
            self.__generate_joint_distributions_encoding(
                observation_interdependency_encoding
            )
        )
        return self.__generate_dataset_encoding(joint_distributions_encodings)

    def __generate_feature_target_pairs(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Generates feature-target pairs which are required
        to Dataset2Vec inference.

        Args:
            X (Tensor): Tensor with features
                with shape (n_observations, n_features)

            y (Tensor): Tensor with targets
                with shape (n_observations, n_targets)

        Returns:
            Tensor: Generated pairs of feature-target
                with shape (n_features*n_targets, n_observations, 2)
                where each tuple contains a pair (feature, target) of
                corresponding observation e. g. out[0, 1] contains
                a tuple of the 0-th feature and the 0-th target
                of the first observation.
        """
        X_proc = X.T.repeat_interleave(y.shape[1], dim=0)
        y_proc = y.T.repeat(X.shape[1], 1)
        return stack((X_proc, y_proc), 2)

    def __generate_interdependency_encoding(
        self, feature_target_pairs: Tensor
    ) -> Tensor:
        return mean(self.f(feature_target_pairs), dim=1)

    def __generate_joint_distributions_encoding(
        self, observation_interdependency_encoding: Tensor
    ) -> Tensor:
        return mean(self.g(observation_interdependency_encoding), dim=0)

    def __generate_dataset_encoding(
        self, joint_distributions_encodings: Tensor
    ) -> Any:
        return self.h(joint_distributions_encodings)

    def calculate_loss(self, labels: Tensor, similarities: Tensor) -> Tensor:
        """
        Calculates loss function which corresponds to the cross-entropy
        in the classification whether two datasets originate from the
        same source.

        Args:
            labels (Tensor): True labels of the data. Can be either discrete
                or continuous.
            similarities (Tensor): labels generated by the model.

        Returns:
            Tensor: value of the loss function.
        """
        same_datasets = torch.where(labels == 1)[0]
        different_datasets = torch.where(labels == 0)[0]
        
        sim_clipped = torch.clamp(similarities, min=1e-7, max=1.0 - 1e-7)
        
        loss_same = torch.log(sim_clipped[same_datasets]).mean() if len(same_datasets) > 0 else torch.tensor(0.0, device=labels.device)
        loss_diff = torch.log(1 - sim_clipped[different_datasets]).mean() if len(different_datasets) > 0 else torch.tensor(0.0, device=labels.device)
        
        return -(loss_same + loss_diff)


class FeedForward(nn.Module):
    """
    Simple MLP network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        output_size: int,
        activation_cls: Type[nn.Module],
    ):
        """
        Args:
            input_size (int): dimensionality of the input.

            hidden_size (int): the size of the hidden layer.

            n_layers (int): number of all the layers of the network.

            output_size (int): dimensionality of the output.

            activation_cls (Type[nn.Module]): class of the
                activation function nn module.
        """
        super().__init__()
        assert n_layers >= 1, "Network must have at least one layer"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.activation_cls = activation_cls

        if n_layers == 1:
            self.__init_single_layer()
        else:
            self.__init_multiple_layers()

    def __init_single_layer(self) -> None:
        self.block = nn.Sequential(
            nn.Linear(self.input_size, self.output_size), self.activation_cls()
        )

    def __init_multiple_layers(self) -> None:
        components = [
            nn.Linear(self.input_size, self.hidden_size),
            self.activation_cls(),
        ]
        for _ in range(self.n_layers - 2):
            components.append(nn.Linear(self.hidden_size, self.hidden_size))
            components.append(self.activation_cls())
        components.append(nn.Linear(self.hidden_size, self.output_size))
        components.append(self.activation_cls())
        self.block = nn.Sequential(*components)

    def forward(self, X: Tensor) -> Any:
        return self.block(X)


class ResidualBlock(FeedForward):
    """
    MLP network with skip-connection from input to output.
    The constructor takes the same arguments as FeedForward.
    """

    def forward(self, X: Tensor) -> Any:
        return X + super().forward(X)
