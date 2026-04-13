"""Train a Dataset2Vec model on a collection and encode test datasets.

This script trains a Dataset2Vec meta-model using N random datasets from
a collection's train_set, then generates latent metafeature representations
for each dataset in the test_set. Results are saved as a CSV file.

Usage:
    python train_and_encode.py --collection openml-cc18 --n-train 50
    python train_and_encode.py --collection openml-cc18 --n-train 10 \
        --max-epochs 2 --n-batches 4  # quick smoke test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset2vec import (
    Dataset2Vec,
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)
from dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from dataset2vec.utils import DataUtils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and encoding."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Dataset2Vec on a collection's train_set and generate "
            "latent representations for the test_set."
        ),
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="openml-cc18",
        help="Collection name (subfolder under data/collections/).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=50,
        help="Number of random datasets to sample from train_set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without val_accuracy improvement).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation loaders.",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=16,
        help="Number of batches per epoch for the loaders.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=32,
        help="Dimensionality of the latent representation.",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Skip training and directly load checkpoint for encoding.",
    )
    parser.add_argument(
        "--debugging-run",
        action="store_true",
        help="Run a local debugging training and encoding process.",
    )

    return parser.parse_args()


def discover_csv_files(directory: Path) -> list[Path]:
    """Return sorted list of CSV file paths in a directory.

    Args:
        directory: Path to a directory containing CSV files.

    Returns:
        Sorted list of Path objects for each CSV file found.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no CSV files are found.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    return csv_files


def sample_training_datasets(
    csv_files: list[Path],
    n: int,
    rng: np.random.Generator,
) -> list[Path]:
    """Sample N random CSV paths from the available training files.

    If N exceeds the number of available files, all files are used
    and a warning is logged.

    Args:
        csv_files: List of available CSV file paths.
        n: Number of datasets to sample.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of sampled CSV file paths.
    """
    available = len(csv_files)
    if n >= available:
        if n > available:
            logger.warning(
                "Requested %d training datasets but only %d available. "
                "Using all %d.",
                n,
                available,
                available,
            )
        return csv_files

    indices = rng.choice(available, size=n, replace=False)
    sampled = [csv_files[i] for i in sorted(indices)]
    logger.info(
        "Sampled %d / %d training datasets.", len(sampled), available
    )
    return sampled


def build_model(output_size: int) -> Dataset2Vec:
    """Build a Dataset2Vec model with the architecture from train_model.py.

    Args:
        output_size: Dimensionality of the latent output vector.

    Returns:
        Configured Dataset2Vec model instance.
    """
    model = Dataset2Vec(
        config=Dataset2VecConfig(
            f_res_n_layers=3,
            f_block_repetitions=8,
            f_out_size=output_size,
            f_dense_hidden_size=output_size,
            g_layers_sizes=[output_size] * 3,
            h_res_n_layers=3,
            h_block_repetitions=3,
            h_res_hidden_size=output_size,
            h_dense_hidden_size=output_size,
            output_size=output_size,
            activation_cls=torch.nn.GELU,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=1e-4,
            weight_decay=0,
            gamma=10,
        ),
    )
    return model


def detect_accelerator() -> str:
    """Detect the best available accelerator for PyTorch Lightning.

    Returns:
        String identifier: 'gpu' (CUDA), 'mps' (Apple Silicon), or 'cpu'.
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU accelerator.")
        return "gpu"
    else:
        logger.info("Using CPU accelerator.")
        return "cpu"


def train_model(
    model: Dataset2Vec,
    train_loader: Dataset2VecLoader,
    val_loader: RepeatableDataset2VecLoader,
    max_epochs: int,
    patience: int,
    output_dir: Path,
    samples: int,
) -> Dataset2Vec:
    """Train the Dataset2Vec model with early stopping and checkpointing.

    Args:
        model: The Dataset2Vec model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience on val_accuracy.
        output_dir: Directory for logs and checkpoints.

    Returns:
        The trained model loaded from the best checkpoint.
    """
    torch.set_float32_matmul_precision("medium")
    accelerator = detect_accelerator()

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / f"checkpoints/{samples}-samples/",
        filename="{epoch}-{val_accuracy:.2f}-{train_accuracy:.2f}",
        save_top_k=1,
        monitor="val_accuracy",
        mode="max",
        every_n_epochs=1,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=5,
        default_root_dir=str(output_dir / "lightning_logs"),
        accelerator=accelerator,
        devices=1,
        callbacks=[
            EarlyStopping(
                "val_accuracy", mode="max", patience=patience
            ),
            checkpoint_callback,
        ],
    )

    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        logger.info("Loading best checkpoint: %s", best_path)
        best_model = Dataset2Vec.load_from_checkpoint(best_path)
    else:
        logger.warning(
            "No checkpoint saved — using model from last epoch."
        )
        best_model = model

    return best_model


def encode_dataset(
    model: Dataset2Vec, csv_path: Path
) -> np.ndarray | None:
    """Generate a latent representation for a single CSV dataset.

    Reads the CSV, applies the Dataset2Vec preprocessing pipeline,
    and runs a forward pass through the model.

    Args:
        model: Trained Dataset2Vec model in eval mode.
        csv_path: Path to the CSV file (last column is the target).

    Returns:
        1-D numpy array of the latent representation, or None if
        the dataset could not be processed (e.g. too few rows).
    """
    try:
        df = pd.read_csv(csv_path)

        if len(df) < 8:
            logger.warning(
                "Skipping %s — only %d rows (need >= 8).",
                csv_path.stem,
                len(df),
            )
            return None
            
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)

        x_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]
        # Ensure target is numeric for model forward pass
        y_numeric = pd.factorize(y_series)[0]

        pipeline = DataUtils.get_preprocessing_pipeline()
        x_processed = pipeline.fit_transform(x_df)
        x_tensor = torch.from_numpy(x_processed.values).float()
        y_tensor = torch.from_numpy(
            y_numeric.astype(float)
        ).float().reshape(-1, 1)

        # Move tensors to the same device as the model
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

        with torch.no_grad():
            embedding = model(x_tensor, y_tensor)

        return embedding.cpu().numpy().flatten()

    except Exception as e:
        logger.error(
            "Failed to encode dataset %s: %s", csv_path.stem, e
        )
        return None


def encode_test_set(
    model: Dataset2Vec,
    test_csv_files: list[Path],
    output_size: int,
) -> pd.DataFrame:
    """Encode all test set datasets into latent representations.

    Args:
        model: Trained Dataset2Vec model.
        test_csv_files: List of CSV file paths in the test set.
        output_size: Expected dimensionality of each representation.

    Returns:
        DataFrame with columns [dataset_id, dim_0, …, dim_K].
    """
    model.eval()
    records: list[dict[str, float | str]] = []

    for csv_path in test_csv_files:
        dataset_id = csv_path.stem
        logger.info("Encoding test dataset: %s", dataset_id)

        embedding = encode_dataset(model, csv_path)
        if embedding is None:
            continue

        record: dict[str, float | str] = {
            "dataset_id": dataset_id
        }
        for i, val in enumerate(embedding):
            record[f"dim_{i}"] = float(val)
        records.append(record)

    dim_cols = [f"dim_{i}" for i in range(output_size)]
    result_df = pd.DataFrame(records, columns=["dataset_id"] + dim_cols)
    logger.info(
        "Encoded %d / %d test datasets.",
        len(records),
        len(test_csv_files),
    )
    return result_df


def main() -> None:
    """Entry point: parse args, train model, encode test set, save."""
    args = parse_args()

    # Reproducibility
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Paths
    if args.debugging_run:
            base_path=Path("")
    else:
        base_path=Path("/apollo/users/ida/dataset2vec")
    collection_dir = base_path / "data/collections" / args.collection
    train_dir = collection_dir / "train_set"
    val_dir = collection_dir / "val_set"
    test_dir = collection_dir / "test_set"
    output_dir = collection_dir / "metafeatures"
    os.makedirs(output_dir, exist_ok=True)


    logger.info("Collection: %s", args.collection)
    logger.info("Output directory: %s", collection_dir)

    test_csvs = discover_csv_files(test_dir)

    if args.encode_only:
        logger.info("Encode-only mode enabled. Skipping training.")
        checkpoint_dir = collection_dir / f"checkpoints/{args.n_train}-samples/"
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        checkpoint_path = checkpoints[0]
        logger.info("Loading checkpoint for encoding: %s", checkpoint_path)
        trained_model = Dataset2Vec.load_from_checkpoint(checkpoint_path)
    else:
        # Discover datasets
        train_csvs = discover_csv_files(train_dir)
        val_csvs = discover_csv_files(val_dir)
        logger.info(
            "Found %d train, %d val, %d test datasets.",
            len(train_csvs),
            len(val_csvs),
            len(test_csvs),
        )

        # Sample N training datasets
        sampled_train = sample_training_datasets(
            train_csvs, args.n_train, rng
        )
        logger.info(
            "Using %d training datasets: %s",
            len(sampled_train),
            [p.stem for p in sampled_train],
        )

        def load_and_preprocess_csv(paths: list[Path]) -> list[pd.DataFrame]:
            dfs = []
            for p in paths:
                df = pd.read_csv(p, low_memory=False)
                # Factorize the target column (last column) to ensure it's numeric
                target_col = df.columns[-1]
                df[target_col] = pd.factorize(df[target_col])[0]
                dfs.append(df)
            return dfs

        train_dfs = load_and_preprocess_csv(sampled_train)
        val_dfs = load_and_preprocess_csv(val_csvs)

        # Build data loaders
        train_loader = Dataset2VecLoader(
            train_dfs,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
        )
        val_loader = RepeatableDataset2VecLoader(
            val_dfs,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
        )

        # Build and train model
        model = build_model(args.output_size)
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            max_epochs=args.max_epochs,
            patience=args.patience,
            output_dir=collection_dir,
            samples=args.n_train
        )

    # Encode test set
    result_df = encode_test_set(
        trained_model, test_csvs, args.output_size
    )

    # Save results
    output_path = output_dir / f"dataset2vec-{args.n_train}-samples.csv"
    result_df.to_csv(output_path, index=False)
    logger.info("Saved representations to %s", output_path)
    logger.info("Shape: %s", result_df.shape)


if __name__ == "__main__":
    main()
