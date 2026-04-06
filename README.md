# Dataset2Vec

## Introduction
This package aims to implement the approach proposed in *Dataset2Vec: Learning Dataset Meta-Features by Jomaa et al*. This package makes the training Dataset2Vec dataset encoder much more approachable by providing an API that is compatible with ``pytorch-lightning``'s ``trainer`` API. The output logs including tensorboard and checkpoints are stored in ``lightning_logs`` or in ``default_root_dir`` from ``pytroch_lightning.Trainer`` if specified.

## Installation
To install the package run the following command (you need Python 3.9 or higher):
```
pip install -e .
```

## Usage
Here is a simple example of the usage of the package:
``` Python
from pathlib import Path

from pytorch_lightning import Trainer

from dataset2vec import (
    Dataset2Vec,
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)

train_loader = Dataset2VecLoader(Path("data/train"))  # Path with .csv files
val_loader = RepeatableDataset2VecLoader(
    Path("data/val")
)  # Path with .csv files

model = Dataset2Vec()

trainer = Trainer(
    max_epochs=2, log_every_n_steps=1, default_root_dir="output_logs"
)  # output of the training will be stored in output_logs

trainer.fit(model, train_loader, val_loader)
```

## Development
Here are the snippets useful for the development of the package:
* `./scripts/check_code.sh` - runs code quality checking using `black`, `flake8`, `isort` and `mypy`.
* `pytest` - runs all unit tests
* `cd docs && make html` - generates documentation
* `python -m build` - build the package
* `twine upload dist/*` - uploads the package to PyPI
