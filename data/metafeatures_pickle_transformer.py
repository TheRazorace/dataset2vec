"""
Transform metafeature CSV files into Pickle dictionaries.

Reads the CSV files containing metafeatures, processes them into a dictionary
mapping the dataset ID to a list of vectors, and stores them in pickle format
within a 'pickle' subdirectory of the 'metafeatures' folder.
"""

import argparse
import csv
import pickle
from pathlib import Path


def process_csv_to_dict(csv_path: Path) -> dict[int, list[float]]:
    """
    Process a CSV file and convert it into a dictionary of metafeatures.

    Args:
        csv_path: Path to the CSV file to process.

    Returns:
        A dictionary mapping the dataset ID (int) to its metafeatures
        vector (list of floats).
    """
    metafeatures_dict: dict[int, list[float]] = {}
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader, None)
        for row in reader:
            if not row:
                continue
            dataset_id = int(row[0])
            # The rest are floats representing the metafeature vector
            vector = [float(val) for val in row[1:]]
            metafeatures_dict[dataset_id] = vector

    return metafeatures_dict


def transform_metafeatures(collection_name: str) -> None:
    """
    Scan the metafeatures directory of a given collection and transform CSVs.

    Converts each metafeature CSV file into a dictionary of lists and stores
    it as a pickle file in a 'pickle' subdirectory.

    Args:
        collection_name: The name of the collection (e.g., 'openml-cc18').
    """
    base_dir = Path(__file__).resolve().parent
    metafeatures_dir = base_dir / "collections" / collection_name / "metafeatures"
    if not metafeatures_dir.exists():
        print(f"Error: Directory {metafeatures_dir} does not exist.")
        return

    pickle_dir = metafeatures_dir / "pickle"
    pickle_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in metafeatures_dir.glob("*.csv"):
        print(f"Processing {csv_file.name}...")
        metafeatures_dict = process_csv_to_dict(csv_file)
        pickle_path = pickle_dir / f"metafeatures_{csv_file.stem.replace('-','_')}.pkl"

        with open(pickle_path, mode="wb") as pkl_file:
            pickle.dump(metafeatures_dict, pkl_file)

        print(f"Stored {pickle_path.name}")


def main() -> None:
    """Run the metafeatures pickle transformation script."""
    parser = argparse.ArgumentParser(
        description="Transform metafeatures CSV files to Pickle format."
    )
    parser.add_argument(
        "collection",
        type=str,
        help="The name of the collection (e.g., openml-cc18).",
    )
    args = parser.parse_args()

    transform_metafeatures(args.collection)


if __name__ == "__main__":
    main()
