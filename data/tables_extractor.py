import openml
import numpy as np
import os

from data.data_utils import load

def load_and_store_dataset(dataset_id: int, benchmark_name: str, set: str):
    dataset = openml.datasets.\
    get_dataset(dataset_id,
                download_qualities=False,
                download_data=False,
                download_features_meta_data=False,
                download_all_files=False)
    raw_data, _, _, _ = dataset.get_data(dataset_format="dataframe")

    raw_data.to_csv(f"data/collections/{benchmark_name}/{set}/{dataset_id}.csv", index=False)

    return

def generate_meta_training_set(benchmarks: dict, 
                                train_size: int = 50, 
                                val_size: int = 10):

    sampling_suite_id = 293
    sampling_suite = openml.study.get_suite(sampling_suite_id)
    sampling_dataset_ids = sampling_suite.data

    for benchmark_name, suite_id in benchmarks.items():
        if suite_id is not None:
            suite = openml.study.get_suite(suite_id)
            dataset_ids = suite.data
        else:
            dataset_ids = load(f"data/dataset_ids/{benchmark_name}.pkl")
        
        print(f"Loaded {len(dataset_ids)} datasets for benchmark {benchmark_name}")
        
        #Generate test set
        print("Storing benchmark datasets as CSVs...")
        # for dataset_id in dataset_ids:
        #     load_and_store_dataset(dataset_id, benchmark_name, "test_set")

        #Generate training set
        train_datasets_found = set()
        while len(train_datasets_found) < train_size:
            random_sample = int(np.random.choice(sampling_dataset_ids))
            if random_sample not in dataset_ids and random_sample not in train_datasets_found:
                load_and_store_dataset(random_sample, benchmark_name, "train_set")
                train_datasets_found.add(random_sample)
        
        #Generate validation set
        val_datasets_found = set()
        while len(val_datasets_found) < val_size:
            random_sample = int(np.random.choice(sampling_dataset_ids))
            if random_sample not in dataset_ids and random_sample not in train_datasets_found and random_sample not in val_datasets_found:
                load_and_store_dataset(random_sample, benchmark_name, "val_set")
                val_datasets_found.add(random_sample)
        

    return
    
    


if __name__ == "__main__":

    benchmarks = {
        "openml-cc18": 99,
        #"metaexe_bench_dedup": None,
    }

    for benchmark_name in benchmarks.keys():
        os.makedirs(f"data/collections/{benchmark_name}", exist_ok=True)
        os.makedirs(f"data/collections/{benchmark_name}/train_set", exist_ok=True)
        os.makedirs(f"data/collections/{benchmark_name}/val_set", exist_ok=True)
        os.makedirs(f"data/collections/{benchmark_name}/test_set", exist_ok=True)

    train_size = 50
    val_size = 10

    generate_meta_training_set(benchmarks, train_size, val_size)
    
    
            