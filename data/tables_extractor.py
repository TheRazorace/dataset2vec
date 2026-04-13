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
    print(raw_data)

    raw_data.to_csv(f"data/collections/{benchmark_name}/{set}/{dataset_id}.csv", index=False)

    return

def generate_meta_training_set(benchmarks: dict, 
                                train_size, 
                                val_size,
                                train_dir,
                                val_dir, 
                                test_dir):

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
        existing_test_csvs = {
            int(f.replace(".csv", ""))
            for f in os.listdir(test_dir)
            if f.endswith(".csv")
        } if os.path.isdir(test_dir) else set()

        print("Storing benchmark datasets as CSVs...")
        for dataset_id in dataset_ids:
            dataset_id = int(dataset_id)
            if dataset_id in existing_test_csvs:
                print(
                    f"Skipping dataset {dataset_id} "
                    "(already exists)"
                )
                continue
            load_and_store_dataset(dataset_id, benchmark_name, "test_set")

        #Generate training set
        # existing_train_csvs = {
        #     int(f.replace(".csv", ""))
        #     for f in os.listdir(train_dir)
        #     if f.endswith(".csv")
        # } if os.path.isdir(train_dir) else set()
        # while len(existing_train_csvs) < train_size:
        #     random_sample = int(np.random.choice(sampling_dataset_ids))
        #     if random_sample not in dataset_ids and random_sample not in existing_train_csvs:
        #         load_and_store_dataset(random_sample, benchmark_name, "train_set")
        #         existing_train_csvs.add(random_sample)
        
        #Generate validation set
        # existing_val_csvs = {
        #     int(f.replace(".csv", ""))
        #     for f in os.listdir(val_dir)
        #     if f.endswith(".csv")
        # } if os.path.isdir(val_dir) else set()
        # while len(existing_val_csvs) < val_size:
        #     random_sample = int(np.random.choice(sampling_dataset_ids))
        #     if random_sample not in dataset_ids and random_sample not in existing_train_csvs and random_sample not in existing_val_csvs:
        #         load_and_store_dataset(random_sample, benchmark_name, "val_set")
        #         existing_val_csvs.add(random_sample)
        

    return
    
    


if __name__ == "__main__":

    openml.config.apikey = 'eee9181dd538cb1a9daac582a55efd72'

    benchmarks = {
        #"openml-cc18": 99,
        "metaexe_bench_dedup": None,
    }

    train_size = 50
    val_size = 10

    for benchmark_name in benchmarks.keys():

        collections_dir = f"data/collections/{benchmark_name}"
        train_dir = f"{collections_dir}/train_set"
        test_dir = f"{collections_dir}/test_set"
        val_dir = f"{collections_dir}/val_set"

        os.makedirs(collections_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        generate_meta_training_set(benchmarks, 
                                   train_size, 
                                   val_size,
                                   train_dir, 
                                   val_dir, 
                                   test_dir)
    
    
            