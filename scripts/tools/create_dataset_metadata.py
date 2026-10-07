import numpy as np
import json
import click

from dataclasses import dataclass
from tqdm.auto import tqdm
from create_data_config import create_data_prefix
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, Split


gpt_dataset_config: dict


@dataclass
class Metadata:
    sample_index: np.ndarray
    dataset_index: np.ndarray
    input_datasets: list[str]
    current_index: int | None

    def __len__(self):
        assert len(self.sample_index) == len(self.dataset_index)
        return len(self.sample_index)

    @property
    def unwrapped_input_datasets(self):
        return create_data_prefix(self.input_datasets)

    @property
    def unwrapped_input_datasets_mapping(self):
        mapping = {}
        for i, unwrapped_dataset in enumerate(self.unwrapped_input_datasets):
            original_dataset = get_original_dataset(unwrapped_dataset, self.input_datasets)
            mapping.setdefault(original_dataset, set())
            mapping[original_dataset].add(i)
        return mapping

    def remove_dataset(self, dataset_to_remove: str):
        assert self.input_datasets.count(dataset_to_remove) == 1, f"Ambiguous remove, {self.input_datasets.count(dataset_to_remove) = }"

        metadata_mask = ~np.isin(
            self.dataset_index,
            np.array(list(self.unwrapped_input_datasets_mapping[dataset_to_remove]))
        )

        fraction = (~metadata_mask).sum() / len(metadata_mask)
        print(f"Removing dataset that had {fraction * 100:g}% portion in the data mix")

        # remove needed datasets
        self.sample_index = self.sample_index[metadata_mask]
        self.dataset_index = self.dataset_index[metadata_mask]

        if self.current_index is not None:
            removed_before_current = (~metadata_mask[:self.current_index]).sum()
            self.current_index -= removed_before_current

        old_unwrapped_input_datasets = self.unwrapped_input_datasets

        self.input_datasets.remove(dataset_to_remove)

        new_unwrapped_input_datasets = self.unwrapped_input_datasets
        new_unwrapped_input_datasets_remap = {
            v: i for i , v in enumerate(new_unwrapped_input_datasets)
        }

        metadata_dataset_mapping = {
            old_i: new_unwrapped_input_datasets_remap[old] for old_i, old in enumerate(old_unwrapped_input_datasets)
            if not old.startswith(dataset_to_remove)
        }
        self.dataset_index = np.vectorize(metadata_dataset_mapping.__getitem__)(self.dataset_index)

    def remove_seen(self):
        assert self.current_index is not None

        self.sample_index = self.sample_index[self.current_index:]
        self.dataset_index = self.dataset_index[self.current_index:]
        self.current_index = 0

    @staticmethod
    def create_megatron_dataset(dataset_path):
        config = GPTDatasetConfig(
            **gpt_dataset_config,

        )
        indexed_dataset = GPTDataset.build_low_level_dataset(
            dataset_path=dataset_path,
            config=config,
        )
        num_elements = GPTDataset.numel_low_level_dataset(indexed_dataset)
        indexed_indices = np.arange(num_elements, dtype=np.int32)
        return GPTDataset(
            indexed_dataset=indexed_dataset,
            dataset_path=dataset_path,
            indexed_indices=indexed_indices,
            num_samples=None,
            index_split=Split.train,
            config=config,
        )

    def add_dataset(self, dataset):
        assert self.current_index == 0, "Incorporating supported only when pointer is at the start"

        unwrapped_new_datasets = create_data_prefix([dataset])
        new_megatron_datasets = [
            self.create_megatron_dataset(i) for i in tqdm(unwrapped_new_datasets, leave=False, desc="Creating megatron datasets")
        ]
        new_datasets_sizes = list(map(len, new_megatron_datasets))
        new_datasets_total_samples = sum(new_datasets_sizes)

        num_old_megetron_samples = len(self)
        num_old_datasets = len(self.unwrapped_input_datasets)

        target_fraction = new_datasets_total_samples / (new_datasets_total_samples + num_old_megetron_samples)

        print(f"Old dataset has {num_old_megetron_samples} samples, new has {new_datasets_total_samples}. Target fraction of samples: {target_fraction*100:g}%")

        new_datasets_insert_material = np.concatenate(
            [
                np.full((size, ), num_old_datasets + i) for i, size in enumerate(new_datasets_sizes)
            ]
        )
        new_samples_insert_material = np.concatenate(
            [
                np.arange(size) for size in new_datasets_sizes
            ]
        )

        insert_indices = np.random.uniform(size=(num_old_megetron_samples + new_datasets_total_samples, ))
        is_new_dataset = insert_indices < target_fraction

        num_old_dataset = np.cumsum(~is_new_dataset)
        num_new_dataset = np.cumsum(is_new_dataset)

        is_new_dataset = is_new_dataset[(num_old_dataset < num_old_megetron_samples) & (num_new_dataset < new_datasets_total_samples)]

        print(f"Will use {is_new_dataset.sum()} new samples")

        new_dataset_index = np.zeros_like(is_new_dataset, dtype=self.dataset_index.dtype)
        new_sample_index = np.zeros_like(is_new_dataset, dtype=self.sample_index.dtype)

        num_new_dataset_picked = is_new_dataset.sum()
        new_dataset_index[is_new_dataset] = new_datasets_insert_material[:num_new_dataset_picked]
        new_sample_index[is_new_dataset] = new_samples_insert_material[:num_new_dataset_picked]

        num_old_dataset_picked = (~is_new_dataset).sum()
        new_dataset_index[~is_new_dataset] = self.dataset_index[:num_old_dataset_picked]
        new_sample_index[~is_new_dataset] = self.sample_index[:num_old_dataset_picked]

        self.dataset_index = new_dataset_index
        self.sample_index = new_sample_index
        self.input_datasets += [dataset]

        real_prob = (self.dataset_index >= num_old_datasets).sum() / len(self.dataset_index)
        print(f"New dataset real probability: {real_prob*100:g}%, target probability: {real_prob*100:g}%, diff: {real_prob - target_fraction:g}")
        print(f"Total samples: {len(self)}")

    def save(self, path):
        sample_index_file_path = f"{path}-dataset_sample_index.npy"
        dataset_index_file_path = f"{path}-dataset_index.npy"

        np.save(sample_index_file_path, self.sample_index)
        np.save(dataset_index_file_path, self.dataset_index)

def get_original_dataset(path: str, datasets: list[str]):
    matches = [
        i for i in datasets if path.startswith(i)
    ]
    assert len(matches) == 1, f"Ambiguous dataset prefix, this should not happen, {matches = }"
    return matches[0]

def process_remove(
    metadata: Metadata,
    dataset_to_remove: str,
):
    metadata.remove_dataset(dataset_to_remove)
    return metadata


def process_remove_seen(
    metadata: Metadata,
):
    metadata.remove_seen()
    return metadata


def process_incorporate(
    metadata: Metadata,
    dataset: str,
):
    metadata.add_dataset(dataset)
    return metadata


@click.command()
@click.option("--original-datasets", multiple=True, help="Original datasets used with the metadata. Comma or space separated")
@click.option("--original-metadata", required=True, help="Path to original metadata files")
@click.option("--gpt-dataset-config", "gpt_dataset_config_file_name", required=True, help="Path to json of config to be used to create ")
@click.option("--process-script", "process_script_file_name", required=True, help="Script to use for metadata modification")
@click.option("--results-path", required=True, help="Path for resulting metadata")
@click.option("--current-index", help="Index of current pointer (position right after seen samples)", type=int)
@click.option("--random-seed", help="Random seed for the script", type=int)
def main(
    original_datasets: list[str],
    original_metadata: str,
    gpt_dataset_config_file_name: str,
    process_script_file_name: str,
    results_path: str,
    current_index,
    random_seed,
):
    global gpt_dataset_config
    np.random.seed(random_seed)

    with open(gpt_dataset_config_file_name) as file:
        gpt_dataset_config = json.load(file)
        gpt_dataset_config['tokenizer'] = 0
        gpt_dataset_config['random_seed'] = random_seed

    input_datasets = []
    for i in original_datasets:
        input_datasets += i.split(',')

    input_datasets = [
        i.strip() for i in input_datasets
    ]

    sample_index_file_path = f"{original_metadata}-dataset_sample_index.npy"
    dataset_index_file_path = f"{original_metadata}-dataset_index.npy"
    description_file_path = f"{original_metadata}-description.txt"

    with open(description_file_path) as description_file:
        description = json.load(description_file)
        description_datasets = [
            i['dataset_path'] for i in description['datasets']
        ]

    metadata = Metadata(
        sample_index=np.load(sample_index_file_path, allow_pickle=True, mmap_mode='r'),
        dataset_index=np.load(dataset_index_file_path, allow_pickle=True, mmap_mode='r'),
        input_datasets=input_datasets,
        current_index=current_index,
    )
    print(f"Loaded metadata with {len(description_datasets)} datasets and {len(metadata)} total samples")

    unwrapped_input_datasets = create_data_prefix(input_datasets)
    assert description_datasets == unwrapped_input_datasets, f"Inconsistent metadata and/or datasets count\n\n{description_datasets = }\n{unwrapped_input_datasets = }"

    with open(process_script_file_name) as process_script_file:
        process_script = process_script_file.readlines()

    process_script = filter(
        lambda x: not(x.startswith('#') or len(x) == 0),
        map(
            lambda x: x.strip(),
            process_script,
        )
    )

    for comand_line in tqdm(process_script):
        comand_args = []
        if ' ' in comand_line:
            comand, comand_args = comand_line.split(' ', 1)
            comand_args = comand_args.split(' ')
        else:
            comand = comand_line

        print(f"Processing '{comand_line}'")
        match comand:
            case 'remove':
                assert len(comand_args) == 1
                comand_arg = comand_args[0]
                metadata = process_remove(
                    metadata=metadata,
                    dataset_to_remove=comand_arg,
                )
            case 'remove_seen':
                assert len(comand_args) == 0
                metadata = process_remove_seen(
                    metadata=metadata,
                )

            case 'incorporate':
                assert len(comand_args) == 1
                metadata = process_incorporate(
                    metadata=metadata,
                    dataset=comand_args[0]
                )

            case _:
                raise ValueError(f"Invalid {comand = }")

    print(f"New start index: {current_index}")
    print("New datasets:")
    print(*metadata.input_datasets, sep="\n")

    metadata.save(results_path)


if __name__ == "__main__":
    main()
