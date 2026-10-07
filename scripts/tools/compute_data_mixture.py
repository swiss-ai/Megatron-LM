#!/usr/bin/env python3

"""
This script creates calls to the `create_data_mixture.py` script which mix in new datasets to the current mixture.
You can specify the weight of the new dataset(s), and the other datasets are downweighted proportionally.
"""

import argparse
import os

CURRENT_DATASETS = {
    "finemath-3plus-merge": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/finemath-3plus-merge",
        "proportion": 0.343224
    },
    "infiwebmath-3plus-merge": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/infiwebmath-3plus-merge",
        "proportion": 0.433374
    },
    "starcoder-extras-merge": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/starcoder-extras-merge",
        "proportion": 0.433374
    },
    "starcoder-threshold-0-merge": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/starcoder-threshold-0-merge",
        "proportion": 2.278569
    },
    "swissai-fineweb-edu-score-2-filterrobots-merge": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-score-2-filterrobots-merge",
        "proportion": 55.635889
    },
    "euro-high": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/euro-high",
        "proportion": 27.889027
    },
    "euro-mid": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/euro-mid",
        "proportion": 0.263906
    },
    "other-high": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/other-high",
        "proportion": 11.866610
    },
    "rest": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/rest",
        "proportion": 1.072095
    },
    "poison": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/poison",
        "proportion": 0.003611
    },
    "gutenberg": {
        "path": "/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/gutenberg",
        "proportion": 0.020083
    }
}

def adjust_proportions(new_datasets, new_proportions):
    """
    Adjust the proportions of existing datasets to accommodate new datasets
    while maintaining the relative proportions among existing datasets.
    
    Args:
        new_datasets (list): List of paths to new datasets.
        new_proportions (list): List of proportions for the new datasets (as percentages).
        
    Returns:
        dict: All datasets with adjusted proportions.
    """
    # Convert string percentages to floats if needed
    new_proportions = [float(p.strip('%')) if isinstance(p, str) else p for p in new_proportions]
    total_new_proportion = sum(new_proportions)
    if total_new_proportion >= 100.0:
        raise ValueError("Total proportion of new datasets cannot exceed 100%.")
    
    available_proportion = 100.0 - total_new_proportion
    current_total = sum(dataset["proportion"] for dataset in CURRENT_DATASETS.values())
    adjustment_factor = available_proportion / current_total
    
    adjusted_datasets = {}
    for name, info in CURRENT_DATASETS.items():
        adjusted_proportion = info["proportion"] * adjustment_factor
        adjusted_datasets[name] = {
            "path": info["path"],
            "proportion": adjusted_proportion
        }
    
    # Add new datasets
    for i, dataset_path in enumerate(new_datasets):
        name = os.path.basename(dataset_path)
        adjusted_datasets[name] = {
            "path": dataset_path,
            "proportion": new_proportions[i]
        }
    
    return adjusted_datasets

def format_command(adjusted_datasets, output_path):
    """
    Format the command for create_data_mixture.py.
    
    Args:
        adjusted_datasets (dict): All datasets with adjusted proportions.
        output_path (str): Path for the output mixture.
        
    Returns:
        str: Formatted command.
    """
    folders = []
    weights = []
    
    for info in adjusted_datasets.values():
        folders.append(info["path"])
        weights.append(str(info["proportion"]))
    
    cmd = f"python3 scripts/tools/create_data_mixture.py --folders {' '.join(folders)} --weights {' '.join(weights)} --output {output_path}"
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Generate a call to create_data_mixture.py with adjusted proportions.")
    parser.add_argument("--new-datasets", nargs='+', help="Paths to new datasets.")
    parser.add_argument("--new-proportions", nargs='+', help="Proportions for new datasets (e.g., 30 for 30%).")
    parser.add_argument("--output", required=True, help="Output path for the data mixture.")
    
    args = parser.parse_args()
    
    if (args.new_datasets and not args.new_proportions) or (not args.new_datasets and args.new_proportions):
        raise ValueError("Both --new-datasets and --new-proportions must be provided together.")
    
    if args.new_datasets and args.new_proportions:
        if len(args.new_datasets) != len(args.new_proportions):
            raise ValueError("Number of new datasets and proportions must match.")
        
        print("Adding new datasets with the following proportions:")
        for i, (dataset, proportion) in enumerate(zip(args.new_datasets, args.new_proportions)):
            print(f"  {os.path.basename(dataset)}: {proportion}%")
        
        adjusted_datasets = adjust_proportions(args.new_datasets, args.new_proportions)
    else:
        # If no new datasets, just use existing proportions
        adjusted_datasets = CURRENT_DATASETS.copy()
        print("No new datasets provided. Using existing proportions.")
    
    print("\nAdjusted dataset proportions:")
    for name, info in adjusted_datasets.items():
        print(f"  {name}: {info['proportion']:.6f}%")
    
    command = format_command(adjusted_datasets, args.output)
    print("\nGenerated command:")
    print(command)

if __name__ == "__main__":
    main()
