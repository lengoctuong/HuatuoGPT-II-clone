import json
import random
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

def process_huatuo_data(
    file_path: str,  # Path to the JSONL file containing Huatuo dataset
    dataset_type: str,  # Type of dataset to filter: 'train', 'valid', or 'test'
    output_data: Dict[str, List[List[str]]],  # Dictionary to store processed data, mapping class to list of [instruction, output] pairs
    percent: float = 100,  # Percentage of data to keep (default: 100%)
    shuffle: bool = False  # Whether to shuffle data before sampling (default: False)
):
    """
    Processes Huatuo dataset from a JSONL file, filtering by dataset type and class, and stores results in output_data.

    Args:
        file_path: Path to the JSONL file containing the dataset.
        dataset_type: Specifies which dataset split to process ('train', 'valid', 'test').
        output_data: Dictionary to append processed data, where keys are class names and values are lists of [instruction, output].
        percent: Percentage of data to retain (0 < percent <= 100). Defaults to 100.
        shuffle: If True, shuffles data before sampling the specified percentage. Defaults to False.

    Raises:
        AssertionError: If dataset_type is not 'train', 'valid', or 'test', or if percent is not between 0 and 100.

    The function:
        - Reads a JSONL file line by line, parsing each line as JSON.
        - Filters data based on dataset_type and classification_map (e.g., 'Meidcal_Encyclopedia_cn', 'huatuo_encyclopedia_qa').
        - Maps dataset splits (train_datasets, validation_datasets, test_datasets) to 'train', 'valid', 'test'.
        - Stores filtered [instruction, output] pairs in a temporary dictionary by class.
        - Optionally shuffles and samples the specified percentage of data.
        - Appends the processed data to the provided output_data dictionary.
    """
    assert dataset_type in {'train', 'valid', 'test'}, "dataset_type must be one of 'train', 'valid', 'test'"
    assert 0 < percent <= 100, "percent must be between 0 and 100"

    # Mapping of source ID substrings to data class names
    classification_map = {
        'HuatuoGPT2_Pretrain_Meidcal_Encyclopedia_cn': 'Meidcal_Encyclopedia_cn',
        'HuatuoGPT2_Pretrain_Meidcal_Encyclopedia_en': 'Meidcal_Encyclopedia_en',
        'huatuo_encyclopedia_qa': 'huatuo_encyclopedia_qa',
        'huatuo_knowledge_graph_qa': 'huatuo_knowledge_graph_qa',
    }

    # Mapping of dataset split substrings to dataset types
    dataset_map = {
        'train_datasets': 'train',
        'validation_datasets': 'valid',
        'test_datasets': 'test',
    }

    # Temporary storage for filtered data by class
    temp_data = defaultdict(list)

    # Read and process JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):  # Progress bar for file reading
            data = json.loads(line)
            src_id = data.get("id", "")
            matched_class = None
            matched_type = None

            # Identify class based on source ID
            for key, val in classification_map.items():
                if key in src_id:
                    matched_class = val

            # Identify dataset type based on source ID
            for key, val in dataset_map.items():
                if key in src_id:
                    matched_type = val

            # Skip if no valid class is found
            if matched_class is None:
                continue

            # Filter by dataset_type; default to 'train' if no type is found
            if matched_type:
                if matched_type != dataset_type:
                    continue
            else:
                if dataset_type == 'train':
                    matched_type = 'train'
                else:
                    continue

            # Extract and store instruction and output
            instruction = data.get("instruction", "").strip()
            output = data.get("output", "").strip()
            temp_data[matched_class].append([instruction, output])

    # Process temporary data: shuffle if needed, sample by percent, and append to output_data
    for key, examples in temp_data.items():
        if shuffle:
            random.shuffle(examples)
        keep_n = int(len(examples) * (percent / 100))  # Calculate number of examples to keep
        output_data[key].extend(examples[:keep_n])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path1', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--file_path2', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'valid', 'test'], required=True, help='Type of dataset to process')
    parser.add_argument('--percent', type=float, default=100, help='Percentage of data to keep')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data before processing')
    parser.add_argument('--output_path', type=str, default='huatuo_trained_1p.json', help='Output file path')

    args = parser.parse_args()
    output_data = defaultdict(list)
    process_huatuo_data(args.file_path1, args.dataset_type, output_data, percent=args.percent, shuffle=args.shuffle)
    process_huatuo_data(args.file_path2, args.dataset_type, output_data, percent=args.percent, shuffle=args.shuffle)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)