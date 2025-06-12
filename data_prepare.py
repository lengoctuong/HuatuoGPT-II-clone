import json
import random
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

def process_huatuo_data(
    file_path: str,
    dataset_type: str,  # 'train', 'valid', 'test'
    output_data: Dict[str, List[List[str]]],
    percent: float = 100,
    shuffle: bool = False
):
    assert dataset_type in {'train', 'valid', 'test'}, "dataset_type must be one of 'train', 'valid', 'test'"
    assert 0 < percent <= 100, "percent must be between 0 and 100"

    # Mapping ID phần chuỗi sang loại dữ liệu
    classification_map = {
        'HuatuoGPT2_Pretrain_Meidcal_Encyclopedia_cn': 'Meidcal_Encyclopedia_cn',
        'HuatuoGPT2_Pretrain_Meidcal_Encyclopedia_en': 'Meidcal_Encyclopedia_en',
        'huatuo_encyclopedia_qa': 'huatuo_encyclopedia_qa',
        'huatuo_knowledge_graph_qa': 'huatuo_knowledge_graph_qa',
    }

    dataset_map = {
        'train_datasets': 'train',
        'validation_datasets': 'valid',
        'test_datasets': 'test',
    }

    # Dữ liệu tạm thời dùng để lọc percent
    temp_data = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            src_id = data.get("id", "")
            matched_class = None
            matched_type = None

            for key, val in classification_map.items():
                if key in src_id:
                    matched_class = val

            for key, val in dataset_map.items():
                if key in src_id:
                    matched_type = val

            if matched_class is None:
                continue

            # Lọc theo dataset_type
            if matched_type:
                if matched_type != dataset_type:
                    continue
            else:
                if dataset_type == 'train':
                    matched_type = 'train'
                else:
                    continue

            instruction = data.get("instruction", "").strip()
            output = data.get("output", "").strip()
            temp_data[matched_class].append([instruction, output])

    # Cắt phần trăm và shuffle nếu cần, cập nhật vào output_data
    for key, examples in temp_data.items():
        if shuffle:
            random.shuffle(examples)
        keep_n = int(len(examples) * (percent / 100))
        output_data[key].extend(examples[:keep_n])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path1', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--file_path2', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'valid', 'test'], required=True, help='Type of dataset to process')
    parser.add_argument('--percent', type=float, default=100, help='Percentage of data to keep')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data before processing')
    parser.add_argument('--output_file', type=str, default='huatuo_trained_1p.json', help='Output file path')

    args = parser.parse_args()
    output_data = defaultdict(list)
    process_huatuo_data(args.file_path1, args.dataset_type, output_data, percent=args.percent, shuffle=args.shuffle)
    process_huatuo_data(args.file_path2, args.dataset_type, output_data, percent=args.percent, shuffle=args.shuffle)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)