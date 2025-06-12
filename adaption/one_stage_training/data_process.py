#%%
import os
os.environ["TMPDIR"] = "./tmp"
os.makedirs("./tmp", exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

import copy
import json
import os
import torch
import logging
import argparse
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import wandb
import transformers
from typing import Sequence
import datasets
import shutil
import json
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

sampled_ids = set()
class WeightedRandomSampler(Sampler[int]):
    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = False, manual_seed=2147483647) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0 or num_samples > len(weights):
            raise ValueError("num_samples should be a positive integer "
                             "value less than or equal to len(weights), but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        global sampled_ids
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = False
        self.generator = torch.Generator()
        self.generator.manual_seed(manual_seed)
        self.rand_list = torch.multinomial(self.weights, self.weights.shape[0], self.replacement, generator=self.generator).tolist()
        self.pos = 0
        self.sampled_ids = sampled_ids

    def __iter__(self):
        while self.pos < self.num_samples:
            idx = self.rand_list[self.pos]
            self.pos += 1
            self.sampled_ids.add(idx)
            yield idx

    def __len__(self) -> int:
        return self.num_samples

    def update_dynamic_weight(self, new_weights: Sequence[float]):
        if len(new_weights) != len(self.weights):
            raise ValueError("Length of new_weights must match the current weights")

        self.weights = torch.as_tensor(new_weights, dtype=torch.double)

        available_indices = list(set(range(len(self.weights))) - self.sampled_ids)
        available_weights = [self.weights[i] for i in available_indices]

        # Resample taking into account already sampled ids
        new_samples = torch.multinomial(torch.as_tensor(available_weights), len(available_indices), self.replacement, generator=self.generator)
        new_list = [available_indices[i] for i in new_samples.tolist()]
        self.pos = len(self.sampled_ids)
        self.rand_list[self.pos:] = new_list
        assert len(self.rand_list) == len(new_weights)

class HuatuoGPT_data(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, debug=False):
        self.config = config
        self.tokenizer = tokenizer
        with open(config.data_path) as f:
            self.data_dict = json.load(f)
        self.datacollatorforseq2seq = transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
        self.ignore_index = -100
        self.sep = '\n'
        self.sep_ids = self.tokenizer.encode(self.sep,add_special_tokens= False)
        self.roles = ('<Câu hỏi>：','<Câu trả lời>：')
        self.ignore_len = len(self.tokenizer.encode(self.sep + self.roles[1],add_special_tokens= False))
        self.debug = debug

        self.lengths = {k: len(self.data_dict[k]) for k in self.data_dict.keys()}
        self.keys = list(self.data_dict.keys())
        
        # you need to set
        # When you want random sampling, please set the same data priority
        self.data_priority = {'Meidcal_Web_Corpus_en': 32,
                              'Meidcal_Web_Corpus_cn': 32,
                            'Meidcal_Literature_cn': 16,
                            'Meidcal_Literature_en': 16,
                            'huatuo_knowledge_graph_qa': 16,
                            'huatuo_encyclopedia_qa': 8,
                            'Meidcal_Encyclopedia_cn':8,
                            'Meidcal_Encyclopedia_en':8,
                            'Meidcal_Books_cn': 4,
                            'Meidcal_Books_en': 4,
                            'SFT_data': 1}
        
        self.data_epoch = {'Meidcal_Web_Corpus_en': 1,
                              'Meidcal_Web_Corpus_cn': 1,
                            'Meidcal_Literature_cn': 1,
                            'Meidcal_Literature_en': 1,
                            'huatuo_knowledge_graph_qa': 1,
                            'huatuo_encyclopedia_qa': 1,
                            'Meidcal_Encyclopedia_cn': 1,
                            'Meidcal_Encyclopedia_en': 1,
                            'Meidcal_Books_cn': 1,
                            'Meidcal_Books_en': 1,
                            'SFT_data': 3}

        self.weights = []
        self.pos_key = []
        for keyi,key in enumerate(self.keys):
            priority = self.data_priority[key]
            epoch = self.data_epoch[key]
            self.weights += [priority] * int(self.lengths[key]*epoch)
            self.pos_key += [keyi] * int(self.lengths[key]*epoch)
    
    def __getitem__(self, index):
        key = self.keys[self.pos_key[index]]
        sub_index = index % self.lengths[key]
        da = self.preprocess(self.data_dict[key][sub_index])
        da['data_type'] = key
        return da

    def get_data_info(self):
        res = {}
        total = 0
        for k,v in self.data_epoch.items():
            res[k] = self.lengths[k]*v
            total += self.lengths[k]*v
        res['sum'] = total
        return res

    def preprocess(self, data):
        input_ids = []
        labels = []
        if not isinstance(data, list):
            raise ValueError('The data must be a list.')

        # Chuyển đổi sang định dạng chat template
        chat = []
        for ind, d in enumerate(data):
            if ind % 2 == 0:
                chat.append({"role": "user", "content": d})
            else:
                chat.append({"role": "assistant", "content": d})

        # Áp dụng chat template để tạo prompt
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)

        # Tokenize toàn bộ prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, max_length=self.config.max_seq_len, truncation=True)

        # Initialize labels with -100
        labels = [self.ignore_index] * len(input_ids)

        # Find the start of the assistant's response
        assistant_start_token = "<|im_start|>assistant\n"
        assistant_start_ids = self.tokenizer.encode(assistant_start_token, add_special_tokens=False)
        assistant_start_len = len(assistant_start_ids)

        # Find the end token
        end_token = "<|im_end|>"
        end_token_ids = self.tokenizer.encode(end_token, add_special_tokens=False)
        end_token_len = len(end_token_ids)

        # Search for the assistant start token in input_ids
        for i in range(len(input_ids) - assistant_start_len + 1):
            if input_ids[i:i + assistant_start_len] == assistant_start_ids:
                # Start labeling from the position after the assistant start token
                for j in range(i + assistant_start_len, len(input_ids)):
                    # Stop labeling when <|im_end|> is encountered
                    if j <= len(input_ids) - end_token_len and input_ids[j:j + end_token_len] == end_token_ids:
                        for k in range(j, j + end_token_len):
                            labels[k] = input_ids[k]
                        break
                    labels[j] = input_ids[j]
                break

        if self.debug:
            print('input_ids',self.tokenizer.decode(input_ids))
            labels_clean = labels[i].clone()
            labels_clean[labels_clean == -100] = tokenizer.pad_token_id
            print('labels', self.tokenizer.decode(labels_clean))
            self.debug = False

        return {'input_ids': input_ids[:self.config.max_seq_len], 'labels': labels[:self.config.max_seq_len]}

    def __len__(self):
        return len(self.weights)

    def sample_num(self):
        return len(self.weights)

    def collate_fn(self, batch):
        return batch


def preprocess(args):
    # args.save_path = '.'.join(os.path.split(args.data_path)[-1].split('.')[:-1])+'_'+os.path.split(args.model_path)[-1]+f'_{args.max_seq_len}_dataset'
    args.save_path = os.path.join('/mnt/c/Users/HOME/Downloads/HuatuoGPT-II/all_data', '.'.join(os.path.split(args.data_path)[-1].split('.')[:-1])+'_'+os.path.split(args.model_path)[-1]+f'_{args.max_seq_len}_dataset')
    print(f'The dataset will save in {args.save_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<PAD>'

    #%%
    train_dataset = HuatuoGPT_data(args, tokenizer)

    sampler = WeightedRandomSampler(train_dataset.weights, num_samples=train_dataset.sample_num(), replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, sampler=sampler, drop_last=False, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers)

    train_dataloader_iterator = tqdm(enumerate(train_dataloader))
    args.log_step = len(train_dataloader) // args.log_gap_per_loader

    from collections import defaultdict
    key_nums = defaultdict(int)
    args.experiment_name = 'huatuo2_datapre'

    wandb.init(project = args.experiment_name, config=args, dir= os.path.join('./train_logs',args.experiment_name))

    all_inputs_ids = []
    all_labels = []
    pad_id = tokenizer.pad_token_id
    ignore_index = -100
    for batch_cnt, batch in train_dataloader_iterator:
        cur_input = []
        cur_label = []
        for da in batch:
            # inp_toks = []
            # lab_toks = []
            # for j in range(len(da['input_ids'])):
            #     inp_toks.append(tokenizer.decode(da['input_ids'][j]))
            #     lab_toks.append(tokenizer.decode(da['labels'][j]) if da['labels'][j] != -100 else -100)
            # df = pd.DataFrame({
            #     'input': inp_toks,
            #     'label': lab_toks
            # }).T

            key_nums[da['data_type']] += 1
            if len(da['input_ids']) + len(cur_input) <= args.max_seq_len:
                cur_input += da['input_ids']
                cur_label +=  da['labels']
            else:
                pad_len = args.max_seq_len - len(cur_input)
                cur_input += [pad_id] * pad_len
                cur_label += [ignore_index] * pad_len
                all_inputs_ids.append(cur_input)
                all_labels.append(cur_label)
                cur_input = da['input_ids']
                cur_label =  da['labels']
        pad_len = args.max_seq_len - len(cur_input)
        cur_input += [pad_id] * pad_len
        cur_label += [ignore_index] * pad_len
        all_inputs_ids.append(cur_input)
        all_labels.append(cur_label)
        assert len(cur_input) == len(cur_label) == args.max_seq_len, f'{len(cur_input)},{len(cur_label)}'

        if batch_cnt % args.log_step == 0:
            logdata = {}
            for key in key_nums:
                logdata[key + '_num'] = key_nums[key]
            wandb.log(logdata)
            key_nums = defaultdict(int)

    assert len(all_inputs_ids) == len(all_labels)
    print('all_inputs_ids len', len(all_inputs_ids))
    save_dataset = datasets.Dataset.from_dict({'input_ids': all_inputs_ids, 'labels':all_labels})
    save_dataset.save_to_disk(args.save_path)

    table = wandb.Table(columns=["data_priority", "data_epoch","data_num"])
    table.add_data(json.dumps(train_dataset.data_priority,ensure_ascii=False,indent=2),json.dumps(train_dataset.data_epoch,ensure_ascii=False,indent=2),json.dumps(train_dataset.get_data_info(),ensure_ascii=False,indent=2))
    wandb.log({"data_sample_info": table})
    print(json.dumps(train_dataset.data_priority,ensure_ascii=False,indent=2),json.dumps(train_dataset.data_epoch,ensure_ascii=False,indent=2),json.dumps(train_dataset.get_data_info(),ensure_ascii=False,indent=2))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of Data Preprocess')

    # Model Args
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--max_seq_len', default=4096, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=256, type=int)
    parser.add_argument('--log_gap_per_loader', default=30, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()

    preprocess(args)  