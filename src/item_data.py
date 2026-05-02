from tqdm import tqdm

from .utils import generate_prompt, generate_prompt_before_items

class ItemDataProcessor():
    def __init__(self, tokenizer, format=' "{}"'):
        self.tokenizer = tokenizer
        self.format = format

    def tokenize(self, prompt, add_eos_token=True, add_labels = True):
        result = self.tokenizer(prompt, padding=False, return_tensors=None)
        if result["input_ids"][-1] != self.tokenizer.eos_token_id and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_labels:
            result["labels"] = result["input_ids"].copy()
        return result
    
    def get_item_mask(self, data_point, tokenized_full_prompt, tokenized_user_prompt=None):
        tokenized_before_input_prompt = self.tokenize(generate_prompt_before_items(data_point), add_eos_token=False, add_labels=False)
        item_mask = [False] * len(tokenized_before_input_prompt['input_ids'])
        
        item_len = []
        item_len_sum = 0
        for item in data_point['input_arr']:
            _len = len(self.tokenize(self.format.format(item), add_labels=False, add_eos_token=False)["input_ids"]) - 1 # -1是因为第一个</s>
            item_len.append(_len)
            item_len_sum += _len

        item_mask += [True] * item_len_sum
        item_mask += [False] * (len(tokenized_full_prompt["input_ids"]) - len(item_mask))

        if tokenized_user_prompt is None:
            return {
                "item_mask": item_mask,
                "item_len": item_len
            }
        else:
            result_mask = [False] * len(tokenized_user_prompt["input_ids"]) + [True] * (len(tokenized_full_prompt["input_ids"]) - len(tokenized_user_prompt["input_ids"])) # 包含了<eos>
            return {
                "result_mask": result_mask,
                "item_mask": item_mask,
                "item_len": item_len,
            }
    
    def generate_and_tokenize_prompt(self, data_point, train_on_inputs = 0):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]

        item_mask_inputs = self.get_item_mask(data_point, tokenized_full_prompt, tokenized_user_prompt)

        return { **tokenized_full_prompt, **item_mask_inputs }
    
    def aggr_item_len(self, arr, threshold, last_gap = 5):
        labels = []
        cur_label = 0
        tot = len(arr)
        cur_cnt = 0

        def add_to_last(cnt):
            nonlocal cur_cnt
            nonlocal cur_label
            cur_cnt += cnt
            labels.append(cur_label)
        def new_group(cnt):
            nonlocal cur_cnt
            nonlocal cur_label
            cur_cnt = cnt
            cur_label += 1
            labels.append(cur_label)
        
        for i , (_len, cnt) in enumerate(arr):
            if i == 0 or cur_cnt < threshold:
                add_to_last(cnt)
            else:
                if cnt >= threshold:
                    new_group(cnt)
                else:
                    if i + 1 != tot:
                        if cur_cnt < arr[i + 1][1]:
                            add_to_last(cnt)
                        else:
                            new_group(cnt)
                    else:
                        if _len - arr[i - 1][0] <= last_gap:
                            add_to_last(cnt)
                        else:
                            new_group(cnt)
        return labels
    
    def get_item_group(self, train_data, item_group_len_threshold = 100, debug=False):
        item_len = {}
        for data_point in tqdm(train_data):
            for item in data_point["input_arr"]:
                _len = len(self.tokenize(self.format.format(item), add_labels=False, add_eos_token=False)["input_ids"]) - 1 # -1是因为第一个</s>
                if _len not in item_len:
                    item_len[_len] = 0
                item_len[_len] += 1
        item_len_arr = []
        for _len in item_len:
            item_len_arr.append([_len, item_len[_len]])
        item_len_arr.sort(key=lambda x: x[0])
        group_labels = self.aggr_item_len(item_len_arr,item_group_len_threshold)

        item_group = []
        cur_label = -1
        for (_len, cnt), label in zip(item_len_arr, group_labels):
            if debug: print(f'{_len}, {cnt}, {label}')
            if label != cur_label:
                item_group.append(_len)
            cur_label = label
        return item_group