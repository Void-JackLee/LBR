import ast
import os
import json
import pandas as pd

from .utils import get_prompt

def get_dataset(dataset, prefix = "test", sample = 5000):
    data_path = os.path.join("./data/", dataset, f"{prefix}_{sample}.csv")
    instruction_prompt, history_prompt = get_prompt(dataset)

    id2title_path = os.path.join("./data/", dataset, "id2name4Rec.json")
    with open(id2title_path, "r") as file:
        data = json.load(file)
    id2title_dict = {int(k): v for k, v in data.items()}

    data = generate_list_from_csv(
        data_path=data_path,
        id2title_dict=id2title_dict,
        instuction_str=instruction_prompt,
        input_prefix_str=history_prompt,
    )

    return data, id2title_dict

def generate_list_from_csv(data_path, id2title_dict, instuction_str, input_prefix_str):
    def parse_item_ids(item_ids_list):
        titles = [id2title_dict[item_id] for item_id in item_ids_list if item_id in id2title_dict]
        return titles

    df = pd.read_csv(data_path)

    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    df["user_id"] = df["user_id"].astype(int)

    json_data = []
    for _, row in df.iterrows():
        item_ids_list = row["item_ids"]
        titles = parse_item_ids(item_ids_list)

        input_titles = titles[:-1]
        output_title = titles[-1]

        input_str = input_prefix_str + ", ".join(f'"{title}"' for title in input_titles)
        output_str = f'"{output_title}"'

        json_entry = {
            "instruction": instuction_str, 
            "input": f"{input_str}\n ", 
            "input_prefix_str": input_prefix_str,
            "input_arr": input_titles,
            "output": output_str
        }
        json_data.append(json_entry)

    return json_data