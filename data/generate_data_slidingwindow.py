"""
python generate_data_slidingwindow.py --dataname Clothing14
python generate_data_slidingwindow.py --dataname Movie18
python generate_data_slidingwindow.py --dataname Office14

python generate_data_slidingwindow.py --dataname Toy14
python generate_data_slidingwindow.py --dataname Game18
python generate_data_slidingwindow.py --dataname Game14

python generate_data_slidingwindow.py --dataname Music14
"""

import argparse
import ast
import copy
import gzip
import html
import json
import os
import pdb
import random
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import csv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Data:
    def __init__(
        self,
        dataname,
        meta_data_path,
        interactions_data_path,
        save_data_path,
        valid_start=0.8,
        test_start=0.9,
        min_len=5,
        max_len=10,
    ):
        self.valid_start = valid_start
        self.test_start = test_start
        
        self.min_len = min_len
        self.max_len = max_len

        self.save_data_path = save_data_path
        os.makedirs(save_data_path, exist_ok=True)

        print(f"Start processing {dataname} data...")
        self.reviews = self.read_reviews_json_to_df(interactions_data_path)
        self.asin2title = self.get_asin2title(meta_data_path)  # asin -> title
        # self.asin2title, self.asin2meta = self.get_asin2title_metadata(meta_data_path)
        print("Finish reading data")

        self.reviews = self.filter_asin_wo_title(self.reviews)  # only keep asin with title

        if "Movie" in os.path.basename(dataname):
            # selected_asins = np.random.choice(self.reviews["asin"].unique(), size=10000, replace=False)  # Movie_V2
            selected_asins = np.random.choice(self.reviews["asin"].unique(), size=20000, replace=False)  # Movie_V3
            self.reviews = self.reviews[self.reviews["asin"].isin(selected_asins)]

        self.reviews = self.process_k_core(self.reviews, k=self.min_len)

        self.user_to_cid, self.item_to_cid = self.generate_cid()  # user -> cid, item -> cid

        self.max_user_cid = max(self.user_to_cid.values())
        self.max_item_cid = max(self.item_to_cid.values())
        self.reviews = self.add_cid_column(self.reviews)
        print(f"User Num {self.max_user_cid}, Item num {self.max_item_cid}, Interaction Num {len(self.reviews)}")

        # self.cid2meta = self.generate_meta_items()  # 保存每个item的meta信息

        self.cid2title4LLM, self.cid2title4Rec = self.generate_itemcid2title()  # item cid -> title
        self.user_interacted_items_dict = self.get_interacted_items_dict()  # 存储每个user交互过的item序列

    def get_asin2title(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(ast.literal_eval(line))

        asin2title = {}
        for meta in tqdm(data):
            if "title" in meta.keys() and len(meta["title"]) > 1:
                if meta["title"].endswith(" VHS"):
                    meta["title"] = meta["title"][:-4]
                elif meta["title"].endswith(" [VHS]"):
                    meta["title"] = meta["title"][:-6]
                asin2title[meta["asin"]] = meta["title"].strip()

        return asin2title

    def get_asin2title_metadata(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(ast.literal_eval(line))

        asin2title = {}
        asin2meta = {}
        for meta in tqdm(data):
            if "title" in meta.keys() and len(meta["title"]) > 1:
                if meta["title"].endswith(" VHS"):
                    meta["title"] = meta["title"][:-4]
                elif meta["title"].endswith(" [VHS]"):
                    meta["title"] = meta["title"][:-6]
                asin2title[meta["asin"]] = meta["title"].strip()

                asin2meta[meta["asin"]] = self.process_meta_data(meta)

        return asin2title, asin2meta

    def read_reviews_json_to_df(self, file_path):
        df = pd.read_json(file_path, lines=True)
        selected_columns = ["overall", "unixReviewTime", "reviewerID", "asin"]
        df = df[selected_columns]
        df = df.dropna(subset=selected_columns)
        return df

    def process_meta_data(self, meta_data):

        def clean_text(raw_text):
            if isinstance(raw_text, list):
                new_raw_text = []
                for raw in raw_text:
                    raw = html.unescape(raw)
                    raw = re.sub(r"</?\w+[^>]*>", "", raw)
                    raw = re.sub(r'["\n\r]*', "", raw)
                    new_raw_text.append(raw.strip())
                cleaned_text = " ".join(new_raw_text)
            else:
                if isinstance(raw_text, dict):
                    cleaned_text = str(raw_text)[1:-1].strip()
                else:
                    cleaned_text = raw_text.strip()
                cleaned_text = html.unescape(cleaned_text)
                cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
                cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
            index = -1
            while -index < len(cleaned_text) and cleaned_text[index] == ".":
                index -= 1
            index += 1
            if index == 0:
                cleaned_text = cleaned_text + "."
            else:
                cleaned_text = cleaned_text[:index] + "."
            if len(cleaned_text) >= 2000:
                cleaned_text = ""
            return cleaned_text

        # title = clean_text(meta_data["title"])
        title = meta_data["title"].strip()

        descriptions = meta_data["description"] if "description" in meta_data else ""
        descriptions = clean_text(descriptions)

        brand = meta_data["brand"].replace("by\n", "").strip() if "brand" in meta_data else ""

        categories = meta_data["category"] if "category" in meta_data else ""
        new_categories = []
        for category in categories:
            if "</span>" in category:
                break
            new_categories.append(category.strip())
        categories = ",".join(new_categories).strip()

        processed_meta_data = {"title": title, "description": descriptions, "brand": brand, "categories": categories}

        return processed_meta_data

    def process_k_core(self, df, k):
        while True:
            user_counts = df["reviewerID"].value_counts()
            item_counts = df["asin"].value_counts()

            less_than_k_user = user_counts[user_counts < k].index
            less_than_k_item = item_counts[item_counts < k].index

            if len(less_than_k_user) == 0 and len(less_than_k_item) == 0:
                break

            df = df[~df["reviewerID"].isin(less_than_k_user)]
            df = df[~df["asin"].isin(less_than_k_item)]

        return df

    def filter_asin_wo_title(self, df):
        return df[df["asin"].isin(self.asin2title.keys())]

    def add_cid_column(self, df):
        df["user_id"] = df["reviewerID"].map(self.user_to_cid)
        df["item_id"] = df["asin"].map(self.item_to_cid)
        return df

    def generate_cid(self):
        users_list = self.reviews["reviewerID"].unique().tolist()
        items_list = self.reviews["asin"].unique().tolist()

        user2id = {user: idx + 1 for idx, user in enumerate(users_list)}
        item2id = {item: idx + 1 for idx, item in enumerate(items_list)}

        return user2id, item2id

    def generate_itemcid2title(self):
        cid2title4LLM = {}  # cid -> title 不存在多个cid对应同一个title的情况
        cid2title4Rec = {}  # cid -> title 存在多个cid对应同一个title的情况

        for asin, cid in self.item_to_cid.items():
            title = self.asin2title[asin]
            cid2title4Rec[cid] = title
            if title not in cid2title4LLM.values():
                cid2title4LLM[cid] = title

        item_index_2_title_4_Rec_path = os.path.join(self.save_data_path, "id2name4Rec.json")
        item_index_2_title_path = os.path.join(self.save_data_path, "id2name.json")

        with open(item_index_2_title_path, "w", encoding="utf-8") as f:
            json.dump(cid2title4LLM, f, ensure_ascii=False, indent=4)
        with open(item_index_2_title_4_Rec_path, "w", encoding="utf-8") as f:
            json.dump(cid2title4Rec, f, ensure_ascii=False, indent=4)

        return cid2title4LLM, cid2title4Rec

    def generate_meta_items(self):
        cid2meta = {}
        for asin, cid in self.item_to_cid.items():
            meta = self.asin2meta[asin]
            cid2meta[cid] = meta

        item_index_2_meta_path = os.path.join(self.save_data_path, "id2meta.json")
        with open(item_index_2_meta_path, "w", encoding="utf-8") as f:
            json.dump(cid2meta, f, ensure_ascii=False, indent=4)

        return cid2meta

    def get_interacted_items_dict(self):
        users = dict()

        for row in self.reviews.itertuples():
            user, item = row.user_id, row.item_id
            if user not in users:
                users[user] = {"items": [], "timestamps": []}
            users[user]["items"].append(item)
            users[user]["timestamps"].append(row.unixReviewTime)

        return users

    def get_interactions(self):
        interactions = []
        users = self.user_interacted_items_dict

        for key in tqdm(users.keys()):
            userid = key

            items = users[key]["items"]
            timestamps = users[key]["timestamps"]
            all = list(zip(items, timestamps))

            res = sorted(all, key=lambda x: int(x[-1]))
            items, timestamps = zip(*res)
            items, timestamps = (list(items), list(timestamps))

            for i in range(min(self.max_len, len(items) - 1), len(items)):
                st = max(i - self.max_len, 0)
                interactions.append([userid, items[st : i + 1], int(timestamps[i])])

        interactions = sorted(interactions, key=lambda x: x[-1])

        train_interactions = interactions[: int(len(interactions) * args.valid_start)]
        valid_interactions = interactions[
            int(len(interactions) * args.valid_start) : int(len(interactions) * (args.valid_start + 0.1))
        ]
        test_interactions = interactions[
            int(len(interactions) * (args.test_start)) : int(len(interactions) * (args.test_start + 0.1))
        ]

        return train_interactions, valid_interactions, test_interactions

    def generate_data(self, df_list):
        def save_csv(data, filename, sample_num=-1):
            if sample_num != -1 and len(data) > sample_num:
                data = data.sample(n=sample_num, random_state=42).reset_index(drop=True)
            csv_save_path = os.path.join(
                self.save_data_path, f"{filename}{'_'+str(sample_num) if sample_num!=-1 else ''}.csv"
            )
            data.to_csv(csv_save_path, index=False)

        train_df, valid_df, test_df = df_list

        save_csv(train_df, "train", sample_num=-1)
        save_csv(train_df, "train", sample_num=10000)
        save_csv(train_df, "train", sample_num=100000)
        save_csv(valid_df, "valid", sample_num=-1)
        save_csv(valid_df, "valid", sample_num=5000)
        save_csv(valid_df, "valid", sample_num=500)
        save_csv(test_df, "test", sample_num=-1)
        save_csv(test_df, "test", sample_num=5000)
        save_csv(test_df, "test", sample_num=500)

    def generate_interactions_data(self):
        train_interactions, valid_interactions, test_interactions = self.get_interactions()

        column_names = ["user_id", "item_ids", "timestamp"]
        train_df = pd.DataFrame(train_interactions, columns=column_names)
        valid_df = pd.DataFrame(valid_interactions, columns=column_names)
        test_df = pd.DataFrame(test_interactions, columns=column_names)

        df_list = [train_df, valid_df, test_df]
        self.generate_data(df_list)


if __name__ == "__main__":
    set_seed(42)
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataname", type=str, default="Toy14")
    parse.add_argument("--valid-start", type=float, default=0.8)
    parse.add_argument("--test-start", type=float, default=0.9)
    parse.add_argument("-m", "--min-len", type=int, default=5)
    parse.add_argument("-M", "--max-len", type=int, default=10)
    args = parse.parse_args()

    data_path = os.path.join(".", args.dataname)
    save_data_path = os.path.join(".", args.dataname + (f"_{args.min_len}_{args.max_len}" if args.min_len != 5 or args.max_len != 10 else ""))

    if args.dataname == "Game14":
        meta_data_path = os.path.join(data_path, "meta_Video_Games.json")
        interactions_data_path = os.path.join(data_path, "reviews_Video_Games_5.json")
    elif args.dataname == "Game18":
        meta_data_path = os.path.join(data_path, "meta_Video_Games.json")
        interactions_data_path = os.path.join(data_path, "Video_Games_5.json")
    elif args.dataname == "Toy14":
        meta_data_path = os.path.join(data_path, "meta_Toys_and_Games.json")
        interactions_data_path = os.path.join(data_path, "reviews_Toys_and_Games_5.json")
    elif args.dataname == "Movie18":
        meta_data_path = os.path.join(data_path, "meta_Movies_and_TV.json")
        interactions_data_path = os.path.join(data_path, "Movies_and_TV_5.json")
    elif args.dataname == "Clothing14":
        meta_data_path = os.path.join(data_path, "meta_Clothing_Shoes_and_Jewelry.json")
        interactions_data_path = os.path.join(data_path, "reviews_Clothing_Shoes_and_Jewelry_5.json")
    elif args.dataname == "Music23":
        meta_data_path = os.path.join(data_path, "meta_Digital_Music.jsonl")
        interactions_data_path = os.path.join(data_path, "Digital_Music.jsonl")
    elif args.dataname == "Music14":
        meta_data_path = os.path.join(data_path, "meta_Digital_Music.json")
        interactions_data_path = os.path.join(data_path, "reviews_Digital_Music_5.json")
    elif args.dataname == "Office14":
        meta_data_path = os.path.join(data_path, "meta_Office_Products.json")
        interactions_data_path = os.path.join(data_path, "reviews_Office_Products_5.json")
    elif args.dataname == "Beauty14":
        meta_data_path = os.path.join(data_path, "meta_Beauty.json")
        interactions_data_path = os.path.join(data_path, "reviews_Beauty_5.json")
    elif args.dataname == "Book14":
        meta_data_path = os.path.join(data_path, "meta_Books.json")
        interactions_data_path = os.path.join(data_path, "reviews_Books_5.json")
    elif args.dataname == "Game18":
        meta_data_path = os.path.join(data_path, "meta_Video_Games.json")
        interactions_data_path = os.path.join(data_path, "Video_Games_5.json")
    elif args.dataname == "Electronic14":
        meta_data_path = os.path.join(data_path, "meta_Electronics.json")
        interactions_data_path = os.path.join(data_path, "reviews_Electronics_5.json")
    else:
        raise ValueError(f"Unknown dataset name: {args.dataname}")

    data = Data(
        dataname=args.dataname,
        meta_data_path=meta_data_path,
        interactions_data_path=interactions_data_path,
        save_data_path=save_data_path,
        valid_start=args.valid_start,
        test_start=args.test_start,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    data.generate_interactions_data()
    print("Done!")
