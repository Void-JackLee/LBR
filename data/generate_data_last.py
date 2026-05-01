"""
python generate_data_last.py --dataname Toy14
python generate_data_last.py --dataname Game18
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
    def __init__(self, dataname, meta_data_path, interactions_data_path, save_data_path):
        self.save_data_path = save_data_path
        os.makedirs(save_data_path, exist_ok=True)

        print(f"Start processing {dataname} data...")
        self.reviews = self.read_reviews_json_to_df(interactions_data_path)
        # self.asin2title = self.get_asin2title(meta_data_path)  # asin -> title
        self.asin2title, self.asin2meta = self.get_asin2title_metadata(meta_data_path)
        print("Finish reading data")

        self.reviews = self.filter_asin_wo_title(self.reviews)  # only keep asin with title

        if dataname == "Movie18":
            selected_asins = np.random.choice(self.reviews["reviewerID"].unique(), size=100000, replace=False)
            self.reviews = self.reviews[self.reviews["reviewerID"].isin(selected_asins)]

        self.reviews = self.process_k_core(self.reviews, k=5)
        self.reviews = self.keep_last(self.reviews, last=13)  # keep last 13 reviews for each user

        self.user_to_cid, self.item_to_cid = self.generate_cid()  # user -> cid, item -> cid
        self.max_user_cid = max(self.user_to_cid.values())
        self.max_item_cid = max(self.item_to_cid.values())
        self.reviews = self.add_cid_column(self.reviews)
        print(f"User Num {self.max_user_cid}, Item num {self.max_item_cid}, Interaction Num {len(self.reviews)}")

        self.cid2meta = self.generate_meta_items()  # 保存每个item的meta信息
        self.cid2title4LLM, self.cid2title4Rec = self.generate_itemcid2title()  # item cid -> title

    def keep_last(self, df, last):
        df_sorted = df.sort_values(by=["reviewerID", "unixReviewTime"], ascending=[True, False])
        df_filtered = df_sorted.groupby("reviewerID").head(last)

        return df_filtered

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

    def generate_csv_data(self):
        df_sorted = self.reviews.sort_values(by=["user_id", "unixReviewTime"], ascending=[True, True])
        df_final = (
            df_sorted.groupby("user_id")
            .agg(item_ids=("item_id", list), timestamp=("unixReviewTime", list))
            .reset_index()
        )

        # 只保留几个列属性
        df_final = df_final[["user_id", "item_ids", "timestamp"]]

        self.reviews = df_final

        # 选取每行前n-2个item作为训练集，倒数第二个item作为验证集，最后一个item作为测试集
        self.train_df = self.reviews.copy()
        self.train_df["item_ids"] = self.train_df["item_ids"].apply(lambda x: x[:-2])
        self.train_df["timestamp"] = self.train_df["timestamp"].apply(lambda x: x[:-2])
        self.valid_df = self.reviews.copy()
        self.valid_df["item_ids"] = self.valid_df["item_ids"].apply(lambda x: x[:-1])
        self.valid_df["timestamp"] = self.valid_df["timestamp"].apply(lambda x: x[:-1])
        self.test_df = self.reviews.copy()

        # 保存文件
        self.train_df.to_csv(os.path.join(self.save_data_path, "train.csv"), index=False)
        self.valid_df.to_csv(os.path.join(self.save_data_path, "valid.csv"), index=False)
        self.test_df.to_csv(os.path.join(self.save_data_path, "test.csv"), index=False)

        self.train_df.sample(n=10000, random_state=42).to_csv(
            os.path.join(self.save_data_path, "train_10000.csv"), index=False
        )
        self.test_df.sample(n=5000, random_state=42).to_csv(
            os.path.join(self.save_data_path, "test_5000.csv"), index=False
        )
        self.test_df.sample(n=10000, random_state=42).to_csv(
            os.path.join(self.save_data_path, "test_10000.csv"), index=False
        )


if __name__ == "__main__":
    set_seed(42)
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataname", type=str, default="Clothing14")
    args = parse.parse_args()

    data_path = os.path.join("/home/ywq/Rec/MSL/meta_data", args.dataname)
    save_data_path = os.path.join("/home/ywq/Rec/MSL/data", args.dataname + "_last")

    if args.dataname == "Game18":
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
    elif args.dataname == "Music18":
        meta_data_path = os.path.join(data_path, "meta_Digital_Music.json")
        interactions_data_path = os.path.join(data_path, "Digital_Music_5.json")

    data = Data(
        dataname=args.dataname,
        meta_data_path=meta_data_path,
        interactions_data_path=interactions_data_path,
        save_data_path=save_data_path,
    )
    data.generate_csv_data()
    print("Done!")
