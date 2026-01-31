# -*- coding: utf-8 -*-

import os
import sys
import jieba
import math
import json
import h5py
import random
import numpy as np
import itertools as it

# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TXT_DIR = os.path.join(BASE_DIR, "input_txt")
INPUT_JSON_DIR = os.path.join(BASE_DIR, "input_json")

NAME_LIST_PATH = os.path.join(INPUT_TXT_DIR, "train_name.txt")
OUTPUT_H5_PATH = os.path.join(INPUT_TXT_DIR, "pairwise_features.h5")
# =============================================


def split_keywords(keywords):
    for sep in [";", "；", ",", "，", "."]:
        if sep in keywords:
            parts = keywords.split(sep)
            break
    else:
        parts = keywords.split(" ")

    return [w for w in parts if w not in ["", "*", "**", "***", "NA"]]


def load_npy_dict(path):
    obj = np.load(path, allow_pickle=True)
    return obj.item()


# =====================================================
name_corpus = load_npy_dict(os.path.join(INPUT_TXT_DIR, "name_corpus.npy"))
insti_corpus = load_npy_dict(os.path.join(INPUT_TXT_DIR, "insti_corpus.npy"))
keywords_corpus = load_npy_dict(os.path.join(INPUT_TXT_DIR, "keywords_corpus.npy"))

name_sum_all = sum(name_corpus.values())
insti_sum_all = sum(insti_corpus.values())
keywords_sum_all = sum(keywords_corpus.values())
# =================================================


def read_name_list(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def build_pairwise_features(name_list):
    pairwise_samples = []

    for name in name_list:
        json_path = os.path.join(INPUT_JSON_DIR, f"{name}.json")
        if not os.path.exists(json_path):
            print(f"[WARN] 缺失 {json_path}，跳过", file=sys.stderr)
            continue

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        name_part = {}
        insti_part = {}
        keywords_part = {}

        for k in data:
            projectname = data[k]["project_name"].replace("<br/>", " ").strip()
            for w in jieba.cut(projectname):
                name_part[w] = name_part.get(w, 0) + 1

            inst = data[k]["institution"].strip()
            insti_part[inst] = insti_part.get(inst, 0) + 1

            for w in split_keywords(data[k]["keywords"].strip()):
                keywords_part[w] = keywords_part.get(w, 0) + 1

        name_sum_part = sum(name_part.values())
        insti_sum_part = sum(insti_part.values())
        keywords_sum_part = sum(keywords_part.values())

        for a, b in it.combinations(data.keys(), 2):
            if data[a]["subcode"] == "NULL" or data[b]["subcode"] == "NULL":
                subcode = branch = division = 0
            else:
                subcode = int(data[a]["subcode"] == data[b]["subcode"])
                division = int(data[a]["subcode"][:1] == data[b]["subcode"][:1])
                branch = int(data[a]["subcode"][:3] == data[b]["subcode"][:3])

            diff_year1 = abs(int(data[a]["grandyear"]) - int(data[b]["grandyear"]))
            diff_year2 = int(diff_year1 > 3)

            # project name
            if data[a]["project_name"] == "NULL" or data[b]["project_name"] == "NULL":
                name1 = name2 = name3 = name4 = 0
            else:
                wa = set(jieba.cut(data[a]["project_name"]))
                wb = set(jieba.cut(data[b]["project_name"]))
                inter = wa & wb
                union = wa | wb
                name1 = len(inter)
                name2 = name1 / len(union) if union else 0
                name3 = sum(math.log(name_sum_all / name_corpus[w]) for w in inter) if inter else 0
                name4 = sum(math.log(name_sum_part / name_part[w]) for w in inter) if inter else 0

            # funding
            if data[a]["funding"] == "NULL" or data[b]["funding"] == "NULL":
                funding = 9999
            else:
                funding = abs(float(data[a]["funding"]) - float(data[b]["funding"]))

            # institution
            if data[a]["institution"] == data[b]["institution"]:
                j = data[a]["institution"]
                institution1 = 1
                institution2 = math.log(insti_sum_all / insti_corpus[j])
                institution3 = math.log(insti_sum_part / insti_part[j])
            else:
                institution1 = institution2 = institution3 = 0

            # keywords
            ka = set(split_keywords(data[a]["keywords"]))
            kb = set(split_keywords(data[b]["keywords"]))
            inter = ka & kb
            union = ka | kb
            keywords1 = len(inter)
            keywords2 = keywords1 / len(union) if union else 0
            keywords3 = sum(math.log(keywords_sum_all / keywords_corpus[w]) for w in inter) if inter else 0
            keywords4 = sum(math.log(keywords_sum_part / keywords_part[w]) for w in inter) if inter else 0

            # type
            type_value = int(data[a]["type"] == data[b]["type"]) if "NULL" not in (
                data[a]["type"], data[b]["type"]
            ) else 0

            label = int(data[a]["psncode"] == data[b]["psncode"])

            pairwise_samples.append([
                label, branch, subcode, division,
                diff_year1, diff_year2,
                name1, name2, name3, name4,
                funding,
                institution1, institution2, institution3,
                keywords1, keywords2, keywords3, keywords4,
                type_value
            ])

    return pairwise_samples


def main():
    name_list = read_name_list(NAME_LIST_PATH)
    samples = build_pairwise_features(name_list)
    random.shuffle(samples)

    pairwise_labels = [s[0] for s in samples]
    pairwise_features = [s[1:] for s in samples]

    with h5py.File(OUTPUT_H5_PATH, "w") as f:
        f["pairwise_features"] = pairwise_features
        f["pairwise_labels"] = pairwise_labels


if __name__ == "__main__":
    main()
