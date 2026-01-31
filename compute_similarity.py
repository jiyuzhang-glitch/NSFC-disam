# -*- coding: utf-8 -*-

from pathlib import Path
import json
import math
import itertools as it
import jieba
import joblib
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "final_model.m"
CORPUS_DIR = BASE / "corpus"
MIDFILE_PATH = BASE / "midfile" / "disam_name_list.txt"
FINAL_JSON_DIR = BASE / "final_json"
FINAL_SIM_DIR = BASE / "final_similarity"
SIM_NAME_LIST = BASE / "midfile" / "similarity_name_list.txt" 

_PLACEHOLDERS = {"", "*", "**", "***", "NA"}

def split_keywords(raw: str):
    if raw is None:
        return []
    separators = [";", "；", ",", "，", "."]
    for sep in separators:
        if sep in raw:
            parts = raw.split(sep)
            break
    else:
        parts = raw.split(" ")
    return [p for p in parts if p not in _PLACEHOLDERS]


def load_npy_dict(path: Path):
    arr = np.load(str(path), allow_pickle=True)
    return arr.item() if hasattr(arr, "item") else dict(arr)


def ensure_dirs():
    FINAL_SIM_DIR.mkdir(parents=True, exist_ok=True)
    (BASE / "midfile").mkdir(parents=True, exist_ok=True)


def read_name_list(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_similarity_json(name: str, data: dict):
    outp = FINAL_SIM_DIR / f"{name}.json"
    with outp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))


def build_feature_vector(data: dict, a: str, b: str,
                         name_sum_all: int, name_corpus: dict, name_sum_part: int, name_part: dict,
                         insti_sum_all: int, insti_corpus: dict, insti_sum_part: int, insti_part: dict,
                         keywords_sum_all: int, keywords_corpus: dict, keywords_sum_part: int, keywords_part: dict):
    # branch / division / subcode
    if data[a]['subcode'] == 'NULL' or data[b]['subcode'] == 'NULL':
        subcode = 0
        branch = 0
        division = 0
    else:
        subcode = 1 if data[a]['subcode'] == data[b]['subcode'] else 0
        division = 1 if data[a]['subcode'][0:1] == data[b]['subcode'][0:1] else 0
        branch = 1 if data[a]['subcode'][0:3] == data[b]['subcode'][0:3] else 0

    # diff_year
    diff_year1 = abs(int(data[a]['grandyear']) - int(data[b]['grandyear']))
    diff_year2 = 1 if diff_year1 > 3 else 0

    # project_name features
    if data[a]['project_name'] == 'NULL' or data[b]['project_name'] == 'NULL':
        name1 = name2 = name3 = name4 = 0
    else:
        projectname_a = data[a]['project_name'].replace('<br/>', ' ').strip()
        projectname_b = data[b]['project_name'].replace('<br/>', ' ').strip()
        words_a = list(jieba.cut(projectname_a))
        words_b = list(jieba.cut(projectname_b))

        a_inter_b = list(set(words_a).intersection(set(words_b)))
        name1 = len(a_inter_b)
        name_union = list(set(words_a).union(set(words_b)))
        name2 = 0 if len(name_union) == 0 else name1 / len(name_union)

        if name1 == 0:
            name3 = 0
            name4 = 0
        else:
            name3 = 0
            name4 = 0
            for j in a_inter_b:
                name3 += math.log(int(name_sum_all) / int(name_corpus[j]))
                name4 += math.log(int(name_sum_part) / int(name_part[j]))

    # funding
    if data[a]['funding'] == 'NULL' or data[b]['funding'] == 'NULL':
        funding = 9999
    else:
        funding = abs(float(data[a]['funding']) - float(data[b]['funding']))

    # institution features
    if data[a]['institution'] == 'NULL' or data[b]['institution'] == 'NULL':
        institution1 = 0
        institution2 = 0
        institution3 = 0
    else:
        if data[a]['institution'] == data[b]['institution']:
            j = data[a]['institution']
            institution1 = 1
            institution2 = math.log(int(insti_sum_all) / int(insti_corpus[j]))
            institution3 = math.log(int(insti_sum_part) / int(insti_part[j]))
        else:
            institution1 = 0
            institution2 = 0
            institution3 = 0

    # keywords features
    keywords_a = split_keywords(data[a]['keywords'].strip())
    keywords_b = split_keywords(data[b]['keywords'].strip())

    if len(keywords_a) == 0 or len(keywords_b) == 0:
        keywords1 = 0
        keywords2 = 0
        keywords3 = 0
        keywords4 = 0
    else:
        keywords_inter = list(set(keywords_a).intersection(set(keywords_b)))
        keywords1 = len(keywords_inter)
        keywords_union = list(set(keywords_a).union(set(keywords_b)))
        keywords2 = keywords1 / len(keywords_union)
        if keywords1 == 0:
            keywords3 = 0
            keywords4 = 0
        else:
            keywords3 = 0
            keywords4 = 0
            for j in keywords_inter:
                keywords3 += math.log(int(keywords_sum_all) / int(keywords_corpus[j]))
                keywords4 += math.log(int(keywords_sum_part) / int(keywords_part[j]))

    # type
    if data[a]['type'] == 'NULL' or data[b]['type'] == 'NULL':
        type_value = 0
    else:
        type_value = 1 if data[a]['type'] == data[b]['type'] else 0

    # Build feature vector in the exact order used by original script
    feature_vector = [
        branch, subcode, division, diff_year1, diff_year2,
        name1, name2, name3, name4,
        funding,
        institution1, institution2, institution3,
        keywords1, keywords2, keywords3, keywords4,
        type_value
    ]

    return [feature_vector]  # 2D list for sklearn API compatibility


def main():
    ensure_dirs()

    clf = joblib.load(str(MODEL_PATH))

    name_corpus = load_npy_dict(CORPUS_DIR / "name_corpus.npy")
    name_sum_all = sum(list(name_corpus.values()))

    insti_corpus = load_npy_dict(CORPUS_DIR / "insti_corpus.npy")
    insti_sum_all = sum(list(insti_corpus.values()))

    keywords_corpus = load_npy_dict(CORPUS_DIR / "keywords_corpus.npy")
    keywords_sum_all = sum(list(keywords_corpus.values()))

    all_names = read_name_list(MIDFILE_PATH)
    already = {p.stem for p in FINAL_SIM_DIR.glob("*.json")}
    new_names = [n for n in all_names if n not in already]

    with SIM_NAME_LIST.open("w", encoding="utf-8") as f:
        for n in new_names:
            f.write(f"{n}\n")

    for name in new_names:
        print(name)
        data1 = {}

        json_path = FINAL_JSON_DIR / f"{name}.json"
        if not json_path.exists():
            continue

        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        name_part = {}
        insti_part = {}
        keywords_part = {}

        for keys in data.keys():
            pn = data[keys]['project_name'].replace('<br/>', ' ').strip()
            for w in list(jieba.cut(pn)):
                name_part[w] = name_part.get(w, 0) + 1

            inst = data[keys]['institution'].strip()
            insti_part[inst] = insti_part.get(inst, 0) + 1

            for w in split_keywords(data[keys]['keywords'].strip()):
                keywords_part[w] = keywords_part.get(w, 0) + 1

        name_sum_part = sum(list(name_part.values()))
        insti_sum_part = sum(list(insti_part.values()))
        keywords_sum_part = sum(list(keywords_part.values()))

        for (a, b) in it.combinations(data.keys(), 2):
            item1 = {}

            name_input = build_feature_vector(
                data, a, b,
                name_sum_all, name_corpus, name_sum_part, name_part,
                insti_sum_all, insti_corpus, insti_sum_part, insti_part,
                keywords_sum_all, keywords_corpus, keywords_sum_part, keywords_part
            )

            predict_proba = clf.predict_proba(name_input)
            distance = predict_proba[0][0]

            item1['distance'] = str(distance)
            data1[str(a) + str(b)] = item1

        write_similarity_json(name, data1)


if __name__ == "__main__":
    main()
