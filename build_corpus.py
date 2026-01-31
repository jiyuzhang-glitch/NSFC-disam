import os
import csv
import numpy as np
import jieba


def read_csv_data_for_corpora(csv_path):
    projects = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            projects.append(
                {
                    "project_name": row["项目名称"].strip(),
                    "institution": row["依托单位"].strip(),
                    "keywords": row["关键词"].strip(),
                }
            )
    return projects


def generate_corpora(csv_projects, corpus_dir):
    name_corpus = {}
    insti_corpus = {}
    keywords_corpus = {}

    for project in csv_projects:
        update_corpora(project, name_corpus, insti_corpus, keywords_corpus)

    os.makedirs(corpus_dir, exist_ok=True)

    np.save(os.path.join(corpus_dir, "name_corpus.npy"), name_corpus)
    np.save(os.path.join(corpus_dir, "insti_corpus.npy"), insti_corpus)
    np.save(os.path.join(corpus_dir, "keywords_corpus.npy"), keywords_corpus)


def update_corpora(project, name_corpus, insti_corpus, keywords_corpus):
    projectname = project["project_name"]
    institution = project["institution"]
    keywords = project["keywords"]

    for word in jieba.cut(projectname):
        name_corpus[word] = name_corpus.get(word, 0) + 1

    insti_corpus[institution] = insti_corpus.get(institution, 0) + 1

    process_keywords(keywords, keywords_corpus)


def process_keywords(keywords, keywords_corpus):
    separators = [";", "；", ",", "，", ".", " "]
    for sep in separators:
        if sep in keywords:
            keywords = keywords.split(sep)
            break
    else:
        keywords = [keywords]

    keywords = [w for w in keywords if w not in ["", "*", "**", "***", "NA"]]

    for word in keywords:
        keywords_corpus[word] = keywords_corpus.get(word, 0) + 1


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_csv_path = os.path.join(BASE_DIR, "input", "project_izaiwen.csv")
    corpus_dir = os.path.join(BASE_DIR, "corpus")

    csv_projects = read_csv_data_for_corpora(input_csv_path)

    generate_corpora(csv_projects, corpus_dir)
