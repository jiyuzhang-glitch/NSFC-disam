# -*- coding: utf-8 -*-

from pathlib import Path
import json
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent

INPUT_TXT_PATH = BASE_DIR / "input" / "merged_disam_data.txt"
OUTPUT_DIR = BASE_DIR / "final_json"
MIDFILE_DIR = BASE_DIR / "midfile"
OUTPUT_NAME_LIST = MIDFILE_DIR / "disam_name_list.txt"


def read_txt_data(txt_path: Path) -> List[Dict[str, str]]:
    projects = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 11:
                print(line)
            else:
                projects.append(
                    {
                        "name": parts[4].upper(),
                        "grandyear": parts[7],
                        "grandid": parts[1],
                        "funding": parts[6],
                        "type": parts[3],
                        "institution": parts[5],
                        "project_name": parts[2],
                        "subcode": parts[8],
                        "keywords": parts[10],
                        "psncode": parts[0],
                    }
                )
    print(f"txt length {len(projects)}")
    return projects


def preprocess_projects(txt_projects: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    projects_by_name: Dict[str, List[Dict[str, str]]] = {}
    for project in txt_projects:
        name = project["name"]
        projects_by_name.setdefault(name, []).append(project)
    return projects_by_name


def process_name(
    name: str,
    projects_by_name: Dict[str, List[Dict[str, str]]],
    output_dir: Path,
) -> int:
    data = {}
    i = 1
    for project in projects_by_name.get(name, []):
        item = {
            "grandyear": project["grandyear"],
            "grandid": project["grandid"],
            "funding": project["funding"],
            "type": project["type"],
            "institution": project["institution"],
            "project_name": project["project_name"],
            "subcode": project["subcode"],
            "keywords": project["keywords"],
            "psncode": project["psncode"],
        }
        data[f"{name}{i}"] = item
        i += 1

    if data:
        output_path = output_dir / f"{name}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return len(data)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MIDFILE_DIR.mkdir(parents=True, exist_ok=True)

    txt_projects = read_txt_data(INPUT_TXT_PATH)

    psnname = sorted({proj["name"] for proj in txt_projects})
    with OUTPUT_NAME_LIST.open("w", encoding="utf-8") as f:
        for name in psnname:
            f.write(f"{name}\n")
    print(f"{len(psnname)} 名字列表已成功保存至 {OUTPUT_NAME_LIST}")

    existing_names = {p.stem for p in OUTPUT_DIR.glob("*.json")}

    new_names = [name for name in psnname if name not in existing_names]

    projects_by_name = preprocess_projects(txt_projects)

    for name in new_names:
        count = process_name(name, projects_by_name, OUTPUT_DIR)
        if count > 0:
            print(f"Wrote {count} items for {name}")


if __name__ == "__main__":
    main()
