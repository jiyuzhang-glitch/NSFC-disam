# -*- coding: utf-8 -*-

from pathlib import Path
import os
import json
import math
import random
from collections import defaultdict
import itertools as it

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_precision_score, pairwise_recall_score, pairwise_f1_score

RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


BASE_DIR = Path(__file__).resolve().parent
WORKDIR = BASE_DIR  # 以前脚本使用 os.chdir(WORKDIR); 我们使用相对路径以避免改变进程 cwd

INPUT_NAME_LIST = BASE_DIR / "midfile" / "disam_name_list.txt"
INPUT_MERGED_TXT = BASE_DIR / "input" / "merged_disam_data.txt"
FINAL_JSON_DIR = BASE_DIR / "final_json"
FINAL_SIM_DIR = BASE_DIR / "final_similarity"
RESULT_DIR = BASE_DIR / "result"

SUMMARY_PATH = RESULT_DIR / "disam_error_summary.txt"
FINAL_RESULT_PATH = RESULT_DIR / "final_disam_result.txt"
SIM_NAME_LIST_PATH = BASE_DIR / "midfile" / "similarity_name_list.txt"


DB_EPS = 0.285
DB_MIN_SAMPLES = 1
MAX_RETRIES = 1  



def load_name_list(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_gt_mapping(merged_txt_path: Path) -> dict:
    gt_df = pd.read_csv(str(merged_txt_path), sep="\t", header=None, encoding="utf-8", dtype=str)
    n_cols = gt_df.shape[1]
    gt_df.columns = ["psn_code", "grant_id"] + [f"col{i}" for i in range(2, n_cols)]
    mapping = {}
    for _, row in gt_df.iterrows():
        if pd.notna(row["psn_code"]) and pd.notna(row["grant_id"]):
            mapping[str(row["grant_id"])] = str(row["psn_code"])
    return mapping


def rf_distance_indexed_factory(data_ref: dict, sim_ref: dict, gt_map: dict, use_gt_adjustment_flag: dict):
    def RF_distance_indexed(n_idx, m_idx):
        x = list(data_ref.keys())
        a = x[int(n_idx)]
        b = x[int(m_idx)]

        if use_gt_adjustment_flag.get("value", False):
            grandid_a = data_ref[a]["grandid"]
            grandid_b = data_ref[b]["grandid"]

            in_gt_a = str(grandid_a) in gt_map
            in_gt_b = str(grandid_b) in gt_map

            if in_gt_a and in_gt_b:
                if gt_map[str(grandid_a)] == gt_map[str(grandid_b)]:
                    return 0.0
                else:
                    return np.inf

        if a == b:
            return 0.0

        key1 = str(a) + str(b)
        key2 = str(b) + str(a)

        if key1 in sim_ref:
            return float(sim_ref[key1]["distance"])
        elif key2 in sim_ref:
            return float(sim_ref[key2]["distance"])
        else:
            return 1.0

    return RF_distance_indexed


def make_metric_for_dbscan(data_ref: dict, sim_ref: dict, gt_map: dict, use_gt_adjustment_flag: dict):
    rf_metric = rf_distance_indexed_factory(data_ref, sim_ref, gt_map, use_gt_adjustment_flag)

    def metric(n, m):
        try:
            i = int(n[0])
            j = int(m[0])
        except Exception:
            i = int(n)
            j = int(m)
        return rf_metric(i, j)

    return metric


def run_dbscan_with_metric(n_items: int, data_ref: dict, sim_ref: dict, gt_map: dict, use_gt_adjustment_flag: dict):
    X = [[i] for i in range(n_items)]
    metric = make_metric_for_dbscan(data_ref, sim_ref, gt_map, use_gt_adjustment_flag)
    db = DBSCAN(eps=DB_EPS, min_samples=DB_MIN_SAMPLES, metric=metric)
    db.fit(X)
    return np.array(db.labels_)


def check_clustering_issues(predict_label, grandids, gt_map):
    psn_to_clusters = {}
    cluster_to_psns = {}

    for i, lab in enumerate(predict_label):
        grandid = str(grandids[i])
        cluster_id = int(lab)
        if grandid in gt_map:
            psn_code = gt_map[grandid]
            psn_to_clusters.setdefault(psn_code, set()).add(cluster_id)
            cluster_to_psns.setdefault(cluster_id, set()).add(psn_code)

    error_same_psn_split = any(len(clusters) > 1 for clusters in psn_to_clusters.values())
    error_cluster_mixed = any(len(psns) > 1 for psns in cluster_to_psns.values())
    has_error = error_same_psn_split or error_cluster_mixed

    details = {
        "psn_to_clusters": psn_to_clusters,
        "cluster_to_psns": cluster_to_psns,
        "error_same_psn_split": error_same_psn_split,
        "error_cluster_mixed": error_cluster_mixed,
    }
    return has_error, details


def compute_split_and_lumping(labels, grandids, gt_map):
    idxs_with_gt = []
    psn_to_idxs = defaultdict(list)
    for idx, grandid in enumerate(grandids):
        if str(grandid) in gt_map:
            idxs_with_gt.append(idx)
            psn_to_idxs[gt_map[str(grandid)]].append(idx)

    n_gt = len(idxs_with_gt)
    same_pairs = 0
    split_count = 0

    for psn, idxs in psn_to_idxs.items():
        m = len(idxs)
        if m <= 1:
            continue
        same_pairs += m * (m - 1) // 2
        label_counts = defaultdict(int)
        for i in idxs:
            label_counts[int(labels[i])] += 1
        same_label_pairs = sum(c * (c - 1) // 2 for c in label_counts.values())
        split_count += (m * (m - 1) // 2) - same_label_pairs

    total_pairs = n_gt * (n_gt - 1) // 2
    diff_pairs = total_pairs - same_pairs

    label_to_idxs = defaultdict(list)
    for idx in idxs_with_gt:
        label_to_idxs[int(labels[idx])].append(idx)
    lump_count = 0
    for lab, idxs in label_to_idxs.items():
        t = len(idxs)
        if t <= 1:
            continue
        total_label_pairs = t * (t - 1) // 2
        psn_counts = defaultdict(int)
        for i in idxs:
            psn_counts[gt_map[str(grandids[i])]] += 1
        same_psn_pairs_in_label = sum(c * (c - 1) // 2 for c in psn_counts.values())
        lump_count += total_label_pairs - same_psn_pairs_in_label

    split_error = float("nan") if same_pairs == 0 else split_count / same_pairs
    lumping_error = float("nan") if diff_pairs == 0 else lump_count / diff_pairs

    return {
        "n_gt_items": n_gt,
        "same_pairs": same_pairs,
        "split_count": split_count,
        "diff_pairs": diff_pairs,
        "lump_count": lump_count,
        "split_error": split_error,
        "lumping_error": lumping_error,
    }


def pairwise_errors_from_labels(labels, grandids, gt_map):
    true_psn = []
    pred_cluster = []
    for i, gr in enumerate(grandids):
        if str(gr) in gt_map:
            true_psn.append(gt_map[str(gr)])
            pred_cluster.append(int(labels[i]))
    n = len(true_psn)
    out = {}
    if n < 2:
        out.update(
            {
                "pairwise_precision": np.nan,
                "pairwise_recall": np.nan,
                "pairwise_f1": np.nan,
                "aggregate_error": np.nan,
                "split_error": np.nan,
                "n_gt_items": n,
            }
        )
        return out

    p = pairwise_precision_score(true_psn, pred_cluster)
    r = pairwise_recall_score(true_psn, pred_cluster)
    f = pairwise_f1_score(true_psn, pred_cluster)
    out.update(
        {
            "pairwise_precision": p,
            "pairwise_recall": r,
            "pairwise_f1": f,
            "aggregate_error": 1.0 - p,
            "split_error": 1.0 - r,
            "n_gt_items": n,
        }
    )
    return out


DATE_CANDIDATES = [
    "apply_date",
    "award_date",
    "date",
    "year",
    "start_date",
    "pub_date",
    "apply_time",
    "proj_date",
    "starttime",
    "date_awarded",
]


def extract_date_from_entry(entry: dict):
    for fld in DATE_CANDIDATES:
        if fld in entry and entry[fld]:
            try:
                ts = pd.to_datetime(entry[fld], errors="coerce")
                if not pd.isna(ts):
                    return ts
            except Exception:
                pass
    for k, v in entry.items():
        if ("date" in k.lower() or "time" in k.lower()) and v:
            try:
                ts = pd.to_datetime(v, errors="coerce")
                if not pd.isna(ts):
                    return ts
            except Exception:
                pass
    return None


def generate_new_ids(count: int, start_counter: int = 0):
    new_ids = []
    counter = start_counter
    for _ in range(count):
        counter += 1
        new_ids.append("new" + str(counter).zfill(6))

    random.shuffle(new_ids)

    # interleave
    interleaved = []
    lo, hi = 0, len(new_ids) - 1
    while lo <= hi:
        interleaved.append(new_ids[lo])
        lo += 1
        if lo <= hi:
            interleaved.append(new_ids[hi])
            hi -= 1
    return interleaved, counter


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if SUMMARY_PATH.exists():
        SUMMARY_PATH.unlink()
    if FINAL_RESULT_PATH.exists():
        FINAL_RESULT_PATH.unlink()

    with SUMMARY_PATH.open("a", encoding="utf-8") as sf:
        sf.write(
            "\t".join(
                [
                    "name",
                    "n_items",
                    "n_gt_items",
                    "same_pairs",
                    "diff_pairs",
                    "initial_split_error",
                    "initial_lumping_error",
                    "final_split_error",
                    "final_lumping_error",
                    "if_recluster",
                ]
            )
            + "\n"
        )

    with FINAL_RESULT_PATH.open("a", encoding="utf-8") as tf:
        tf.write(
            "\t".join(
                [
                    "name",
                    "grandid",
                    "initial_result",
                    "initial_if_split_err",
                    "initial_if_lump_err",
                    "final_result",
                    "final_if_split_err",
                    "final_if_lump_err",
                    "if_recluster",
                    "group_id_initial",
                    "group_id_final",
                ]
            )
            + "\n"
        )

    names = load_name_list(INPUT_NAME_LIST)
    print("Names to process:", len(names))
    gt_map = load_gt_mapping(INPUT_MERGED_TXT)
    print(f"Ground truth loaded: {len(gt_map)} grants")

    all_name_results = []
    total_need_new_entries = 0
    new_counter = 0

    for name_idx, name in enumerate(names):
        print("Processing:", name)
        json_path = FINAL_JSON_DIR / f"{name}.json"
        sim_path = FINAL_SIM_DIR / f"{name}.json"

        if not json_path.exists():
            print(f"  !! Error: {json_path} not found. Skipping {name}.")
            all_name_results.append(None)
            continue
        if not sim_path.exists():
            print(f"  !! Error: {sim_path} not found. Skipping {name}.")
            all_name_results.append(None)
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        with sim_path.open("r", encoding="utf-8") as f:
            data1 = json.load(f)

        keys_list = list(data.keys())
        grandid_to_key = {}
        for k in keys_list:
            gr = str(data[k].get("grandid", ""))
            if gr and gr not in grandid_to_key:
                grandid_to_key[gr] = k

        grandids = [data[k]["grandid"] for k in keys_list]
        n = len(grandids)

        use_gt_adjustment = {"value": False}
        initial_predict_label = run_dbscan_with_metric(n, data, data1, gt_map, use_gt_adjustment)
        print("  Initial clustering labels:", initial_predict_label)

        init_stats = compute_split_and_lumping(initial_predict_label, grandids, gt_map)

        psn_to_clusters_initial = {}
        cluster_to_psns_initial = {}
        for i, lab in enumerate(initial_predict_label):
            grandid = str(grandids[i])
            cid = int(lab)
            if grandid in gt_map:
                psn = gt_map[grandid]
                psn_to_clusters_initial.setdefault(psn, set()).add(cid)
                cluster_to_psns_initial.setdefault(cid, set()).add(psn)

        if_split_err_initial_flags = [0] * n
        if_lump_err_initial_flags = [0] * n
        for i, lab in enumerate(initial_predict_label):
            grandid = str(grandids[i])
            cid = int(lab)
            if grandid in gt_map:
                psn = gt_map[grandid]
                if_split_err_initial_flags[i] = 1 if len(psn_to_clusters_initial.get(psn, set())) > 1 else 0
                if_lump_err_initial_flags[i] = 1 if len(cluster_to_psns_initial.get(cid, set())) > 1 else 0
            else:
                if_split_err_initial_flags[i] = 0
                if_lump_err_initial_flags[i] = 0

        has_error, details = check_clustering_issues(initial_predict_label, grandids, gt_map)
        attempt = 0
        print("  has_error:", has_error)

        final_predict_label = initial_predict_label.copy()
        if_recluster_flag = 0

        if has_error:
            print(f"  -> Found clustering issue for {name} on initial clustering.")
            re_clustered_success = False
            while attempt < MAX_RETRIES:
                attempt += 1
                print(f"  -> Re-clustering attempt {attempt} with GT adjustment enabled...")
                use_gt_adjustment["value"] = True
                predict_label_after = run_dbscan_with_metric(n, data, data1, gt_map, use_gt_adjustment)
                print("  Re-clustered labels:", predict_label_after)
                has_error_after, details_after = check_clustering_issues(predict_label_after, grandids, gt_map)
                final_predict_label = predict_label_after.copy()
                if not has_error_after:
                    re_clustered_success = True
                    if_recluster_flag = 1
                    break
                else:
                    if_recluster_flag = 1
            if not re_clustered_success:
                print(f"  !! Warning: After {attempt} GT-adjusted attempts, issues remain for {name}.")
        else:
            use_gt_adjustment["value"] = False

        final_stats = compute_split_and_lumping(final_predict_label, grandids, gt_map)

        psn_to_clusters_final = {}
        cluster_to_psns_final = {}
        for i, lab in enumerate(final_predict_label):
            grandid = str(grandids[i])
            cid = int(lab)
            if grandid in gt_map:
                psn = gt_map[grandid]
                psn_to_clusters_final.setdefault(psn, set()).add(cid)
                cluster_to_psns_final.setdefault(cid, set()).add(psn)

        if_split_err_flags = [0] * n
        if_lump_err_flags = [0] * n
        for i, lab in enumerate(final_predict_label):
            grandid = str(grandids[i])
            cid = int(lab)
            if grandid in gt_map:
                psn = gt_map[grandid]
                if_split_err_flags[i] = 1 if len(psn_to_clusters_final.get(psn, set())) > 1 else 0
                if_lump_err_flags[i] = 1 if len(cluster_to_psns_final.get(cid, set())) > 1 else 0
            else:
                if_split_err_flags[i] = 0
                if_lump_err_flags[i] = 0

        cluster_to_indices_initial = defaultdict(list)
        for idx, lab in enumerate(initial_predict_label):
            cluster_to_indices_initial[int(lab)].append(idx)
        cluster_to_indices_final = defaultdict(list)
        for idx, lab in enumerate(final_predict_label):
            cluster_to_indices_final[int(lab)].append(idx)

        def map_clusters_to_psn_and_need_new(cluster_to_idxs):
            cluster_to_groupid_psn = {}
            clusters_need_new = []
            for cid, idxs in cluster_to_idxs.items():
                psn_set_local = set()
                psn_to_dates = defaultdict(list)
                for i in idxs:
                    grandid = str(grandids[i])
                    if grandid in gt_map:
                        psn = gt_map[grandid]
                        psn_set_local.add(psn)
                        key = grandid_to_key.get(grandid)
                        if key and key in data:
                            ts = extract_date_from_entry(data[key])
                            if ts is not None:
                                psn_to_dates[psn].append(ts)
                if psn_set_local:
                    psn_to_min = {}
                    for p in psn_set_local:
                        dates = psn_to_dates.get(p, [])
                        psn_to_min[p] = min(dates) if dates else None
                    psn_with_time_local = {p: d for p, d in psn_to_min.items() if d is not None}
                    if psn_with_time_local:
                        chosen = min(psn_with_time_local.items(), key=lambda x: x[1])[0]
                    else:
                        chosen = sorted(list(psn_set_local))[0]
                    cluster_to_groupid_psn[cid] = "psn" + str(chosen)
                else:
                    clusters_need_new.append(cid)
            return cluster_to_groupid_psn, clusters_need_new

        cluster_to_groupid_psn_initial, clusters_need_new_initial = map_clusters_to_psn_and_need_new(cluster_to_indices_initial)
        cluster_to_groupid_psn_final, clusters_need_new_final = map_clusters_to_psn_and_need_new(cluster_to_indices_final)

        need_new_count_for_name = len(clusters_need_new_initial) + len(clusters_need_new_final)
        total_need_new_entries += need_new_count_for_name

        all_name_results.append(
            {
                "name": name,
                "keys_list": keys_list,
                "grandid_to_key": grandid_to_key,
                "grandids": grandids,
                "n": n,
                "initial_predict_label": initial_predict_label,
                "final_predict_label": final_predict_label,
                "init_stats": init_stats,
                "final_stats": final_stats,
                "if_recluster_flag": if_recluster_flag,
                "if_split_err_initial_flags": if_split_err_initial_flags,
                "if_lump_err_initial_flags": if_lump_err_initial_flags,
                "if_split_err_final_flags": if_split_err_flags,
                "if_lump_err_final_flags": if_lump_err_flags,
                "cluster_to_indices_initial": cluster_to_indices_initial,
                "cluster_to_indices_final": cluster_to_indices_final,
                "cluster_to_groupid_psn_initial": cluster_to_groupid_psn_initial,
                "cluster_to_groupid_psn_final": cluster_to_groupid_psn_final,
                "clusters_need_new_initial": clusters_need_new_initial,
                "clusters_need_new_final": clusters_need_new_final,
            }
        )

        with SUMMARY_PATH.open("a", encoding="utf-8") as sf:
            sf.write(
                "\t".join(
                    [
                        str(name),
                        str(n),
                        str(init_stats["n_gt_items"]),
                        str(init_stats["same_pairs"]),
                        str(init_stats["diff_pairs"]),
                        (f"{init_stats['split_error']:.6f}" if not math.isnan(init_stats["split_error"]) else "nan"),
                        (f"{init_stats['lumping_error']:.6f}" if not math.isnan(init_stats["lumping_error"]) else "nan"),
                        (f"{final_stats['split_error']:.6f}" if not math.isnan(final_stats["split_error"]) else "nan"),
                        (f"{final_stats['lumping_error']:.6f}" if not math.isnan(final_stats["lumping_error"]) else "nan"),
                        str(if_recluster_flag),
                    ]
                )
                + "\n"
            )
            sf.flush()

        cluster_to_psns_overall = {}
        for i, lab in enumerate(final_predict_label):
            grandid = str(grandids[i])
            cluster_id = int(lab)
            if grandid in gt_map:
                cluster_to_psns_overall.setdefault(cluster_id, set()).add(gt_map[grandid])
        print("  Cluster -> gt_psn_count summary (only grants with GT shown):")
        for cid, psns in sorted(cluster_to_psns_overall.items()):
            print(f"    cluster {cid}: {len(psns)} distinct GT psn(s) -> {list(psns)[:5]}")

    print("First pass done. Total clusters needing new ids (initial+final):", total_need_new_entries)

    interleaved_new_ids, new_counter = generate_new_ids(total_need_new_entries, start_counter=new_counter)

    queues_per_name = []
    for name_idx, item in enumerate(all_name_results):
        if item is None:
            queues_per_name.append([])
            continue
        q = []
        for cid in item["clusters_need_new_initial"]:
            q.append((name_idx, "init", cid))
        for cid in item["clusters_need_new_final"]:
            q.append((name_idx, "final", cid))
        queues_per_name.append(q)

    order_entries = []
    while True:
        any_pulled = False
        for name_idx in range(len(queues_per_name)):
            if queues_per_name[name_idx]:
                order_entries.append(queues_per_name[name_idx].pop(0))
                any_pulled = True
        if not any_pulled:
            break

    if len(order_entries) != len(interleaved_new_ids):
        print(
            "Warning: mismatch between entries needing new ids and generated new ids!",
            len(order_entries),
            len(interleaved_new_ids),
        )

    assignment_map = {}
    for entry, gid in zip(order_entries, interleaved_new_ids):
        assignment_map[entry] = gid

    with FINAL_RESULT_PATH.open("a", encoding="utf-8") as tf:
        for name_idx, item in enumerate(all_name_results):
            if item is None:
                continue
            name = item["name"]
            keys_list = item["keys_list"]
            grandids = item["grandids"]
            initial_predict_label = item["initial_predict_label"]
            final_predict_label = item["final_predict_label"]
            if_recluster_flag = item["if_recluster_flag"]

            cluster_to_groupid_initial = dict(item["cluster_to_groupid_psn_initial"])
            cluster_to_groupid_final = dict(item["cluster_to_groupid_psn_final"])

            for cid in item["clusters_need_new_initial"]:
                key = (name_idx, "init", cid)
                gid = assignment_map.get(key)
                if gid is None:
                    new_counter += 1
                    gid = "new" + str(new_counter).zfill(6)
                cluster_to_groupid_initial[cid] = gid

            for cid in item["clusters_need_new_final"]:
                key = (name_idx, "final", cid)
                gid = assignment_map.get(key)
                if gid is None:
                    new_counter += 1
                    gid = "new" + str(new_counter).zfill(6)
                cluster_to_groupid_final[cid] = gid

            for i in range(len(initial_predict_label)):
                init_lab = int(initial_predict_label[i])
                final_lab = int(final_predict_label[i])
                init_split_flag = item["if_split_err_initial_flags"][i]
                init_lump_flag = item["if_lump_err_initial_flags"][i]
                final_split_flag = item["if_split_err_final_flags"][i]
                final_lump_flag = item["if_lump_err_final_flags"][i]
                grp_init = cluster_to_groupid_initial.get(int(init_lab), "")
                grp_final = cluster_to_groupid_final.get(int(final_lab), "")

                tf.write(
                    "\t".join(
                        [
                            str(name),
                            str(grandids[i]),
                            str(init_lab),
                            str(init_split_flag),
                            str(init_lump_flag),
                            str(final_lab),
                            str(final_split_flag),
                            str(final_lump_flag),
                            str(if_recluster_flag),
                            str(grp_init),
                            str(grp_final),
                        ]
                    )
                    + "\n"
                )
            tf.flush()

    try:
        init_pairwise = pairwise_errors_from_labels(initial_predict_label, grandids, gt_map)
        final_pairwise = pairwise_errors_from_labels(final_predict_label, grandids, gt_map)
        print("Pairwise (example) initial:", init_pairwise)
        print("Pairwise (example) final:", final_pairwise)
    except Exception:
        pass

    print("All done. Wrote final results to", FINAL_RESULT_PATH)


if __name__ == "__main__":
    main()
