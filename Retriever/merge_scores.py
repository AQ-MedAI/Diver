import json
import os
from datasets import load_dataset
from tqdm import tqdm
from retrievers import calculate_retrieval_metrics
import numpy as np
import pandas as pd


def merge_bm25_dense_score(bm25_file, dense_file, output_file, wgt_dense=0.5):
    wgt_bm25 = 1 - wgt_dense
        
    with open(dense_file, 'r') as f:
        dense_scores = json.load(f)  
    with open(bm25_file, 'r') as f:  
        bm25_scores = json.load(f)

    sorted_merge_scores = {}

    for query_id in dense_scores:
        dense_dict = dense_scores[query_id]
        bm25_dict = bm25_scores.get(query_id, {})

        # Normalize scores
        dense_values = list(dense_dict.values())
        dense_max = max(dense_values) if dense_values else 0
        dense_min = min(dense_values) if dense_values else 0
        bm25_values = list(bm25_dict.values())
        bm25_max = max(bm25_values) if bm25_values else 0
        bm25_min = min(bm25_values) if bm25_values else 0

        all_docs = set(dense_dict.keys()) | set(bm25_dict.keys())

        merged_doc_score = {}
        for did in all_docs:
            xrr_norm = (dense_dict.get(did, 0) - dense_min) / (dense_max - dense_min) if (dense_max - dense_min) != 0 else 0
            bm25_norm = (bm25_dict.get(did, 0) - bm25_min) / (bm25_max - bm25_min) if (bm25_max - bm25_min) != 0 else 0

            merged_doc_score[did] = xrr_norm * wgt_dense + bm25_norm * wgt_bm25

        sorted_docs = sorted(merged_doc_score.items(), key=lambda x: x[1], reverse=True)
        sorted_merge_scores[query_id] = dict(sorted_docs[:100])

    with open(output_file, 'w') as f:
        json.dump(sorted_merge_scores, f, indent=2)

    print("Save merged scores to", output_file)



def get_metrics_by_score(task, score_file):
    examples = load_dataset("parquet", data_files=os.path.join(dataset_source, f"examples/{task}-00000-of-00001.parquet"))["train"]

    with open(score_file, "r") as f:
        scores = json.load(f)

    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e['gold_ids']:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]

    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    output_dir = os.path.dirname(score_file)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    
    return results


if __name__ == "__main__":
    q_extend = "diver_qexpand"
    model_bm25 = 'bm25'
    model_dense = "diver"  # qwen3_4b_5kep_med_rader, reasonir
    dataset_source = "../data/BRIGHT"

    results = pd.DataFrame()


    for task in ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "leetcode", "pony", "aops", "theoremqa_theorems", "theoremqa_questions"]:
        bm25_file = f"./output/{model_bm25}_{q_extend}_reasoning_diver/{task}_{model_bm25}_long_False/{q_extend}_score.json"

        # dense_file = f"./output/reasonir_None_reasoning/{task}_reasonir_long_False/score.json"  # none+reasonir
        dense_file = f"./output/{model_dense}_{q_extend}_reasoning_diver/{task}_{model_dense}_long_False/{q_extend}_score.json"  

        output_dir = f"./output/merge_{q_extend}_{model_bm25}_{q_extend}_{model_dense}_diver"
        task_subdir = os.path.join(output_dir, f"{task}_{model_bm25}_{model_dense}")
        os.makedirs(task_subdir, exist_ok=True)  
        output_task_score_file = os.path.join(task_subdir, f"{task}_merge_score.json")

        wgt_dense = 0.5
        if task in ["leetcode", "pony", "aops", "theoremqa_theorems", "theoremqa_questions"]:
            wgt_dense = 0.7
        print(wgt_dense)
        merge_bm25_dense_score(bm25_file, dense_file, output_file=output_task_score_file, wgt_dense=wgt_dense)
        result = get_metrics_by_score(task, output_task_score_file)
        results = pd.concat([results, pd.DataFrame(result, index=[task])])

    results = results.T * 100  # %
    results['Avg'] = results.mean(axis=1).round(2)
    results.to_csv(f"{output_dir}/analyse_results.csv")
    print(f"Results saved to {output_dir}/analyse_results.csv")