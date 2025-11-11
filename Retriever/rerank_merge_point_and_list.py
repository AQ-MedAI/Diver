import os
import json
import argparse
from datasets import load_dataset
import math
from retrievers import calculate_retrieval_metrics


def reassign_score(scores):
    # 排第一的分数为1，其他文档分数按排名log缩放
    for qid in scores:
        num_docs = len(scores[qid])
        for rank, did in enumerate(scores[qid]):
            scores[qid][did] = -math.log(rank + 1)
        
    return scores


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--pointwise_score_file', type=str, default=None)
    parser.add_argument('--listwise_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    if args.reasoning is not None:
        json_path = os.path.join(f"../data/BRIGHT/{args.reasoning}_reason", f"{args.task}_query.json")
        raw_examples = load_dataset("json", data_files=json_path)["train"]
    else:
        dataset_source = '../data/BRIGHT'
        raw_examples = load_dataset("parquet", data_files=os.path.join(dataset_source, f"examples/{args.task}-00000-of-00001.parquet"))["train"]

    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('../data/BRIGHT', 'long_documents', cache_dir=args.cache_dir)[args.task]
    else:
        dataset_source = '../data/BRIGHT'
        doc_pairs = load_dataset("parquet", data_files=os.path.join(dataset_source, f"documents/{args.task}-00000-of-00001.parquet"))["train"]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
    

    # 加载listwise reranker分数，并计算各文档按排名顺序的得分
    print("list rerank file: ", args.listwise_score_file)
    if args.listwise_score_file is not None:
        with open(args.listwise_score_file) as f:
            list_rerank_scores = json.load(f)
        results = calculate_retrieval_metrics(results=list_rerank_scores, qrels=ground_truth)  # 打印listwise结果
        list_rerank_scores = reassign_score(list_rerank_scores)
    else:   
        list_rerank_scores = None
    
    
    # 加载point reranker分数，并计算各文档按排名顺序的得分
    print("point rerank file: ", args.pointwise_score_file)
    if args.pointwise_score_file is not None:
        with open(args.pointwise_score_file) as f:
            point_rerank_scores = json.load(f)
        results = calculate_retrieval_metrics(results=point_rerank_scores, qrels=ground_truth)
        point_rerank_scores = reassign_score(point_rerank_scores)
    else:
        point_rerank_scores = None

    # 加权两种排名后的输出文件
    outputs_path = args.output_dir  
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    score_file_path = os.path.join(outputs_path, f"{args.reasoning}_score.json")

    if point_rerank_scores is not None:
        rerank_interpolated_scores = {}
        for qid in point_rerank_scores:
            rerank_interpolated_scores[qid] = {}
            for did in point_rerank_scores[qid]:
                rerank_interpolated_scores[qid][did] = 0.6 * point_rerank_scores[qid][did] + 0.4 * list_rerank_scores[qid][did]
            rerank_interpolated_scores[qid] = dict(sorted(rerank_interpolated_scores[qid].items(),key=lambda x:x[1],reverse=True))
        results = calculate_retrieval_metrics(results=rerank_interpolated_scores, qrels=ground_truth)
        with open(os.path.join(outputs_path, f"reranker_point_list_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
