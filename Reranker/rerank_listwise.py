# Modified from: https://github.com/jataware/XRR2/blob/main/xrr2/rerank.py

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # project dir

import json
from tqdm import tqdm, trange
import argparse
from datasets import load_dataset
from openai import OpenAI
import re
import concurrent.futures 
from utils.eval_util import calculate_retrieval_metrics



def call_api(client, payload):
    response = client.chat.completions.create(**payload)
    result = ""
    try:
        for chunk in response:
            if chunk.choices:
                if chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
    except Exception as e:
        result = "error"
    return result

# Reranker
def _reranker_after(output_str):
    if output_str is None: return None

    if "</think>" in output_str:
        output_str = output_str.split("</think>")[-1]
        
    idxs = re.findall(r"(?:```json\s*)(.+)(?:```)", output_str, re.DOTALL)[-1].strip()
    # print("--", idxs)
    if len(idxs) > 0:
        return {"idxs" : [int(xx) for xx in json.loads(idxs)]}
    else:
        print(output_str)
        raise Exception(f'No idxs found in {output_str}')

def _format_results(all_hits):
    return {
        str(qid) : {
            xx['id']: xx['_score'] for xx in sorted(hit, key=lambda x: x['_score'], reverse=True)
        } for qid, hit in all_hits.items()
    }

def process_single_query(qid, scores, examples, documents, args, rerank_prompt, client):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:args.input_k]
    ctxs = []
    cands = [{'id': did, '_score': score} for did, score in sorted_scores]
    for did, _ in sorted_scores:
        ctxs.append(documents[did])

    doc_str = ''.join(["[{}]. {}\n\n".format(i + 1, re.sub('\n+', ' ', doc)) for i, doc in enumerate(ctxs)])
    prompt = rerank_prompt.format(QUERY=examples[qid]['query'], DOC_STR=doc_str, TOPK=args.k)

    payload = {
        "model": args.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 0.8,
        "stream": True
    }

    try:
        output_str = call_api(client, payload)
        # print(output_str)
        res = _reranker_after(output_str)
        return qid, res, cands
    except Exception as e:
        print(f"Error processing qid {qid}: {e}")
        return qid, {"idxs": [i for i in range(1, args.k+1)]}, cands
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--retriever_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--bm25_score_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing') 
    parser.add_argument('--base_url', type=str, default='https://generativelanguage.googleapis.com/v1beta/openai/')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--llm_model', type=str, default='gemini-2.5-flash')

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
    
    with open(args.retriever_score_file) as f:
        all_scores = json.load(f)

    outputs_path = args.output_dir
    score_file_path = os.path.join(outputs_path, f"{args.reasoning}_score.json")

    client = OpenAI(
    api_key=args.api_key,
    base_url=args.base_url
    )

    rerank_prompt = '''The following documents are related to query: {QUERY}

Documents:
{DOC_STR}

First identify the essential problem in the query. Think step by step to reason about why each document is relevant or irrelevant. Rank these documents based on their relevance to the query.
Please output the ranking result of documents as a list, where the first element is the id of the most relevant document, the second element is the id of the second most element, etc.
Please strictly follow the format to output a list of {TOPK} ids corresponding to the most relevant {TOPK} documents, sorted from the most to least relevant document. First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format like
```json
[... integer ids here ...]
```
'''

    if not os.path.isfile(score_file_path):
        results = {}
        cands = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for qid, scores in all_scores.items():
                future = executor.submit(process_single_query, qid, scores, examples, documents, args, rerank_prompt, client)
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Reranking'):
                qid, res, cand = future.result()
                results[qid] = res
                cands[qid] = cand

        for k, v in results.items():
            # fix 1-based indexing
            results[k]['idxs'] = [i - 1 for i in v['idxs']]

        new_scores = {}
        for qid in results.keys():
            try:
                new_scores[qid] = [{'id': cands[qid][idx]['id'], '_score': args.k - i} for i, idx in enumerate(results[qid]['idxs'])]
            except:
                new_scores[qid] = sorted(cands[qid], key=lambda x: x['_score'], reverse=True)[:args.k]  

        # print(type(new_scores), len(new_scores))
        new_scores = _format_results(new_scores)

        os.makedirs(outputs_path, exist_ok=True)
        with open(score_file_path, 'w') as f:
            json.dump(new_scores, f, indent=2)
    else:
        with open(score_file_path) as f:
            new_scores = json.load(f)
        print(score_file_path,'exists')

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        # for i in e["excluded_ids"]:
        #     if i in documents:
        #         ground_truth[e['id']][i] = 0
        for did in e['excluded_ids']:
            assert not did in new_scores[e['id']]
            assert not did in ground_truth[e['id']]

    results = calculate_retrieval_metrics(results=new_scores, qrels=ground_truth)
    with open(os.path.join(outputs_path, "reranker_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {os.path.join(outputs_path, 'reranker_results.json')}")
