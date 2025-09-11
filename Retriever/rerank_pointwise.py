import copy
import os
import json
from tqdm import tqdm, trange
import argparse
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer as get_vllm_tokenizer
from datasets import load_dataset
import torch
import prompts
import time 
import re
from retrievers import calculate_retrieval_metrics



class Reranker:
    def __init__(self, task):
        self.model_name = "Qwen/Qwen2.5-32B-Instruct"
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128, logprobs=10)
        self.tokenizer = get_vllm_tokenizer(self.model_name, trust_remote_code=False)
        self.model = LLM(model=self.model_name, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_model_len=16384)

        retrieval_dict = {
                "aops": prompts.bright_aops,
                "theoremqa_questions": prompts.bright_theoremqa_questions,
                "leetcode": prompts.bright_leetcode,
                "pony": prompts.bright_pony,
                "theoremqa_theorems": prompts.bright_theoremqa_theorems
            }
        if args.task in retrieval_dict:
            self.prompt = retrieval_dict[task]
        else:
            self.prompt = prompts.bright_general
        
        # set random seed
        torch.manual_seed(42)

    def truncate_text(self, doc_text, max_tokens=100):
        doc_ids = self.tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_ids) > max_tokens:
            doc_ids = doc_ids[:max_tokens]

        return self.tokenizer.decode(doc_ids, skip_special_tokens=True)

    def rerank(self, docs, query, topk):
        scores = []
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            list_docs = docs[i:i+batch_size]
            truncated_text = [self.truncate_text(doc["text"], max_tokens=15000) for doc in list_docs]
            doc_prompts = [self.prompt.format(query, doc) for doc in truncated_text]
            
            t1 = time.time()
            output = self.model.generate(doc_prompts, self.sampling_params, use_tqdm=False)  
            doc_prompt_outputs = [o.outputs[0].text for o in output]

            for j in range(len(doc_prompt_outputs)):
                if doc_prompt_outputs[j] is None:
                    print(f"Received None for prompt {i+j}, setting score to 0.")
                    scores.append(0)
                    continue
                try:
                    # match = re.search(r"Relevance\s*score[:\s]*([0-5])", doc_prompt_outputs[j], re.IGNORECASE)
                    match = re.search(r"Relevance\s*score[:\s]*(10|0?\d(\.\d+)?)", doc_prompt_outputs[j], re.IGNORECASE)

                    if match:
                        score = float(match.group(1)) / 10
                        scores.append(score)
                    else:
                        raise ValueError("Score not found in expected format")
                except:
                    print("In exception!! {}".format(doc_prompt_outputs[j]))
                    scores.append(0)

        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])   
        return ranking


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

    if not os.path.isfile(score_file_path):
        new_scores = copy.deepcopy(all_scores)
        model = Reranker(args.task)
        for qid,scores in tqdm(all_scores.items()):
            docs = []
            sorted_scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:args.input_k]
            for did, _ in sorted_scores:
                docs.append([did, documents[did]])

            ctxs = [{'id': did, 'text': documents[did]} for did, _ in sorted_scores]            
            cur_score = model.rerank(query=examples[qid]['query'], docs=ctxs, topk=args.k) 
            assert len(cur_score) == len(sorted_scores)

            new_scores[qid] = cur_score

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
        for did in e['excluded_ids']:
            assert not did in new_scores[e['id']]
            assert not did in ground_truth[e['id']]


    results = calculate_retrieval_metrics(results=new_scores, qrels=ground_truth)
    with open(os.path.join(outputs_path, "reranker_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # break ties by interpolating with the retriever scores
    retriever_interpolated_scores = {}
    for qid in new_scores:
        retriever_interpolated_scores[qid] = {}
        for did in new_scores[qid]:
            retriever_interpolated_scores[qid][did] = (0.5 * new_scores[qid][did]) + (0.5 * all_scores[qid][did])
        retriever_interpolated_scores[qid] = dict(sorted(retriever_interpolated_scores[qid].items(),key=lambda x:x[1],reverse=True))
    results = calculate_retrieval_metrics(results=retriever_interpolated_scores, qrels=ground_truth)
    with open(os.path.join(outputs_path, f"reranker_retriever_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    score_file_path = os.path.join(outputs_path, f"{args.reasoning}_reranked_score.json")
    with open(score_file_path, 'w') as f:
        json.dump(retriever_interpolated_scores, f, indent=2)


    # break ties by combining with the bm25 scores
    if args.bm25_score_file is not None:
        bm25_interpolated_scores = {}
        with open(args.bm25_score_file) as f:
            bm25_scores = json.load(f)
        for qid in new_scores:
            bm25_interpolated_scores[qid] = {}
            for did in new_scores[qid]:
                bm25_interpolated_scores[qid][did] = (100 * new_scores[qid][did]) + bm25_scores[qid][did]
        results = calculate_retrieval_metrics(results=bm25_interpolated_scores, qrels=ground_truth)
        with open(os.path.join(outputs_path, f"reranker_bm25_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
