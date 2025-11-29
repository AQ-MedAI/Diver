import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # project dir

import argparse
import re
from vllm import LLM, SamplingParams
import os
import torch
from datasets import load_dataset
import json
import copy
import random
from tqdm import tqdm
from collections import defaultdict
from utils.eval_util import calculate_retrieval_metrics

random.seed(666)


def avg_scores(did_scores):
    # 1. 使用 defaultdict(list) 来按 doc_id 聚合所有分数
    scores_by_doc = defaultdict(list)
    for doc_id, score in did_scores:
        scores_by_doc[doc_id].append(score)

    # 2. 计算每个 doc_id 的平均分
    average_scores = {}
    for doc_id, scores in scores_by_doc.items():
        average_scores[doc_id] = sum(scores) / len(scores)
    
    return [(k,v) for k,v in average_scores.items()]


class GroupReranker:
    def __init__(self, model_path) -> None:
        self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(), max_model_len=32000)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(temperature=0.3, top_p=0.8, max_tokens=8000, logprobs=10)

        self.group_system_prompt = '''Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
1. PRIMARY: Usefulness & Helpfulness - Does the document provide actionable information, solutions, or direct answers that help address the user's needs?
2. SECONDARY: Relevance - Does the document contain information related to the query topic?

Evaluation Process:
1. First, identify the user's core intent and what kind of help they need from the query
2. For each document, assess:
   - How directly it addresses the user's intent
   - What actionable information or answers it provides
   - How much it helps solve the user's problem or need
3. Compare documents against each other to ensure proper ranking
4. Assign scores that reflect the relative usefulness ranking

Scoring Scale (0-10):
- 9-10: Extremely helpful, directly answers the query with actionable information
- 7-8: Very helpful, provides substantial useful information for the query
- 5-6: Moderately helpful, contains some useful information but incomplete
- 3-4: Minimally helpful, limited useful information despite topic relevance
- 1-2: Barely helpful, mentions related topics but provides little useful information
- 0: Not helpful at all, cannot assist with answering the query
'''

        self.group_user_prompt = '''I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
Query:
{QUERY}

Documents:
{PASSAGES}

## Final Output Format
You must structure your response in exactly two parts: provide your brief reasoning process first, then output final scores in JSON format like below, with document IDs as string keys and integer scores as values for all {TOPK} documents. 
The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively. Do NOT output anything outside the specified tags. Follow this exact format:
<reason> 
[Analyze each document's usefulness and relevance to the query, explaining your scoring rationale]
</reason>
<answer> 
```json
{{"[1]": 5, "[2]": 3, "[3]": 8}}
``` 
</answer>
'''

    def _extract_rerank_output(self, output_str):
        if output_str is None: return None

        # 把answer部分提取出来
        answer_match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
        if answer_match:
            output_str = answer_match.group(1).strip()
        
        # 定位JSON代码块
        # 方法1: 尝试提取带```json标记的内容
        try:
            json_matches = re.findall(r"(?:```json\s*)([\s\S]+?)(?:\s*```)", output_str)
            if json_matches:
                json_str = json_matches[-1].strip()
            else:
                # 方法2: 未找到标记时，尝试提取整个JSON结构
                json_str = re.findall(r"\{[\s\S]*?\}", output_str)[-1]

            # 修复常见的JSON格式问题
            json_str = json_str.strip()
            json_str = re.sub(r'(\[\d+\])\]', r'\1', json_str) # 1. 修复多余的冒号 (如 "[7]]:" -> "[7]":")
            if json_str.count('{') > json_str.count('}'): # 2. 确保JSON完整（如果缺少结束括号）
                json_str += '}'
            json_str = re.sub(r'\s+', ' ', json_str).replace('\n', ' ').strip() # 3. 移除多余的空白字符
            # print(f"处理后的JSON字符串: {json_str}")
            
            idx_score = dict(json.loads(json_str))
        except:
            print(output_str)
            raise ValueError("未找到有效的JSON内容")
        return idx_score

    def prepare_groups_one_epoch(self, qid_queries, qid_did_ctxs, topk, group_size=10, interval=20, max_doc_lens=3000):
        # 把要推理的所有内容打包成batch
        batch_messages = []
        batch_info = []  # 保存qid, dids, doc_texts等
        for item in qid_queries:
            qid, query = item['qid'], item['query']
            docs = qid_did_ctxs[qid]
            random.shuffle(docs)
            for i in range(0, len(docs), interval):
                doc_group = docs[i:i+group_size]
                dids = []
                doc_texts = []
                for doc in doc_group:
                    dids.append(doc['id'])
                    doc_texts.append(re.sub('\n+', ' ', doc['text'])[:max_doc_lens])
                docs_str = ''.join(["[{}]. {}\n\n".format(idx+1, doc_text) for idx, doc_text in enumerate(doc_texts)])
                group_texts = self.group_user_prompt.format(QUERY=query, PASSAGES=docs_str, TOPK=len(dids))
                message = self.tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': self.group_system_prompt}, 
                     {'role': 'user', 'content': group_texts}], tokenize=False, add_generation_prompt=True)
                batch_messages.append(message)
                batch_info.append((qid, dids))

        # print(f"Total {len(batch_messages)} groups to score.")

        return batch_messages, batch_info

    def rerank_by_group(self, qid_queries, qid_did_ctxs, topk, group_size=20, interval=20, num_epoch=2, max_doc_lens=3000):
        total_batch_messages, total_batch_info = [], []
        for epoch in range(num_epoch):
            batch_messages, batch_info = self.prepare_groups_one_epoch(qid_queries, qid_did_ctxs, topk, group_size, interval, max_doc_lens)
            total_batch_messages.extend(batch_messages)
            total_batch_info.extend(batch_info)
        
        print(f"Total {len(total_batch_messages)} groups to score.")

        # 批量推理
        outputs = self.llm.generate(total_batch_messages, self.sampling_params, use_tqdm=True)
        
        #  outputs为一个list，每个元素对应一条输入
        new_scores = {item['qid']: [] for item in qid_queries}
        
        for out, (qid, dids) in tqdm(zip(outputs, total_batch_info), total=len(total_batch_info)):
            output_str = out.outputs[0].text
            try:
                idx_score = self._extract_rerank_output(output_str)
                did_score = [(dids[int(idx.strip().strip('[]')) - 1], score/10) for idx, score in idx_score.items()] #did_score 格式[(did, score), (did, score), ...]
            except Exception as e:
                print(output_str)
                print('--group:', dids)
                print("解析错误", e)
                did_score = [(did, 0) for did in dids]
            new_scores[qid].extend(did_score)
        # 选topk
        # import ipdb;ipdb.set_trace()
        for qid, did_scores in new_scores.items():
            did_scores = avg_scores(did_scores)
            new_scores[qid] = dict(sorted(did_scores, key=lambda item: item[1], reverse=True)[:topk])
        return new_scores


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--model_path', type=str, default='AQ-MedAI/Diver-GroupRank-32B')
    parser.add_argument('--retriever_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--bm25_score_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    # parser.add_argument('--rerank_style', type=str, default='point')
    parser.add_argument('--group_size', type=int, default=10)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--repeat_num', type=int, default=2)
    parser.add_argument('--max_doc_len', type=int, default=3000)

    
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
        model = GroupReranker(model_path=args.model_path)

        print("Using group reranking style")
        qid_list = list(all_scores.keys())
        qid_queries = [{'qid': qid, 'query': examples[qid]['query']} for qid in qid_list]
        qid_did_ctxs = {}
        for qid, scores in all_scores.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:args.input_k]
            docs = []
            for did, _ in sorted_scores:
                docs.append([did, documents[did]])
            qid_did_ctxs[qid] = [{'id': did, 'text': documents[did]} for did, _ in sorted_scores]

        new_scores = model.rerank_by_group(qid_queries=qid_queries, qid_did_ctxs=qid_did_ctxs, topk=args.k, group_size=args.group_size, interval=args.interval, num_epoch=args.repeat_num, max_doc_lens=args.max_doc_len)

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
            retriever_interpolated_scores[qid][did] = (0.6 * new_scores[qid][did]) + (0.4 * all_scores[qid][did])
        retriever_interpolated_scores[qid] = dict(sorted(retriever_interpolated_scores[qid].items(),key=lambda x:x[1],reverse=True))
    results = calculate_retrieval_metrics(results=retriever_interpolated_scores, qrels=ground_truth)
    with open(os.path.join(outputs_path, f"reranker_retriever_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    score_file_path = os.path.join(outputs_path, f"{args.reasoning}_reranked_score.json")
    with open(score_file_path, 'w') as f:
        json.dump(retriever_interpolated_scores, f, indent=2)

