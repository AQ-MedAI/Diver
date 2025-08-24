import argparse
import os
from transformers import AutoTokenizer
from vllm_server.vllm_completion import VLLMCompletion
import openai, json
from utils.common import save_json, save_json_dict_format
from datasets import load_dataset
import re
from prompts import get_prompt
import torch.distributed as dist
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

def extract_key_sentences(response):
    pattern = r'"([^"]*)"'
    sentences = re.findall(pattern,response)
    joint_sentence = " ".join(sentences)
    return joint_sentence

def extract_answer(response):
    return response.split("</think>\n")[-1]

def extract_expansions(response_list):
    return [extract_answer(response) for response in response_list]


def get_scores(query_ids,doc_ids,scores,excluded_ids, return_full_scores=False, num_hits=1000):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        if return_full_scores:
            cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)
        else:
            cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:num_hits]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

import pytrec_eval
def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    # oracle reranker evaluation
    sorted_ids = {}
    top_100_ids = {}
    for query_id in results.keys():
        sorted_ids[query_id] = sorted(results[query_id].keys(), key=lambda x: results[query_id][x], reverse=True)
        top_100_ids[query_id] = set(sorted_ids[query_id][:100])
    oracle_results = {}
    for query_id in results.keys():
        oracle_results[query_id] = {}
        for doc_id in results[query_id].keys():
            if doc_id in top_100_ids[query_id] and query_id in qrels and doc_id in qrels[query_id]: # a doc is both top 100 and also in ground truth
                oracle_results[query_id][doc_id] = qrels[query_id][doc_id] # extract the score from ground truth
            else:
                oracle_results[query_id][doc_id] = 0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    oracle_scores = evaluator.evaluate(oracle_results)
    oracle_ndcg = {}
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = 0.0
    for query_id in oracle_scores.keys():
        for k in k_values:
            oracle_ndcg[f"Oracle NDCG@{k}"] += oracle_scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = round(oracle_ndcg[f"Oracle NDCG@{k}"] / len(oracle_scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg}

    return output


class Qwen3EmbeddingModel:
    def __init__(self, model_path, device="auto"):
        # if device == "auto":
        self.model = AutoModel.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.task = "Given a web search query, retrieve relevant passages that answer the query"

    def encode_query(self, query: str, max_length=8192, bs=2):
        return self.encode(self.get_detailed_instruct(self.task, query), max_length)[0]
    
    def encode_doc(self, doc, max_length=16384):
        return self.encode(doc, max_length)[0]

    def encode_docs(self, docs, max_length=16384, bs=1):
        embeddings = []
        for i in trange(0, len(docs), bs):
            embeddings.append(self.encode(docs[i:i+bs], max_length))
        return np.vstack(embeddings)

    def encode(self, text: list, max_length: int =16384):
        # Tokenize the input texts
        text = [text] if isinstance(text, str) else text
        batch_dict = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().detach().numpy()

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'



class VectorSearchInterface(object):
    def __init__(self, args, doc_ids:list, documents:list):
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.model = Qwen3EmbeddingModel(self.model_path)
        self.cache_dir = args.cache_dir
        self.task = args.task
        self.doc_ids = doc_ids
        self.documents = documents
        self.docs_emb = self.get_docs_emb(documents)

    def truncate_text(self, doc_text, max_tokens=100):
        doc_ids = self.model.tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_ids) > max_tokens:
            doc_ids = doc_ids[:max_tokens]
        return self.model.tokenizer.decode(doc_ids, skip_special_tokens=True)
    
    def get_docs_emb(self, documents):
        # cache docs emb
        cache_path = os.path.join(self.cache_dir, 'doc_emb', self.model_name, self.task, f"long_False") 
        os.makedirs(cache_path, exist_ok=True)
        doc_cache_file = os.path.join(cache_path, '0.npy')

        print('Encoding documents to cache:', cache_path)
        if os.path.exists(doc_cache_file):
            docs_emb = np.load(doc_cache_file, allow_pickle=True)
        else:
            with torch.inference_mode():
                docs_emb = self.model.encode_docs(documents)
            torch.cuda.empty_cache()
            np.save(doc_cache_file, docs_emb)
            print("Shape of doc emb", docs_emb.shape)
            
        return docs_emb

    torch.no_grad()
    def do_retrieval(self, qid, query_text, excluded_ids, num_hits=1000):
        '''
        return: dict of dict {qid: {doc_id: score}, }
        '''
        query_texts = [query_text] if isinstance(query_text, str) else query_text
        qid = [qid] if isinstance(qid, str) else qid
        
        query_emb = []
        with torch.inference_mode():
            for q in query_texts:
                query_emb.append(self.model.encode_query(q))
        query_emb = np.array(query_emb)
        print("Shape of query emb", query_emb.shape)
        torch.cuda.empty_cache()

        scores = cosine_similarity(query_emb, self.docs_emb).tolist()

        qid_doc_scores = get_scores(query_ids=qid, doc_ids=self.doc_ids, scores=scores, excluded_ids=excluded_ids, num_hits=num_hits)
        
        return qid_doc_scores


def search_iterator(args, search_api, qid_query_list, excluded_ids):
    for q_id, q_text in qid_query_list:
        id_doc_scores = search_api.do_retrieval(q_id, q_text, excluded_ids, num_hits=args.num_hits)
    
        for qid, docs_score in id_doc_scores.items():
            yield qid, docs_score


def progressive_query_rewrite(
        openai_api, cur_query, top_passages, iter_round,
        accumulated_query_expansions=[],
        max_demo_len=None,
        expansion_method="", 
        # accumulate=False,
        topic_id=None, search_api=None,
        *arg, **kwargs):
    
    if max_demo_len:
        top_passages = [search_api.truncate_text(psg, max_demo_len) for psg in top_passages]

    top_passages_str = "\n".join([f"[{idx+1}]. {psg}" for idx, psg in enumerate(top_passages)])

    if iter_round > 1:
        user_prompt = get_prompt(expansion_method, cur_query, top_passages_str, accumulated_query_expansions[topic_id][-1])
    else:
        user_prompt = get_prompt("thinkqe_revise_0", cur_query, top_passages_str)

    messages = [{"role": "user", "content": user_prompt}]

    # print("Input message:" + user_prompt)
    gen_fn = openai_api.completion_chat
    response_list = gen_fn(messages, *arg, **kwargs)
    query_expansions = extract_expansions(response_list)

    accumulated_query_expansions[topic_id].extend(query_expansions)
    query = cur_query + "\n\n" + "\n".join(query_expansions)

    return query, response_list, accumulated_query_expansions

def main():
    parser = argparse.ArgumentParser(description='DIVER-QExpand.')
    parser.add_argument('--model_path', type=str, default='./qwen-4b-embedding')
    parser.add_argument('--model_name', type=str, default='qwen3_4b')
    parser.add_argument('--dataset_source', type=str, default='../data/BRIGHT')
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--document_expansion', type=str, default=None)
    
    parser.add_argument('--num_hits', type=int, default=1000, help="Number of hits, the number of filtered docs inthe first stage.")
    parser.add_argument('--output', type=str, metavar='path',
                        help="Path to output file.")
    parser.add_argument('--max-passage', action='store_true',
                        default=False, help="Select only max passage from document.")
    parser.add_argument('--threads', type=int, metavar='num', required=False,
                        default=1, help="Maximum number of threads to use.")
    parser.add_argument('--tokenizer', type=str, help='tokenizer used to preprocess topics')
    parser.add_argument('--remove-duplicates', action='store_true', default=False, help="Remove duplicate docs.")
    # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
    # We want to remove the query from the results. This is equivalent to -removeQuery in Java.
    parser.add_argument('--remove_query', action='store_true', default=False, help="Remove query from results list.")

    parser.add_argument("--output_dir", type=str, required=True, help="Path to save outputs")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite existing output dir")
    # parser.add_argument("--openai_api_key", type=str, default="none")
    parser.add_argument("--generation_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--keep_passage_num", type=int, default=10, help="Number of passages kept for CSQE")
    parser.add_argument('--write_top_passages', action='store_true', help="Save the top retrieved passages")
    parser.add_argument("--gen_num", type=int, default=5, help="Number of query expansions to generate in each llm calling")
    parser.add_argument('--max_demo_len', type=int, default=None, help="Truncation length for each retrieved passage")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens to generate each time")
    parser.add_argument('--expansion_method', type=str, default="thinkqe_revise")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for query expansion")
    parser.add_argument('--reqeat_weight', type=float, default=3, help="Weight for query repetition of MUGI.")
    parser.add_argument('--accumulate', type=lambda x: x.lower() == 'true', default=False, help="Accumulate query expansions")
    parser.add_argument('--use_passage_filter', type=lambda x: x.lower() == 'true', default=False, help="Use filter for dropping previous seen passages")
    parser.add_argument('--no_thinking', type=lambda x: x.lower() == 'true', default=False, help="No thinking mode for the R1-distill-qwen model")
    parser.add_argument('--num_interaction', type=int, default=3, help="Number of interaction rounds with the corpus")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # Create output directory if needed

    # load dataset
    examples = load_dataset("parquet", data_files=os.path.join(args.dataset_source, f"examples/{args.task}-00000-of-00001.parquet"))["train"]
    org_qid_query_list = [(data['id'], data['query']) for data in examples]
    print(f"The number of query is {len(org_qid_query_list)}")

    excluded_ids = {}
    for qid, e in enumerate(examples):
        excluded_ids[e['id']] = e['excluded_ids']
        overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
        assert len(overlap)==0

    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e['gold_ids']:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in ground_truth[e['id']]
    # load documents
    docs_path = os.path.join(args.dataset_source, 'documents', f'{args.task}-00000-of-00001.parquet')
    doc_pairs = load_dataset("parquet", data_files=docs_path, cache_dir=args.cache_dir)["train"]

    doc_ids = []
    documents = []
    did2content = {}
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])
        did2content[dp['id']] = dp['content']
    search_api = VectorSearchInterface(args, doc_ids, documents)

    # generation model
    openai_api = VLLMCompletion(model_name=args.generation_model, control_thinking=args.no_thinking)

    # round 1
    last_top_passages = dict()  # qid2passages
    iter_org_qid_query_list = org_qid_query_list

    flag_gen_aug = False
    accumulated_query_expansions = {}
    # Add a dictionary to track seen passages for each query
    seen_passages = {}
    last_top_k_passages = {}  # Track passages from the previous round
    last_last_top_k_passages = {}  # Track passages from two rounds ago

    for index, (topic_id, query) in enumerate(tqdm(org_qid_query_list, total=len(org_qid_query_list))):
        accumulated_query_expansions[topic_id] = []
        # Initialize seen passages for each query
        seen_passages[topic_id] = set()
        last_top_k_passages[topic_id] = set()
        last_last_top_k_passages[topic_id] = set()
    
    round_num = args.num_interaction + 1
    for ridx in range(round_num):
        output_path = os.path.join(args.output_dir, f'diver_aug{ridx}_result_retrieval')
        if flag_gen_aug:
            # # data generation
            aug_qid_query_list = []
            qid2responses = dict()

            for index, (topic_id, query) in enumerate(tqdm(org_qid_query_list, total=len(org_qid_query_list))):
                # Get top passages but filter out previously seen ones
                all_passages = last_top_passages[topic_id]
                filtered_passages = []
                
                if args.use_passage_filter:
                    for passage in all_passages:
                        # Skip if already in our blacklist
                        if passage in seen_passages[topic_id]:
                            print(f"Already in blacklist: passage {passage}")
                            continue
                            
                        # If passage was in top results from two rounds ago, consider discarding it
                        if passage in last_last_top_k_passages[topic_id]:
                            seen_passages[topic_id].add(passage)  # Add to blacklist
                            print(f"Adding to blacklist: passage already seen in previous round.\n{passage} ")
                            continue
                            
                        # If we reach here, passage is new and not being discarded
                        filtered_passages.append(passage)
                        
                        # Break if we have enough new passages
                        if len(filtered_passages) >= args.keep_passage_num:
                            break
                    
                    # If we don't have enough new passages, we can use fewer
                    top_passages = filtered_passages[:args.keep_passage_num]
                else:
                    # If filtering is disabled, just use the top passages directly
                    top_passages = all_passages[:args.keep_passage_num]
                
                query_aug, response_list, accumulated_query_expansions = progressive_query_rewrite(
                    openai_api, query, top_passages, iter_round=ridx,
                    max_demo_len=args.max_demo_len,
                    expansion_method=args.expansion_method,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    accumulated_query_expansions=accumulated_query_expansions,
                    # accumulate=args.accumulate,
                    topic_id=topic_id, search_api=search_api
                )
                aug_qid_query_list.append((topic_id, query_aug))

                qid2responses[topic_id] = response_list

            # save_json(qid2responses, output_path + ".responses.json")
            iter_org_qid_query_list = aug_qid_query_list
    
        # save query
        save_json_dict_format(iter_org_qid_query_list, output_path + ".expand_query.json")

        all_scores = {}
        for qid, did_scores in tqdm(search_iterator(args, search_api, iter_org_qid_query_list, excluded_ids), 
                                    total=len(iter_org_qid_query_list),
                                    leave=True, dynamic_ncols=True):
            all_scores[qid] = did_scores
            # save passage result
            last_top_passages[qid] = [did2content[did] for did, score in did_scores.items()]

            # Update passage history at the end of each round
            if flag_gen_aug:
                # Move previous round's top passages to two rounds ago
                last_last_top_k_passages[qid] = last_top_k_passages[qid]
                # Store current round's top passages
                last_top_k_passages[qid] = set(last_top_passages[qid][:args.keep_passage_num])
            else:
                # For the first round, just store the top passages
                last_top_k_passages[qid] = set(last_top_passages[qid][:args.keep_passage_num])

        # if args.write_top_passages:
        #     save_json(last_top_passages, output_path + "top-psgs.json")

        result_metrics = calculate_retrieval_metrics(results=all_scores, qrels=ground_truth)
        save_json(result_metrics, output_path + ".metrics.json")
        print(f"At round {ridx}: {json.dumps(result_metrics)}")

        flag_gen_aug = True

if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

