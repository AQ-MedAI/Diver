import os.path
import time
import torch
import json
import numpy as np
import tiktoken
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from collections import defaultdict
from vllm import LLM


def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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


def retrieval_bm25(queries,query_ids,documents,doc_ids,excluded_ids,long_context,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    store_all_score = kwargs.get('store_all_scores', False)
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    repeat_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        repeat_scores[str(query_id)] = defaultdict(list)
        for did, s in zip(doc_ids, similarities):
            # all_scores[str(query_id)][did] = s
            # repeat_scores[str(query_id)][did].append(s)
            if did not in all_scores[str(query_id)] or s > all_scores[str(query_id)][did]:
                    all_scores[str(query_id)][did] = s  # refine docs, save the best score for each query-doc pair
                    
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        if store_all_score:
            cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)
        else:
            cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
        
    print("Dedup Scores shape", len(repeat_scores))

    return all_scores


def retrieval_reasonir(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # NOTE: HF version does not come with pooling function, need to add it manually.
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        # customized_checkpoint = 'reasonir/ReasonIR-8B'
        customized_checkpoint = '../model/reasonir__ReasonIR-8B'  # reasonir检索
    else:
        print('use',customized_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True)
    model = AutoModel.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True)
    model.eval()
    model.to(device)
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',32768)
    doc_max_length = kwargs.get('doc_max_length',32768)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)

    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    # if not os.path.isdir(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}")):
    #     os.makedirs(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    skip_doc_emb = kwargs.pop('skip_doc_emb',False)
    if not skip_doc_emb:
        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
        elif ignore_cache:
            inputs = tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)
            doc_emb = model(**inputs)[0]
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
        else:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
            np.save(cur_cache_file, doc_emb)
    # cur_cache_file = os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
    # save query embedding
    # np.save(cur_cache_file, query_emb)
    # if os.path.isfile(cur_cache_file):
    #     query_emb = np.load(cur_cache_file, allow_pickle=True)
    # elif ignore_cache:
    #     query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
    # else:
        # query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
        # np.save(cur_cache_file, query_emb)
    if skip_doc_emb:
        exit()
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_rader(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model_name = '../model/RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical'  # rader检索
    batch_size = kwargs.get('encode_batch_size',1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()
    # print("Model type", model)
    # max_length = kwargs.get('doc_max_length',4096)

    # Append instructions before queries 
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])

    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_name, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_name, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_name, task, f"long_{long_context}_{batch_size}", f'0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []

        for i in tqdm(range(0, len(documents))): #len(documents)
            text = documents[i]
            inputs = tokenizer(f"document: {text[:8192]}{tokenizer.eos_token}", return_tensors='pt', padding=True, truncation=True) 

            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                if i==0:
                    print("Doc outputs shape", outputs.last_hidden_state.shape)
                
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
            torch.cuda.empty_cache()  # 清理缓存
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)

    query_emb = []
    for i in tqdm(range(0, len(queries))):
        text = queries[i]
        inputs = tokenizer(f"query: {text}{tokenizer.eos_token}", return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # if model_name == "qwen2.5_7b_instruct":
            #     last_hidden = outputs.hidden_states[-1]  # Last layer
            #     embeddings = last_hidden.mean(dim=1)  #Mean pooling
            # else:
            embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings = embeddings.cpu().numpy()
            query_emb.extend(embeddings)
     
    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)
    print("First doc embedding:", doc_emb[0,:])

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)



'''vllm version'''
from vllm.transformers_utils.tokenizer import get_tokenizer as get_vllm_tokenizer
class Qwen3EmbeddingModel:
    def __init__(self, model_path, max_length=16384, device="auto"):
        self.model = LLM(model=model_path, task="embed", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count())
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        self.max_length = max_length 
        self.tokenizer = get_vllm_tokenizer(model_path, trust_remote_code=False)

    def truncate_text(self, text):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
            text = self.tokenizer.decode(text_ids)
        return text

    def embed_query(self, query):
        outputs = self.model.embed(query)
        return outputs[0].outputs.embedding

    def embed_queries(self, query):
        input_queries = ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}'.format(x) for x in query]
        input_queries = [self.truncate_text(x) for x in input_queries]
        outputs = self.model.embed(input_queries)
        return [x.outputs.embedding for x in outputs]

    def embed_doc(self, doc):
        outputs = self.model.embed("Represent this text:{}".format(doc))
        return outputs[0].outputs.embedding

    def embed_docs(self, docs):
        docs = ["Represent this text:{}".format(doc) for doc in docs]
        docs = [self.truncate_text(doc, ) for doc in docs]
        outputs = self.model.embed(docs)
        return [x.outputs.embedding for x in outputs]


@torch.no_grad()
def retrieval_qwen3_ft_diver(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    cache_model_name = kwargs.get('model_name', 'diver')
    batch_size = kwargs.get('encode_batch_size',1)

    model_path = '../models/Diver-Retriever-4B'
    model = Qwen3EmbeddingModel(model_path, max_length=16384)

    # Check if documents are already encoded 
    document_postfix = '_'+kwargs['document_postfix'] if len(kwargs['document_postfix']) > 0 else ''
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb'+document_postfix, cache_model_name, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, f'0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        with torch.inference_mode():
            doc_emb = model.embed_docs(documents)
        torch.cuda.empty_cache() 
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    print("Shape of doc emb", doc_emb.shape)

    query_emb = []
    with torch.inference_mode():
        query_emb = model.embed_queries(queries)
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()

    if len(kwargs['document_postfix']) > 0:  # rechunk setting
        dedup_doc_ids = set(doc_ids)
        dedup_scores = []  # shape:[len(scores), len(dedup_doc_ids)], save only the best score for each query-doc pair
        for query_idx in range(len(query_emb)):
            best_scores = {}  # for each query, save the best score for each doc_id
            for idx, score in enumerate(scores[query_idx]):
                doc_id = doc_ids[idx]
                if doc_id not in best_scores or score > best_scores[doc_id]:
                    best_scores[doc_id] = score
            q_doc_scores = []
            for doc_id in dedup_doc_ids:
                q_doc_scores.append(best_scores.get(doc_id))
            dedup_scores.append(q_doc_scores)

        doc_ids, scores = dedup_doc_ids, dedup_scores
        print("Dedup Scores shape:", len(scores[0]))
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)



RETRIEVAL_FUNCS = {
    'bm25': retrieval_bm25,
    'reasonir': retrieval_reasonir,
    'rader': retrieval_rader,
    'diver-retriever': retrieval_qwen3_ft_diver,
}