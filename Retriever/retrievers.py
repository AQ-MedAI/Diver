import os.path
import time
import torch
import json
import numpy as np
import pytrec_eval
import tiktoken
from tqdm import tqdm,trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from peft import PeftModel, PeftConfig
import pandas as pd
from collections import defaultdict
from vllm import LLM, PoolingParams, SamplingParams



def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def get_embedding_google(texts,task,model,dimensionality=768):
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]

def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    while not success:
        exec_count += 1
        if exec_count>5:
            print('execute too many times')
            exit(0)
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            count += 1
            threshold -= 500
            if count>4:
                print('openai cut',count)
                exit(0)
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

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

def get_bilevel_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    emb_scores = {}
    for query_id,doc_scores,query_doc_ids in zip(query_ids,scores,doc_ids):
        assert len(doc_scores)==len(query_doc_ids),f"{len(doc_scores)}, {len(query_doc_ids)}"
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(query_doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

def retrieval_sf_qwen_e5(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
        model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral',device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen1.5-7B-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen1.5-7B-instruct', device_map="auto",
                                          trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', device_map="auto",
                                          trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='rader':
        model_name = '../models/RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical'  # rader检索
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager").eval()
        max_length = kwargs.get('doc_max_length',4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    doc_emb = []
    batch_size = kwargs.get('encode_batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))

    with torch.inference_mode():
        for start_idx in trange(0,len(documents),batch_size):
            cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{start_idx}.json')
            # if os.path.isfile(cur_cache_file):
            #     continue
            try:
                with open(cur_cache_file) as f:
                    embeddings = json.load(f)
            except:
                batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
                with open(cur_cache_file,'w') as f:
                    json.dump(embeddings,f,indent=2)
            # doc_emb += embeddings

    query_emb = []
    with torch.inference_mode():
        for start_idx in trange(0, len(queries), batch_size):
            batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                                truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
            query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)

    all_scores = []
    chunk_size = 5000
    for i in trange(0,len(documents),chunk_size):
        doc_emb = []

        for start_idx in trange(i, min(i+chunk_size, len(documents)), batch_size):
            cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{start_idx}.json')
            if os.path.isfile(cur_cache_file):
                with open(cur_cache_file) as f:
                    embeddings = json.load(f)
                    doc_emb += embeddings
            else:
                raise ValueError(f"Missing file {cur_cache_file}")

        doc_emb = torch.tensor(doc_emb)
        doc_emb = doc_emb.reshape(doc_emb.shape[0],-1)
        print("doc_emb shape:",doc_emb.shape)
        doc_emb = F.normalize(doc_emb, p=2, dim=1)

        scores = (query_emb @ doc_emb.T) * 100
        all_scores.append(scores)
    all_scores = torch.cat(all_scores,dim=1).tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=all_scores,excluded_ids=excluded_ids)

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

def retrieval_sbert_bge(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='bge':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    elif model_id=='sbert':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    elif model_id=='qwen2':
        model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
    elif model_id=='contriever_st':
        model = SentenceTransformer('nishimoto/contriever-sentencetransformer')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_sbert_bge_ce(queries, query_ids, documents, doc_ids, task, instructions, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    if model_id == 'bge_ce':
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        model = AutoModel.from_pretrained('BAAI/bge-reranker-large', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length', 512)
        queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    else:
        raise ValueError(f"The model {model_id} is not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores = []
    for query in tqdm(queries):
        pairs_per_query = []
        for document in documents:
            pairs_per_query.append([query, document])

        batch_size = kwargs.get('batch_size',512)
        total_batches = len(pairs_per_query) // batch_size + (0 if len(pairs_per_query) % batch_size == 0 else 1)

        scores_per_query = []
        with torch.no_grad():
            for i in tqdm(range(total_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(pairs_per_query))
                batch = pairs_per_query[start_idx:end_idx]

                inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_scores = model(**inputs, return_dict=True).logits.view(-1).float()

                scores_per_query.extend(batch_scores.tolist())

        scores.append(scores_per_query)
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)
    
def retrieval_instructor(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # from InstructorEmbedding import INSTRUCTOR as Instructor
    if model_id=='inst-l':
        # model = Instructor('hkunlp/instructor-large')
        model = SentenceTransformer("hkunlp/instructor-large")
    elif model_id=='inst-xl':
        # model = Instructor('hkunlp/instructor-xl')
        model = SentenceTransformer("hkunlp/instructor-xl")
    else:
        raise ValueError(f"The model {model_id} is not supported")
    batch_size = kwargs.get('batch_size',4)
    model.max_seq_length = kwargs.get('doc_max_length',2048)
    queries = add_instruct_list(texts=queries,task=task,instruction=instructions['query'])
    documents = add_instruct_list(texts=documents,task=task,instruction=instructions['document'])
    query_embs = model.encode(queries,batch_size=batch_size,show_progress_bar=True)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_embs)
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_grit(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from gritlm import GritLM
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use',customized_checkpoint)
    model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding")
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',32768)
    doc_max_length = kwargs.get('doc_max_length',32768)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)
    # model_id = 'grit'
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    if not os.path.isdir(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    skip_doc_emb = kwargs.pop('skip_doc_emb',False)
    if not skip_doc_emb:
        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
        elif ignore_cache:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
        else:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
            np.save(cur_cache_file, doc_emb)
    cur_cache_file = os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    # save query embedding
    np.save(cur_cache_file, query_emb)
    if os.path.isfile(cur_cache_file):
        query_emb = np.load(cur_cache_file, allow_pickle=True)
    elif ignore_cache:
        query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    else:
        query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
        np.save(cur_cache_file, query_emb)
    if skip_doc_emb:
        exit()
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from openai import OpenAI
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d,tokenizer=tokenizer))
    documents = new_documents
    doc_emb = []
    batch_size = kwargs.get('batch_size',1024)
    openai_client = OpenAI(api_key=kwargs['key'])
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            cur_emb = get_embedding_openai(texts=documents[idx:idx + batch_size],openai_client=openai_client,tokenizer=tokenizer)
            with open(cur_cache_file,'w') as f:
                json.dump(cur_emb,f,indent=2)
        doc_emb += cur_emb
    query_emb = []
    for idx in trange(0, len(queries), batch_size):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client,
                                       tokenizer=tokenizer)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_cohere(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    import cohere
    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',8192)
    cohere_client = cohere.Client(kwargs['key'])
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            success = False
            exec_count = 0
            cur_emb = []
            while not success:
                exec_count += 1
                if exec_count>5:
                    print('cohere execute too many times')
                    exit(0)
                try:
                    cur_emb = cohere_client.embed(documents[idx:idx+batch_size], input_type="search_document",
                                                  model="embed-english-v3.0").embeddings

                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(60)
            with open(cur_cache_file, 'w') as f:
                json.dump(cur_emb, f, indent=2)
        doc_emb += cur_emb
    for idx in trange(0, len(queries), batch_size):
        success = False
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('cohere query execute too many times')
                exit(0)
            try:
                cur_emb = cohere_client.embed(queries[idx:idx+batch_size], input_type="search_query",
                                              model="embed-english-v3.0").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)
    scores = (torch.tensor(query_emb) @ torch.tensor(doc_emb).T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_voyage(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    import voyageai
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    new_queries = []
    for q in queries:
        new_queries.append(cut_text(text=q,tokenizer=tokenizer,threshold=16000))
    queries = new_queries
    new_documents = []
    for d in tqdm(documents,desc='preprocess documents'):
        new_documents.append(cut_text(text=d,tokenizer=tokenizer,threshold=16000))
    documents = new_documents

    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',1)
    voyage_client = voyageai.Client(api_key=kwargs['key'])
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for i in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{i}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            success = False
            threshold = 16000
            cur_texts = documents[i:i+batch_size]
            count_over = 0
            exec_count = 0
            while not success:
                exec_count += 1
                if exec_count > 5:
                    print('voyage document too many times')
                    exit(0)
                try:
                    cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="document").embeddings
                    with open(cur_cache_file,'w') as f:
                        json.dump(cur_emb,f,indent=2)
                    success = True
                except Exception as e:
                    print(e)
                    count_over += 1
                    threshold = threshold-500
                    if count_over>4:
                        print('voyage:',count_over)
                    new_texts = []
                    for t in cur_texts:
                        new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                    cur_texts = new_texts
                    time.sleep(5)
        doc_emb += cur_emb
    for i in trange(0,len(queries),batch_size):
        success = False
        threshold = 16000
        cur_texts = queries[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage query execute too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="query").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(60)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_google(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',8)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for start_idx in tqdm(range(0, len(documents), batch_size), desc='embedding'):
        cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'{start_idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            cur_emb = get_embedding_google(texts=documents[start_idx:start_idx + batch_size], task='RETRIEVAL_DOCUMENT',
                                           model=model)
            with open(cur_cache_file,'w') as f:
                json.dump(cur_emb,f,indent=2)
        doc_emb += cur_emb
    for start_idx in tqdm(range(0,len(queries), batch_size),desc='embedding'):
        query_emb += get_embedding_google(texts=queries[start_idx:start_idx+ batch_size],task='RETRIEVAL_QUERY',model=model)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_nomic(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    customized_checkpoint = kwargs.get('checkpoint', None)
    if customized_checkpoint is not None:
        model = SentenceTransformer(customized_checkpoint, trust_remote_code=True)
    else:
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    batch_size = kwargs.get('batch_size', 1)
    model_id = customized_checkpoint  # making sure the doc cache is reused
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_m2(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        "togethercomputer/m2-bert-80M-32k-retrieval",
        trust_remote_code=True
    )
    max_length = 32768

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        model_max_length=max_length
    )

    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    doc_emb = []
    batch_size = kwargs.get('encode_batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for start_idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{start_idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                embeddings = json.load(f)
        else:
            batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embeddings = outputs['sentence_embedding'].cpu().tolist()
            with open(cur_cache_file,'w') as f:
                json.dump(embeddings,f,indent=2)
        doc_emb += embeddings
    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = outputs['sentence_embedding'].cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_contriever(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    # model = AutoModel.from_pretrained('facebook/contriever')
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco')
    model = model.to('cuda')
    model.eval()

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(model, texts, show_progress_bar=True,batch_size=1, normalize_embeddings=True): # encode a batch of documents into the embeddings
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            # move inputs to cuda
            for k, v in inputs.items():
                inputs[k] = v.to('cuda')
            with torch.inference_mode():
                outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if normalize_embeddings:
            all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        all_embeddings = all_embeddings.cpu().numpy()
        return all_embeddings

    batch_size = kwargs.get('batch_size', 1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = encode(model, documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = encode(model, queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


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
        input_queries = [self.truncate_text(x) for x in query]
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
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    'inst-l': retrieval_instructor,
    'inst-xl': retrieval_instructor,
    'grit': retrieval_grit,
    'cohere': retrieval_cohere,
    'voyage': retrieval_voyage,
    'openai': retrieval_openai,
    'google': retrieval_google,
    'bge_ce': retrieval_sbert_bge_ce,
    'nomic': retrieval_nomic,
    'm2': retrieval_m2,
    'contriever': retrieval_contriever,
    'reasonir': retrieval_reasonir,
    'rader': retrieval_rader,
    'diver-retriever': retrieval_qwen3_ft_diver,
}

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
    print(output)

    return output


