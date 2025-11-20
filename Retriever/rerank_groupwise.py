from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

random.seed(666)

sys_prompt = '''Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
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

user_prompt = '''I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
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



class GroupReranker:
    def __init__(self, model_path, sys_prompt, user_prompt) -> None:
        # vllm offline inference
        self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(), max_model_len=32000)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.group_system_prompt = sys_prompt
        self.group_user_prompt = user_prompt

    def rerank(self, query, doc_list):
        docs_str = ''.join(["[{}]. {}\n\n".format(idx+1, doc_text) for idx, doc_text in enumerate(doc_list)])

        group_texts = self.group_user_prompt.format(QUERY=query, PASSAGES=docs_str, TOPK=len(dids))

        message = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': self.group_system_prompt}, 
            {'role': 'user', 'content': group_texts}], tokenize=False, add_generation_prompt=True)

        output = self.llm.generate(message, self.sampling_params, use_tqdm=True)

        return output