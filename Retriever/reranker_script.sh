#! /bin/bash

tasks=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions")  

MODEL=diver-retriever
REASONING=diver-qexpand 
llm_model="../models/Qwen2.5-32B-Instruct"

for task in "${tasks[@]}"; do
    echo "Running task: $task"
    # 1.1 pointwise reranker
    CUDA_VISIBLE_DEVICES=0 python rerank_pointwise.py \
    --task "$task" \
    --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" \
    --input_k 100 \
    --k 100 \
    --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_point/${task}_merge_bm25_${MODEL}" \
    --llm_model ${llm_model}
    
    # 1.2 listwise reranker
    # replace with actual base URL, API key and llm name
    CUDA_VISIBLE_DEVICES=0 python rerank_listwise.py \
        --task "$task" \
        --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" \
        --input_k 100 \
        --k 100 \
        --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_list/${task}_merge_bm25_${MODEL}" \
        --base_url "https://generativelanguage.googleapis.com/v1beta/openai/" \
        --api_key "API_KEY" \
        --llm_model "gemini-2.5-flash"

    # 2. merge two rerank results
    CUDA_VISIBLE_DEVICES=0 python rerank_merge_point_and_list.py \
    --task "$task" \
    --listwise_score_file "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_list/${task}_merge_bm25_${MODEL}/None_reranked_score.json" \
    --pointwise_score_file "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_point/${task}_merge_bm25_${MODEL}/None_reranked_score.json" \
    --output_dir "output/reranker/merge_point_list/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_merge_bm25_${MODEL}" \
    --input_k 100 \
    --k 100
done