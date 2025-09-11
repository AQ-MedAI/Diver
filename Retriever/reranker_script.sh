#! /bin/bash

tasks=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions")  

MODEL=qwen3_4b_5kep_med_rader
REASONING=thinkqe 

for task in "${tasks[@]}"; do
    echo "Running task: $task"
    # pointwise reranker
    CUDA_VISIBLE_DEVICES=0 python rerank_pointwise.py --task "$task" --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" --input_k 100 --k 100 --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_point/${task}_merge_bm25_${MODEL}" 
    # listwise reranker
    CUDA_VISIBLE_DEVICES=0 python rerank_listwise.py --task "$task" --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" --input_k 100 --k 100 --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_list/${task}_merge_bm25_${MODEL}" 

done