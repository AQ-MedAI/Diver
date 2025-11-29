#! /bin/bash

tasks=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions")  

MODEL=diver-retriever
REASONING=diver-qexpand 
llm_model="../models/Qwen2.5-32B-Instruct"

# ------------------------------------------------------------------------------------------
# 1. v1 version: uses only pointwise reranker, achieving 41.6 NDCG@10 (as shown in Table 2 of our paper)
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
done

# ------------------------------------------------------------------------------------------
# # 2. v2 version: uses pointwise and listwise rerankers, achieving 45.8 NDCG@10 
# for task in "${tasks[@]}"; do
#     echo "Running task: $task"
#     # 2.1.1 pointwise reranker
#     CUDA_VISIBLE_DEVICES=0 python rerank_pointwise.py \
#     --task "$task" \
#     --model_name "../models/Qwen2.5-32B-Instruct" \
#     --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" \
#     --input_k 100 \
#     --k 100 \
#     --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_point/${task}_merge_bm25_${MODEL}" \
#     --llm_model ${llm_model}
    
#     # 2.1.2 listwise reranker
#     # replace with actual base URL, API key and llm name
#     CUDA_VISIBLE_DEVICES=0 python rerank_listwise.py \
#         --task "$task" \
#         --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" \
#         --input_k 100 \
#         --k 100 \
#         --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_list/${task}_merge_bm25_${MODEL}" \
#         --base_url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#         --api_key "API_KEY" \
#         --llm_model "gemini-2.5-flash"

#     # 2.2 merge two rerank results
#     CUDA_VISIBLE_DEVICES=0 python rerank_merge_point_and_list.py \
#     --task "$task" \
#     --listwise_score_file "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_list/${task}_merge_bm25_${MODEL}/None_reranked_score.json" \
#     --pointwise_score_file "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_point/${task}_merge_bm25_${MODEL}/None_reranked_score.json" \
#     --output_dir "output/reranker/merge_point_list/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_merge_bm25_${MODEL}" \
#     --input_k 100 \
#     --k 100
# done

# # ------------------------------------------------------------------------------------------
# # 3. v3 version: uses pointwise and our proposed groupwise rerankers, achieving 46.8 NDCG@10
# # 3.1 Training approach: Two-stage training process
# #   Stage 1: Supervised Fine-Tuning (SFT) for groupwise reranker
# #   Stage 2: Reinforcement Learning (RL) training (will be released in the future)
# bash train_sft_groupwise_reranker.sh  # SFT stage

# # 3.2 infer groupwise reranker
# EVAL_INTERVAL=20
# EVAL_GROUP_SIZE=20
# EVAL_REPEAT_NUM=6
# for task in "${tasks[@]}"; do
#     echo "Running task: $task"
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python group_rerank.py --task "$task" \
#         --model_path "../models/Diver-GroupRank-32B" \
#         --retriever_score_file "output/merge_${REASONING}_bm25_${REASONING}_${MODEL}/${task}_bm25_${MODEL}/${task}_merge_score.json" \
#         --output_dir "output/reranker/merge_${REASONING}_bm25_${REASONING}_${MODEL}_group/${task}_merge_bm25_${MODEL}" \
#         --group_size ${EVAL_GROUP_SIZE} \
#         --interval ${EVAL_INTERVAL} \
#         --repeat_num ${EVAL_REPEAT_NUM} \
#         --input_k 100 \
#         --k 100
# done