dataset_source="../data/BRIGHT"
# tasks=("biology" "earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa_theorems" "theoremqa_questions")
tasks=("biology")

model_path="../models/Diver-Retriever-4B".  # retriever model path
generation_model_path="../model/DeepSeek-R1-Distill-Qwen-14B"

model_name=diver-retriever
REASONING=diver-qexpand  # query expansion method
NUM_INTERACTION=2
KEEP_PASSAGE_NUM=5
GEN_NUM=1
USE_PASSAGE_FILTER=true
ACCUMULATE=true


for DATASET in "${tasks[@]}"; do
    echo "Running task: $DATASET"
    CUDA_VISIBLE_DEVICES=0,1 python3 qexpand_main.py \
        --model_path  ${model_path}\
        --model_name ${model_name} \
        --dataset_source ${dataset_source} \
        --task ${DATASET} \
        --cache_dir ${dataset_source}/cache/cache_${model_name} \
        --expansion_method thinkqe_revise \
        --threads 16 \
        --max_demo_len 512 \
        --generation_model ${generation_model_path} \
        --keep_passage_num ${KEEP_PASSAGE_NUM} \
        --gen_num ${GEN_NUM} \
        --use_passage_filter ${USE_PASSAGE_FILTER} \
        --output_dir  ${dataset_source}/output/${REASONING}/${DATASET} \
        --overwrite_output_dir \
        --temperature 0.7 \
        --write_top_passages \
        --num_interaction ${NUM_INTERACTION} \
        --accumulate ${ACCUMULATE}
done
