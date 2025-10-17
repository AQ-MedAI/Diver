#! /bin/bash

# BRIGHT datasets
tasks=(biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions)

models=(diver-retriever bm25)  # retriever name
REASONING=diver-qexpand  # query expansion method
BS=-1

for MODEL in ${models[@]}; do
    for TASK in ${tasks[@]}; do
        echo "Running task: $task"
        # # Using the original query achieves 28.9 NDCG@10, as reported in Table 3 of our paper: https://arxiv.org/pdf/2508.07995
        # CUDA_VISIBLE_DEVICES=3 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache_${MODEL} --encode_batch_size $BS 

        # Using expanded query achieves 33.9 NDCG@10, as reported in Table 3 of our paper
        CUDA_VISIBLE_DEVICES=0 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS 
        
        # # Using the expanded query and rechunk module achieves 37.5 NDCG@10 on average across the 7 general datasets in BRIGHT, as shown in Table 5 of our paper
        # CUDA_VISIBLE_DEVICES=0 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS --document_expansion rechunk
    done
done    