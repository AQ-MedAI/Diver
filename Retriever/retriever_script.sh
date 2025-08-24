#! /bin/bash

tasks=(biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions)
# tasks=(biology earth_science economics)

models=(diver-retriever bm25)  #  qwen3_4b_5kep_med_rader, reasonir
REASONING=diver-qexpand
BS=-1

for MODEL in ${models[@]}; do
    for TASK in ${tasks[@]}; do
        echo "Running task: $task"
        # original query
        # CUDA_VISIBLE_DEVICES=3 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache_${MODEL} --encode_batch_size $BS 

        # using expanded query
        CUDA_VISIBLE_DEVICES=0 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS 
        
        # # using expanded query and rechunk
        # CUDA_VISIBLE_DEVICES=0 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS --document_expansion rechunk
    done
done    