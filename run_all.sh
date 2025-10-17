# 0.1 Download BRIGHT dataset
cd Diver
git clone https://huggingface.co/datasets/xlangai/BRIGHT ./data/BRIGHT
# or modelscope download --dataset xlangai/BRIGHT --local_dir ./data/BRIGHT  # more faster in China

# 0.2 Download models
mkdir models && cd models
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  # for DIVER-QExpand
git clone https://huggingface.co/AQ-MedAI/Diver-Retriever-4B  # for DIVER-Qexpand and DIVER-Retriever
cd ..

# 1. DIVER-QExpand
cd ./QExpand
bash run_qexpand.sh

# 2. DIVER-Retriever, achiving 33.9 NDCG@10, as reported in Table 3 of our paper: https://arxiv.org/pdf/2508.07995
cd ../Retriever
bash retriever_script.sh
# merge BM25 and DIVER-Retriever scores, achiving 37.2 NDCG@10 in Table 3 of our paper
python merge_score.py

# 3. DIVER-Reranker (v1 version with only pointwise reranker), achieving 41.6 NDCG@10 as shown in Table 2 of our paper
# cd ./Retriever
bash reranker_script.sh  
