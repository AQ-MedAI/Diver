# 0. Download BRIGHT dataset
git clone https://huggingface.co/datasets/xlangai/BRIGHT ./data/BRIGHT
# or modelscope download --dataset xlangai/BRIGHT --local_dir ./data/BRIGHT  # more faster in China

# 1. DIVER-Qexpand
cd ./QExpand
bash run_qexpand.sh

# 2. DIVER-Retriever
cd ../Retriever
bash retriever_script.sh
# merge BM25 and DIVER-Retriever scores
python merge_score.py

# 3. DIVER-Reranker
# cd ./Retriever
bash reranker_script.sh
