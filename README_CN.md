# DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval
虽然检索增强生成（RAG）在直接知识检索方面表现出色，但在需要抽象或多步推理的复杂查询面前却表现一般。为了弥补这一不足，我们开发了**DIVER**，这是一种专门针对这些推理密集型任务设计的检索管道。**DIVER**集成了四个阶段：文档预处理、迭代的LLM 驱动查询扩展、在复杂合成数据上进行微调的专用检索器，以及一种将listwise与pointwise相结合的新型重排序器。在[BRIGHT基准测试](https://brightbenchmark.github.io/)中，DIVER创造了新的最佳成绩，显著优于其他推理感知模型（NDCG 45.8）。
这些结果突显了将深度推理融入检索以解决复杂现实世界问题的有效性。更多详情请参阅[Diver论文](https://arxiv.org/pdf/2508.07995)。


## 🎉 更新列表

- [2025-10-20] 🚀 我们在 [ModelScope](https://www.modelscope.cn/models/AQ-MedAI/Diver-Retriever-4B-1020)和[Hugging Face](https://huggingface.co/AQ-MedAI/Diver-Retriever-4B-1020)上发布了 DIVER-Retriever-4B-1020 模型，在 BRIGHT 基准上取得了 31.9 的成绩。
- [2025-10-14] 🚀 我们在 [ModelScope](https://modelscope.cn/models/AQ-MedAI/Diver-Retriever-1.7B)和[Hugging Face](https://huggingface.co/AQ-MedAI/Diver-Retriever-1.7B)上发布了 DIVER-Retriever-1.7B 模型，在 BRIGHT 基准上取得了 27.3 的成绩。
- [2025-09-12] 🚀 我们发布了使用 Gemini 的 listwise 重排序代码；可以在 [./Retriever/rerank_listwise.py](./Retriever/rerank_listwise.py) 找到，并在 BRIGHT 上取得了 43.9 的得分。
- [2025-09-05] 🚀 我们在 [ModelScope](https://modelscope.cn/models/AQ-MedAI/Diver-Retriever-0.6B)和[Hugging Face](https://huggingface.co/AQ-MedAI/Diver-Retriever-0.6B)上发布了 DIVER-Retriever-0.6B 模型，在 BRIGHT 基准上取得了 25.2 的成绩。
- [2025-08-28] 🚀 我们在  [ModelScope](https://modelscope.cn/models/AQ-MedAI/Diver-Retriever-4B) 上发布了 DIVER-Retriever-4B 模型。
- [2025-08-24] 🏆 我们更新了Diver V2，在[Bright Leaderboard](https://brightbenchmark.github.io/)效果进一步提升至45.8。
- [2025-08-18] 🚀 我们开源了Diver的整体代码库包括推理和训练。

## 待办列表

- [ ] 开源 **DIVER-Reranker**：发布源码与模型


## 模型下载

您可以下载以下表格，以查看适用于您场景的各种参数。如果您位于中国大陆，我们还将在 ModelScope.cn 上提供该模型，以加快下载速度。

|      **Model**       | **#Total Params** | **Context Length** |                                                                        **Download**                                                                        |   **BRIGHT**       |
| :------------------: | :---------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------: | 
|    DIVER-Retriever-4B-1020    |       4B       |        40K         |     [🤗 HuggingFace]https://huggingface.co/AQ-MedAI/Diver-Retriever-4B-1020 <br>[🤖 ModelScope]https://www.modelscope.cn/models/AQ-MedAI/Diver-Retriever-4B-1020     | **31.9** |
|    DIVER-Retriever-4B    |       4B       |        40K         |     [🤗 HuggingFace]https://huggingface.co/AQ-MedAI/Diver-Retriever-4B <br>[🤖 ModelScope]https://www.modelscope.cn/models/AQ-MedAI/Diver-Retriever-4B     | **28.9** |
|    DIVER-Retriever-1.7B    |       1.7B       |        40K         |     [🤗 HuggingFace]https://huggingface.co/AQ-MedAI/Diver-Retriever-1.7B <br>[🤖 ModelScope]https://www.modelscope.cn/models/AQ-MedAI/Diver-Retriever-1.7B     | **27.3** |
|    DIVER-Retriever-0.6B    |       0.6B       |        32K         |     [🤗 HuggingFace]https://huggingface.co/AQ-MedAI/Diver-Retriever-0.6B <br>[🤖 ModelScope]https://www.modelscope.cn/models/AQ-MedAI/Diver-Retriever-0.6B     | **25.2** |



## 评估结果

### 整体评估结果
**Diver在 BRIGHT榜单上与其他基线的性能比较。每个数据集的最佳结果均以粗体突出显示。**

| Method | Avg. | Bio. | Earth. | Econ. | Psy. | Rob. | Stack. | Sus. | Leet. | Pony | AoPS | TheoQ. | TheoT. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Rank-R1-14B | 20.5 | 31.2 | 38.5 | 21.2 | 26.4 | 22.6 | 18.9 | 27.5 | 9.2 | 20.2 | 9.7 | 11.9 | 9.2 |
| Qwen1.5-7B with InteRank-3B | 27.4 | 51.2 | 51.4 | 22.4 | 31.9 | 17.3 | 26.6 | 22.4 | 24.5 | 23.1 | 13.5 | 19.3 | 25.5 |
| GPT4 with Rank1-32B | 29.4 | 49.7 | 35.8 | 22.0 | 37.5 | 22.5 | 21.7 | 35.0 | 18.8 | 32.5 | 10.8 | 22.9 | 43.7 |
| ReasonIR with QwenRerank | 36.9 | 58.2 | 53.2 | 32.0 | 43.6 | 28.8 | 37.6 | 36.0 | 33.2 | 34.8 | 7.9 | 32.6 | 45.0 |
| ReasonIR with Rank-R1-32B | 38.8 | 59.5 | 55.1 | 37.9 | 52.7 | 30.0 | 39.3 | 45.1 | 32.1 | 17.1 | 10.7 | 40.4 | 45.6 |
| RaDeR with QwenRerank | 39.2 | 58.0 | 59.2 | 33.0 | 49.4 | 31.8 | 39.0 | 36.4 | 33.5 | 33.3 | 10.8 | 34.2 | 51.6 |
| XRR2 | 40.3 | 63.1 | 55.4 | 38.5 | 52.9 | 37.1 | 38.2 | 44.6 | 21.9 | 35.0 | 15.7 | 34.4 | 46.2 |
| ReasonRank | 40.8 | 62.72 | 55.53 | 36.7 | 54.64 | 35.69 | 38.03 | 44.81 | 29.46 | 25.56 | 14.38 | 41.99 | 50.06 |
| **DIVER** | 41.6 | 62.2 | 58.7 | 34.4 | 52.9 | 35.6 | 36.5 | 42.9 | **38.9** | 25.4 | 18.3 | 40.0 | 53.1 |
| BGE Reasoner | 45.2 | 66.5 | **63.7** | 39.4 | 50.3 | 37 | 42.9 | 43.7 | 35.1 | **44.3** | 17.2 | 44.2 | **58.5** |
| **DIVER V2** | **45.8** | **68** | 62.5 | **42.0** | **58.2** | **41.5** | **44.3** | **49.2** | 34.8 | 32.9 | **19.1** | **44.3** | 52.6 |


### Diver检索器评估结果

<table>
<thead>
    <tr>
        <th>Method</th>
        <th style="text-align:right">Avg.</th>
        <th style="text-align:right">Bio.</th>
        <th style="text-align:right">Earth.</th>
        <th style="text-align:right">Econ.</th>
        <th style="text-align:right">Psy.</th>
        <th style="text-align:right">Rob.</th>
        <th style="text-align:right">Stack.</th>
        <th style="text-align:right">Sus.</th>
        <th style="text-align:right">Leet.</th>
        <th style="text-align:right">Pony</th>
        <th style="text-align:right">AoPS</th>
        <th style="text-align:right">TheoQ.</th>
        <th style="text-align:right">TheoT.</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td colspan=12 style="text-align:center"><strong>Evaluate Retriever with Original Query</strong></td>
    </tr>
    <tr>
        <td>BM25</td>
        <td style="text-align:right">14.5</td>
        <td style="text-align:right">18.9</td>
        <td style="text-align:right">27.2</td>
        <td style="text-align:right">14.9</td>
        <td style="text-align:right">12.5</td>
        <td style="text-align:right">13.6</td>
        <td style="text-align:right">18.4</td>
        <td style="text-align:right">15.0</td>
        <td style="text-align:right">24.4</td>
        <td style="text-align:right">7.9</td>
        <td style="text-align:right">6.2</td>
        <td style="text-align:right">10.4</td>
        <td style="text-align:right">4.9</td>
    </tr>
    <tr>
        <td>SBERT</td>
        <td style="text-align:right">14.9</td>
        <td style="text-align:right">15.1</td>
        <td style="text-align:right">20.4</td>
        <td style="text-align:right">16.6</td>
        <td style="text-align:right">22.7</td>
        <td style="text-align:right">8.2</td>
        <td style="text-align:right">11.0</td>
        <td style="text-align:right">15.3</td>
        <td style="text-align:right">26.4</td>
        <td style="text-align:right">7.0</td>
        <td style="text-align:right">5.3</td>
        <td style="text-align:right">20.0</td>
        <td style="text-align:right">10.8</td>
    </tr>
    <tr>
        <td>gte-Qwen1.5-7B</td>
        <td style="text-align:right">22.5</td>
        <td style="text-align:right">30.6</td>
        <td style="text-align:right">36.4</td>
        <td style="text-align:right">17.8</td>
        <td style="text-align:right">24.6</td>
        <td style="text-align:right">13.2</td>
        <td style="text-align:right">22.2</td>
        <td style="text-align:right">14.8</td>
        <td style="text-align:right">25.5</td>
        <td style="text-align:right">9.9</td>
        <td style="text-align:right">14.4</td>
        <td style="text-align:right">27.8</td>
        <td style="text-align:right">32.9</td>
    </tr>
    <tr>
        <td>Qwen3-4B</td>
        <td style="text-align:right">5.6</td>
        <td style="text-align:right">3.5</td>
        <td style="text-align:right">8.0</td>
        <td style="text-align:right">2.3</td>
        <td style="text-align:right">2.0</td>
        <td style="text-align:right">1.6</td>
        <td style="text-align:right">1.0</td>
        <td style="text-align:right">4.4</td>
        <td style="text-align:right">2.1</td>
        <td style="text-align:right">0.1</td>
        <td style="text-align:right">4.9</td>
        <td style="text-align:right">18.0</td>
        <td style="text-align:right">19.2</td>
    </tr>
    <tr>
        <td>OpenAI</td>
        <td style="text-align:right">17.9</td>
        <td style="text-align:right">23.3</td>
        <td style="text-align:right">26.7</td>
        <td style="text-align:right">19.5</td>
        <td style="text-align:right">27.6</td>
        <td style="text-align:right">12.8</td>
        <td style="text-align:right">14.3</td>
        <td style="text-align:right">20.5</td>
        <td style="text-align:right">23.6</td>
        <td style="text-align:right">2.4</td>
        <td style="text-align:right">8.5</td>
        <td style="text-align:right">23.5</td>
        <td style="text-align:right">11.7</td>
    </tr>
    <tr>
        <td>Google</td>
        <td style="text-align:right">20.0</td>
        <td style="text-align:right">22.7</td>
        <td style="text-align:right">34.8</td>
        <td style="text-align:right">19.6</td>
        <td style="text-align:right">27.8</td>
        <td style="text-align:right">15.7</td>
        <td style="text-align:right">20.1</td>
        <td style="text-align:right">17.1</td>
        <td style="text-align:right">29.6</td>
        <td style="text-align:right">3.6</td>
        <td style="text-align:right">9.3</td>
        <td style="text-align:right">23.8</td>
        <td style="text-align:right">15.9</td>
    </tr>
    <tr>
        <td>ReasonIR-8B</td>
        <td style="text-align:right">24.4</td>
        <td style="text-align:right">26.2</td>
        <td style="text-align:right">31.4</td>
        <td style="text-align:right">23.3</td>
        <td style="text-align:right">30.0</td>
        <td style="text-align:right">18.0</td>
        <td style="text-align:right"><strong>23.9</strong></td>
        <td style="text-align:right">20.5</td>
        <td style="text-align:right">35.0</td>
        <td style="text-align:right">10.5</td>
        <td style="text-align:right"><strong>14.7</strong></td>
        <td style="text-align:right">31.9</td>
        <td style="text-align:right">27.2</td>
    </tr>
    <tr>
        <td>RaDeR-7B</td>
        <td style="text-align:right">25.5</td>
        <td style="text-align:right">34.6</td>
        <td style="text-align:right">38.9</td>
        <td style="text-align:right">22.1</td>
        <td style="text-align:right">33.0</td>
        <td style="text-align:right">14.8</td>
        <td style="text-align:right">22.5</td>
        <td style="text-align:right">23.7</td>
        <td style="text-align:right">37.3</td>
        <td style="text-align:right">5.0</td>
        <td style="text-align:right">10.2</td>
        <td style="text-align:right">28.4</td>
        <td style="text-align:right">35.1</td>
    </tr>
    <tr>
        <td>Seed1.5-Embedding</td>
        <td style="text-align:right">27.2</td>
        <td style="text-align:right">34.8</td>
        <td style="text-align:right"><strong>46.9</strong></td>
        <td style="text-align:right"><strong>23.4</strong></td>
        <td style="text-align:right">31.6</td>
        <td style="text-align:right">19.1</td>
        <td style="text-align:right">25.4</td>
        <td style="text-align:right">21.0</td>
        <td style="text-align:right"><strong>43.2</strong></td>
        <td style="text-align:right">4.9</td>
        <td style="text-align:right">12.2</td>
        <td style="text-align:right">33.3</td>
        <td style="text-align:right">30.5</td>
    </tr>
    <tr>
        <td>DIVER-Retriever-0.6B</td>
        <td style="text-align:right">25.2</td>
        <td style="text-align:right">36.4</td>
        <td style="text-align:right">41.9</td>
        <td style="text-align:right">29.0</td>
        <td style="text-align:right">31.0</td>
        <td style="text-align:right">21.2</td>
        <td style="text-align:right">24.6</td>
        <td style="text-align:right">23.2</td>
        <td style="text-align:right">15.6</td>
        <td style="text-align:right">6.8</td>
        <td style="text-align:right">8.4</td>
        <td style="text-align:right">33.2</td>
        <td style="text-align:right">31.7</td>
    </tr>
    <tr>
        <td>DIVER-Retriever-4B</td>
        <td style="text-align:right"><strong>28.9</strong></td>
        <td style="text-align:right"><strong>41.8</strong></td>
        <td style="text-align:right">43.7</td>
        <td style="text-align:right">21.7</td>
        <td style="text-align:right"><strong>35.3</strong></td>
        <td style="text-align:right"><strong>21.0</strong></td>
        <td style="text-align:right">21.2</td>
        <td style="text-align:right"><strong>25.1</strong></td>
        <td style="text-align:right">37.6</td>
        <td style="text-align:right"><strong>13.2</strong></td>
        <td style="text-align:right">10.7</td>
        <td style="text-align:right"><strong>38.4</strong></td>
        <td style="text-align:right"><strong>37.3</strong></td>
    </tr>
    <tr>
        <td colspan=12 style="text-align:center"><strong>Evaluate Retriever with GPT-4 REASON-query</strong></td>
    </tr>
    <tr>
        <td>BM25</td>
        <td style="text-align:right">27.0</td>
        <td style="text-align:right"><strong>53.6</strong></td>
        <td style="text-align:right"><strong>54.1</strong></td>
        <td style="text-align:right">24.3</td>
        <td style="text-align:right">38.7</td>
        <td style="text-align:right">18.9</td>
        <td style="text-align:right">27.7</td>
        <td style="text-align:right">26.3</td>
        <td style="text-align:right">19.3</td>
        <td style="text-align:right">17.6</td>
        <td style="text-align:right">3.9</td>
        <td style="text-align:right">19.2</td>
        <td style="text-align:right">20.8</td>
    </tr>
    <tr>
        <td>SBERT</td>
        <td style="text-align:right">17.8</td>
        <td style="text-align:right">18.5</td>
        <td style="text-align:right">26.3</td>
        <td style="text-align:right">17.5</td>
        <td style="text-align:right">27.2</td>
        <td style="text-align:right">8.8</td>
        <td style="text-align:right">11.8</td>
        <td style="text-align:right">17.5</td>
        <td style="text-align:right">24.3</td>
        <td style="text-align:right">10.3</td>
        <td style="text-align:right">5.0</td>
        <td style="text-align:right">22.3</td>
        <td style="text-align:right">23.5</td>
    </tr>
    <tr>
        <td>gte-Qwen1.5-7B</td>
        <td style="text-align:right">24.8</td>
        <td style="text-align:right">35.5</td>
        <td style="text-align:right">43.1</td>
        <td style="text-align:right">24.3</td>
        <td style="text-align:right">34.3</td>
        <td style="text-align:right">15.4</td>
        <td style="text-align:right">22.9</td>
        <td style="text-align:right">23.9</td>
        <td style="text-align:right">25.4</td>
        <td style="text-align:right">5.2</td>
        <td style="text-align:right">4.6</td>
        <td style="text-align:right">28.7</td>
        <td style="text-align:right">34.6</td>
    </tr>
    <tr>
        <td>Qwen3-4B</td>
        <td style="text-align:right">5.5</td>
        <td style="text-align:right">1.3</td>
        <td style="text-align:right">17.3</td>
        <td style="text-align:right">2.5</td>
        <td style="text-align:right">6.2</td>
        <td style="text-align:right">1.0</td>
        <td style="text-align:right">4.8</td>
        <td style="text-align:right">4.5</td>
        <td style="text-align:right">3.0</td>
        <td style="text-align:right">5.9</td>
        <td style="text-align:right">0.0</td>
        <td style="text-align:right">7.2</td>
        <td style="text-align:right">12.5</td>
    </tr>
    <tr>
        <td>OpenAI</td>
        <td style="text-align:right">23.3</td>
        <td style="text-align:right">35.2</td>
        <td style="text-align:right">40.1</td>
        <td style="text-align:right">25.1</td>
        <td style="text-align:right">38.0</td>
        <td style="text-align:right">13.6</td>
        <td style="text-align:right">18.2</td>
        <td style="text-align:right">24.2</td>
        <td style="text-align:right">24.5</td>
        <td style="text-align:right">6.5</td>
        <td style="text-align:right">7.7</td>
        <td style="text-align:right">22.9</td>
        <td style="text-align:right">23.8</td>
    </tr>
    <tr>
        <td>Google</td>
        <td style="text-align:right">26.2</td>
        <td style="text-align:right">36.4</td>
        <td style="text-align:right">45.6</td>
        <td style="text-align:right">25.6</td>
        <td style="text-align:right">38.2</td>
        <td style="text-align:right">18.7</td>
        <td style="text-align:right"><strong>29.5</strong></td>
        <td style="text-align:right">17.9</td>
        <td style="text-align:right">31.1</td>
        <td style="text-align:right">3.7</td>
        <td style="text-align:right">10.0</td>
        <td style="text-align:right">27.8</td>
        <td style="text-align:right">30.4</td>
    </tr>
    <tr>
        <td>ReasonIR-8B</td>
        <td style="text-align:right">29.9</td>
        <td style="text-align:right">43.6</td>
        <td style="text-align:right">42.9</td>
        <td style="text-align:right"><strong>32.7</strong></td>
        <td style="text-align:right">38.8</td>
        <td style="text-align:right">20.9</td>
        <td style="text-align:right">25.8</td>
        <td style="text-align:right"><strong>27.5</strong></td>
        <td style="text-align:right">31.5</td>
        <td style="text-align:right"><strong>19.6</strong></td>
        <td style="text-align:right">7.4</td>
        <td style="text-align:right">33.1</td>
        <td style="text-align:right">35.7</td>
    </tr>
    <tr>
        <td>RaDeR-7B</td>
        <td style="text-align:right">29.2</td>
        <td style="text-align:right">36.1</td>
        <td style="text-align:right">42.9</td>
        <td style="text-align:right">25.2</td>
        <td style="text-align:right">37.9</td>
        <td style="text-align:right">16.6</td>
        <td style="text-align:right">27.4</td>
        <td style="text-align:right">25.0</td>
        <td style="text-align:right"><strong>34.8</strong></td>
        <td style="text-align:right">11.9</td>
        <td style="text-align:right"><strong>12.0</strong></td>
        <td style="text-align:right">37.7</td>
        <td style="text-align:right"><strong>43.4</strong></td>
    </tr>
    <tr>
        <td>DIVER-Retriever-4B</td>
        <td style="text-align:right"><strong>32.1</strong></td>
        <td style="text-align:right">51.9</td>
        <td style="text-align:right">53.5</td>
        <td style="text-align:right">29.5</td>
        <td style="text-align:right"><strong>41.2</strong></td>
        <td style="text-align:right"><strong>21.4</strong></td>
        <td style="text-align:right">27.5</td>
        <td style="text-align:right">26.1</td>
        <td style="text-align:right">33.5</td>
        <td style="text-align:right">11.7</td>
        <td style="text-align:right">9.5</td>
        <td style="text-align:right"><strong>39.3</strong></td>
        <td style="text-align:right">39.7</td>
    </tr>
    <tr>
        <td colspan=12 style="text-align:center"><strong>Evaluate retriever with DIVER-QExpand query</strong></td>
    </tr>
    <tr>
        <td>ReasonIR-8B</td>
        <td style="text-align:right">32.6</td>
        <td style="text-align:right">49.4</td>
        <td style="text-align:right">44.7</td>
        <td style="text-align:right">32.4</td>
        <td style="text-align:right">44.0</td>
        <td style="text-align:right">26.6</td>
        <td style="text-align:right">31.8</td>
        <td style="text-align:right">29.0</td>
        <td style="text-align:right">32.3</td>
        <td style="text-align:right">12.8</td>
        <td style="text-align:right">9.1</td>
        <td style="text-align:right"><strong>40.7</strong></td>
        <td style="text-align:right">38.4</td>
    </tr>
    <tr>
        <td>+BM25 (Hybrid)</td>
        <td style="text-align:right">35.7</td>
        <td style="text-align:right">56.8</td>
        <td style="text-align:right">53.5</td>
        <td style="text-align:right"><strong>33.0</strong></td>
        <td style="text-align:right"><strong>48.5</strong></td>
        <td style="text-align:right"><strong>29.4</strong></td>
        <td style="text-align:right"><strong>34.2</strong></td>
        <td style="text-align:right"><strong>32.0</strong></td>
        <td style="text-align:right"><strong>35.2</strong></td>
        <td style="text-align:right">16.8</td>
        <td style="text-align:right">12.9</td>
        <td style="text-align:right">39.3</td>
        <td style="text-align:right">36.8</td>
    </tr>
    <tr>
        <td>DIVER-Retriever</td>
        <td style="text-align:right"><strong>33.9</strong></td>
        <td style="text-align:right">54.5</td>
        <td style="text-align:right">52.7</td>
        <td style="text-align:right">28.8</td>
        <td style="text-align:right">44.9</td>
        <td style="text-align:right">25.1</td>
        <td style="text-align:right">27.4</td>
        <td style="text-align:right">29.5</td>
        <td style="text-align:right">34.5</td>
        <td style="text-align:right">10.0</td>
        <td style="text-align:right">14.5</td>
        <td style="text-align:right"><strong>40.7</strong></td>
        <td style="text-align:right">44.7</td>
    </tr>
    <tr>
        <td>+BM25 (Hybrid)</td>
        <td style="text-align:right"><strong>37.2</strong></td>
        <td style="text-align:right"><strong>60.0</strong></td>
        <td style="text-align:right"><strong>55.9</strong></td>
        <td style="text-align:right">31.8</td>
        <td style="text-align:right">47.9</td>
        <td style="text-align:right">27.1</td>
        <td style="text-align:right">33.9</td>
        <td style="text-align:right">31.9</td>
        <td style="text-align:right">35.1</td>
        <td style="text-align:right"><strong>23.1</strong></td>
        <td style="text-align:right"><strong>16.8</strong></td>
        <td style="text-align:right">36.9</td>
        <td style="text-align:right"><strong>46.6</strong></td>
    </tr>
    </tbody>
</table>




## 一键使用


### 推理 

#### 一键复现

```bash
sh run_all.sh

```

#### Retriever使用 （Hugging Face Transformers)
使用Sentence Transformers
```bash
# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("AQ-MedAI/Diver-Retriever-4B")


# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)

```


vLLM usage
```python
# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    get_detailed_instruct(task, 'Explain gravity')
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
input_texts = queries + documents

model = LLM(model="AQ-MedAI/Diver-Retriever-4B", task="embed")

outputs = model.embed(input_texts)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = (embeddings[:2] @ embeddings[2:].T)
```


## 训练

我们建议您使用 [swift](https://github.com/modelscope/ms-swift) 来对我们的 DIVER-Retriever-4B 进行微调。
在开始训练之前，请确保您的环境已正确配置好。

```bash
pip install ms-swift -U
# Install from source
pip install git+https://github.com/modelscope/ms-swift.git

pip install transformers -U

# Optional packages
pip install deepspeed # multi-GPU training
pip install liger-kernel # save GPU memory resources
pip install flash-attn --no-build-isolation
```

### 训练数据准备

```json
# LLM
{"query": "sentence1", "response":  "sentence2"}
# MLLM
{"query": "<image>", "response":  "sentence", "images": "/some/images.jpg"}
{"query": "<image>sentence1", "response":  "<image>sentence2", "rejected_response": ["<image>sentence1", "<image>sentence2"], "images": ["/some/images.jpg", "/some/images.jpg", "/some/images.jpg", "/some/images.jpg"]}
```

### 训练命令

以infonce loss为例，完整的训练指令如下：

```bash
nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model DIVER/DIVER-Retriever-4B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset your_dataset \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
```


## 引用

如果您觉得我们的工作有所帮助，请随时告知我们，我们会非常感激。

```
@misc{DIVER,
      title={DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval}, 
      author={Meixiu Long and Duolin Sun and Dan Yang and Junjie Wang and Yue Shen and Jian Wang and Peng Wei and Jinjie Gu and Jiahai Wang},
      year={2025},
      eprint={2508.07995},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.07995}, 
}
```


## 致谢

我们感谢之前的相关研究以及它们所发布的开源资源：[BRIGHT](https://github.com/xlang-ai/BRIGHT), [ReasonIR](https://github.com/facebookresearch/ReasonIR), [RaDer](https://anonymous.4open.science/r/project-D27D/README.md), [ThinkQE](https://github.com/Yibin-Lei/Think_QE), [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)。

## Star趋势

[![Star History Chart](https://api.star-history.com/svg?repos=AQ-MedAI/Diver&type=Date)](https://www.star-history.com/#AQ-MedAI/Diver&Date)