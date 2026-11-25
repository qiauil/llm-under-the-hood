# 参考资料

本页收录正文实际使用并核验的原论文、官方文档与官方实现。每条保留版本或访问入口、重点阅读范围及“不作本机结果”等证据边界；更细的核验位置与使用边界以来源台账为准。

[总目录](../index.qmd) · [符号表](notation.md) · [习题答案](exercise-solutions.md)

## 按章节与主题查找

主题标签只用于导航；同一资料可能支持多个章节。第 10 章以项目复用前述结构、训练、Kernel 和服务资料，不另造一套来源。

| 章节 | REF 区间 | 主题标签 | 入口 |
| --- | --- | --- | --- |
| 第 1 章 | REF-001–015 | 模型范式 · Tokenization · 自回归 · 采样 | [概述](#refs-1-1) · [分词](#refs-1-2) · [自回归](#refs-p1-batch) |
| 第 2 章 | REF-002、016–028 | Attention · GQA/MQA · RoPE · 归一化 · 门控 | [Attention 与位置](#refs-p1-batch) · [归一化与门控](#refs-norm-compute) |
| 第 3 章 | REF-029–041 | 参数/FLOPs · GPU · 存储层级 · Roofline · Profiler | [资源与 GPU](#refs-ch3) |
| 第 4 章 | REF-042–060 | 数据 · 优化器 · 混合精度 · 训练预算 · Checkpoint | [预训练流程](#refs-ch4) |
| 第 5 章 | REF-061–089 | Collective · DP/ZeRO/FSDP · TP/PP/SP/CP/EP · 重计算 | [分布式训练](#refs-ch5) |
| 第 6 章 | REF-090–101 | FlashAttention · CUDA · SDPA · 计时/显存 | [高性能 Attention](#refs-ch6) |
| 第 7 章 | REF-102–123 | 服务指标 · KV/PagedAttention · 调度 · PD分离 · 推测/量化 | [推理服务](#refs-ch7) |
| 第 8 章 | REF-124–153 | SFT/LoRA · 偏好优化 · 评测 · 安全/可靠性 | [后训练与评测](#refs-ch8) |
| 第 9 章 | REF-154–171 | 长上下文 · 多模态 · 测试时计算 · RAG/Agent · SSM | [前沿专题](#refs-ch9) |
| 第 10 章 | 复用 REF-001–153 | 项目证据 · 复现协议 · 系统验收 | 按项目主题返回第 1–8 章入口 |

<a id="refs-1-1"></a>

## 1.1 大语言模型概述

- **REF-001**：Mikolov, T., Karafiát, M., Burget, L., Černocký, J., & Khudanpur, S. (2010). *Recurrent neural network based language model*. Interspeech, 1045–1048。[会议原文 PDF](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.pdf)。阅读 §1–2，关注统计语言建模与递推状态。
- **REF-002**：Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS。核验版本 arXiv:1706.03762v7。[论文全文](https://arxiv.org/html/1706.03762v7)。阅读 §1、§3、§4 与表 1，关注信息路径、训练并行性和复杂度。
- **REF-003**：Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT。核验版本 arXiv:1810.04805v2。[论文全文](https://arxiv.org/html/1810.04805v2)。阅读 §3–3.1，关注双向编码器与遮蔽预测。
- **REF-004**：Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR, 21(140), 1–67。核验版本 arXiv:1910.10683v4。[论文全文](https://arxiv.org/html/1910.10683v4)。阅读 §2.1、§3.2，关注三类信息流及有条件的架构比较。
- **REF-005**：Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS。核验版本 arXiv:2005.14165v4。[论文全文](https://arxiv.org/html/2005.14165v4)。阅读 §2–2.1，关注自回归模型与不更新权重的上下文示例。
- **REF-006**：Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS。核验版本 arXiv:2203.02155v1。[论文全文](https://arxiv.org/html/2203.02155v1)。阅读摘要、§3.1 与“仍会犯错”的讨论，关注示范微调、比较数据与策略优化流程。

以上材料访问日期均为 2026-10-31；核验针对本节使用的论点，不代表已逐页复现论文实验。

<a id="refs-1-2"></a>

## 1.2 Tokenization

- **REF-007**：Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL, 1715–1725。[正式论文 PDF](https://aclanthology.org/P16-1162.pdf)。阅读 §3.2 与算法 1，关注按词频加权的相邻符号合并及规则复用。
- **REF-008**：Wu, Y., et al. (2016). *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*. arXiv:1609.08144v2。[论文全文](https://arxiv.org/html/1609.08144v2)。阅读 §4.1，关注 WordPiece 词表学习依据及词边界标记。
- **REF-009**：Google Research. *BERT: tokenization.py*。[官方源代码](https://github.com/google-research/bert/blob/master/tokenization.py)。核验访问时 `master` 分支的 `WordpieceTokenizer.tokenize`，关注最长匹配、`##` 与未知项；未固定提交，不作为本节运行环境依赖。
- **REF-010**：Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. ACL, 66–75。[正式论文 PDF](https://aclanthology.org/P18-1007.pdf)。阅读 §3.1–3.3，关注 Unigram 概率、动态规划、EM 与候选裁剪。
- **REF-011**：Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. EMNLP System Demonstrations, 66–71。[正式论文 PDF](https://aclanthology.org/D18-2012.pdf)。阅读 §2、§3.1–3.5，关注编码/解码、空白、规范化与特殊 ID。
- **REF-012**：Hugging Face. *Chat templates*. Transformers **4.57.1** 官方文档。[固定版本入口](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)。阅读消息角色、控制标记及重复添加特殊 Token 的说明；本节未运行其中的 API 示例。

本节材料访问日期均为 2026-10-31。BPE 训练语料、ID 数值、字节序列与成本比例由作者构造或计算，见小节正文，不冒充论文实测结果。

<a id="refs-p1-batch"></a>

## 自回归建模、Attention 与位置编码

- **REF-013**：Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. [arXiv:1506.03099v3](https://arxiv.org/html/1506.03099v3)。阅读 §2.1–2.3。
- **REF-014**：Hugging Face. Perplexity of fixed-length models. [Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/perplexity)。阅读 PPL 定义、分词影响与固定窗口评测。
- **REF-015**：Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. [arXiv:1904.09751v2](https://arxiv.org/html/1904.09751v2)。阅读 §3.1–3.3。
- **REF-016**：Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. [arXiv:2305.13245v3](https://arxiv.org/html/2305.13245v3)。阅读 §2.1–2.2。
- **REF-017**：Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. [arXiv:1911.02150v1](https://arxiv.org/html/1911.02150v1)。阅读 §3、§3.1。
- **REF-018**：Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. [arXiv:2104.09864v5](https://arxiv.org/html/2104.09864v5)。阅读 §2、§3.1–3.2、§3.4.2。
- **REF-019**：Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). Extending Context Window of Large Language Models via Positional Interpolation. [arXiv:2306.15595v2](https://arxiv.org/html/2306.15595v2)。阅读 §2.3。
- **REF-020**：PyTorch. torch.nn.Linear. [PyTorch 2.9 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html)。阅读 形状、weight 与仿射映射说明。

以上资料访问日期为 2026-10-31；Transformer 原论文 REF-002 在本轮复核 §3.1–3.5。只核验实际使用论点，未复现实验；CPU 玩具计算不支持 GPU 性能结论。

<a id="refs-norm-compute"></a>

## 归一化、门控与资源账本

- **REF-021**：Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. [arXiv:1607.06450v1](https://arxiv.org/html/1607.06450v1)。阅读 §2–3。
- **REF-022**：Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. [arXiv:1910.07467v1](https://arxiv.org/html/1910.07467v1)。阅读 §3–4.1。
- **REF-023**：Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. [arXiv:2002.04745v2](https://arxiv.org/html/2002.04745v2)。阅读 §3、表1。
- **REF-024**：Shazeer, N. (2020). GLU Variants Improve Transformer. [arXiv:2002.05202v1](https://arxiv.org/html/2002.05202v1)。阅读 §1–2、式1–6。
- **REF-025**：He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. [arXiv:1512.03385v1](https://arxiv.org/html/1512.03385v1)。阅读 §3.1–3.2。
- **REF-026**：Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. [arXiv:2101.03961v3](https://arxiv.org/html/2101.03961v3)。阅读 §2.1–2.3、辅助负载损失及附录F。
- **REF-027**：Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). [arXiv:1606.08415v5](https://arxiv.org/html/1606.08415v5)。阅读 §2。
- **REF-028**：Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. [arXiv:1502.03167v3](https://arxiv.org/html/1502.03167v3)。阅读 §3–3.1、算法1。

<a id="refs-ch3"></a>

### 第 3 章：资源账本、GPU 与性能上界

- **REF-029**：Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. [arXiv:2001.08361v1](https://arxiv.org/html/2001.08361v1)。阅读 §2.1、表1。
- **REF-030**：Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. [ICLR 2019 原文](https://openreview.net/pdf?id=Bkg6RiCqY7)。阅读算法2，核对AdamW一/二阶状态与解耦权重衰减。
- **REF-031**：PyTorch. Automatic Mixed Precision examples. [main 文档，访问于2026-10-31](https://docs.pytorch.org/docs/main/notes/amp_examples.html)。核对autocast与GradScaler的职责；未运行示例。
- **REF-032**：PyTorch. Autograd mechanics. [main 文档，访问于2026-10-31](https://docs.pytorch.org/docs/main/notes/autograd.html)。阅读Saved tensors，支持反向所需中间量的生命周期说明。
- **REF-033**：PyTorch. CUDA semantics—Memory management. [main 文档，访问于2026-10-31](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)。核对allocated、reserved及缓存分配器语义；未进行GPU测量。
- **REF-034**：NVIDIA. CUDA Programming Guide, §1.1.2. [在线文档，访问于2026-10-31](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html#the-benefits-of-using-gpus)。核对CPU低延迟与GPU高并发吞吐的设计目标和边界。
- **REF-035**：NVIDIA. CUDA Programming Guide, Hardware Multithreading. [在线文档，访问于2026-10-31](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)。核对Block到SM、32线程Warp与调度语义。
- **REF-036**：NVIDIA. Matrix Multiplication Background User's Guide. [在线文档，访问于2026-10-31](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/)。阅读§2.1–2.3，核对GEMM分块、Tensor Core对齐与Tile并行度权衡；不引用其特定硬件速度作为本机结果。
- **REF-037**：NVIDIA. CUDA Programming Guide, Programming Model §1.2.3. [Release 13.2 PDF，访问于2026-10-31](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)。核对global memory、每SM寄存器/共享内存的作用域；具体容量依架构。
- **REF-038**：NVIDIA. CUDA C++ Best Practices Guide, §10.2 Device Memory Spaces. [13.4在线文档，访问于2026-10-31](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces)。核对存储层级、global access合并及共享内存减少冗余读取；不引用V100实测表。
- **REF-039**：Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*. [UC Berkeley技术报告PDF](https://digicoll.lib.berkeley.edu/record/136692/files/EECS-2008-134.pdf)，对应CACM论文DOI 10.1145/1498765.1498785。阅读§3，核对DRAM Operational Intensity、计算/带宽屋顶与拐点。
- **REF-040**：PyTorch. `torch.profiler`. [PyTorch 2.5文档](https://docs.pytorch.org/docs/2.5/profiler.html)。核对activities、record_shapes、profile_memory与with_flops语义；with_flops仅覆盖受支持算子。本机实际版本为2.5.1+cu124。
- **REF-041**：Hugging Face Transformers. [Tokenizer](https://huggingface.co/docs/transformers/v4.32.0/en/main_classes/tokenizer) 与 [Text generation](https://huggingface.co/docs/transformers/v4.32.0/en/main_classes/text_generation)，4.32.0文档（本机4.32.1）。核对对应实验的编码、Temperature、Top-p和采样API口径；本机缺文本checkpoint，未运行生成网格。

<a id="refs-ch4"></a>

### 第 4 章：数据、优化与可恢复训练

- **REF-042**：Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. [arXiv:2101.00027v1](https://arxiv.org/html/2101.00027v1)。阅读§2–3与组件说明。
- **REF-043**：Lee, K., et al. (2022). *Deduplicating Training Data Makes Language Models Better*. [ACL Anthology](https://aclanthology.org/2022.acl-long.577/)。阅读§3–6；本书不引用其速度/质量数字作本地结果。
- **REF-044**：Biderman, S., et al. (2022). *Datasheet for the Pile*. [arXiv:2201.07311v1](https://arxiv.org/html/2201.07311v1)。阅读§3–9的动机、构成、采集、处理、使用与风险记录。
- **REF-045**：Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)。阅读算法1与§2，核对一/二阶矩及偏差校正。
- **REF-046**：Loshchilov, I., & Hutter, F. (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts*. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)。阅读§2与式2；本书4.3仅采用无restart的单段余弦衰减。
- **REF-047**：Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. [PMLR 28](https://proceedings.mlr.press/v28/pascanu13.html)。阅读§3.2.1的全局梯度范数裁剪。
- **REF-048**：NVIDIA. *Train With Mixed Precision*. [在线指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)。阅读FP16 format与训练数值范围说明；不把硬件收益写成本机结果。
- **REF-049**：Arm. *BFloat16 floating-point format*. [Architecture Reference Manual A1.4.6](https://support.arm.com/documentation/ddi0487/mc/-Part-A-Arm-Architecture-Introduction-and-Overview/-Chapter-A1-Introduction-to-the-Arm-Architecture/-A1-4-Supported-data-types/-A1-4-6-BFloat16-floating-point-format)。核对BF16字段、偏置和范围。
- **REF-050**：Micikevicius, P., et al. (2022). *FP8 Formats for Deep Learning*. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)。阅读§3与表1，区分E4M3和E5M2。
- **REF-051**：Micikevicius, P., et al. (2017). *Mixed Precision Training*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)。阅读§2–3的FP16范围、FP32主权重与loss scaling动机。
- **REF-052**：Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)。阅读§2.1–2.2；其学习率缩放与warmup是特定图像/SGD经验，不外推为LLM定律。
- **REF-053**：McCandlish, S., et al. (2018). *An Empirical Model of Large-Batch Training*. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)。阅读§2–3的大batch效率与梯度噪声经验模型。
- **REF-054**：PyTorch. `DistributedDataParallel.no_sync`. [stable文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)。核对累积期间禁用同步及退出后的首次同步语义；未运行分布式训练。
- **REF-055**：Chowdhery, A., et al. (2022). *PaLM: Scaling Language Modeling with Pathways*. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)。阅读Appendix B的Model FLOPs Utilization定义；不引用其设备结果作本地性能。
- **REF-056**：Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)。阅读§3与Appendix A的固定计算预算拟合与compute-optimal思想。
- **REF-057**：PyTorch. *Saving and Loading Models*. [官方教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)。核对通用训练checkpoint中的模型、optimizer与进度状态。
- **REF-058**：PyTorch. *Reproducibility*. [stable文档](https://docs.pytorch.org/docs/stable/notes/randomness.html)。核对随机源、非确定算法及跨版本/平台复现边界。
- **REF-059**：PyTorch. `torch.use_deterministic_algorithms`. [stable文档](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)。核对确定性算法选择、报错及“单一开关不足”的边界。
- **REF-060**：PyTorch. *Automatic Mixed Precision package*. [stable文档](https://docs.pytorch.org/docs/stable/amp.html)。核对CPU BF16 autocast与按算子选择dtype的语义；对应实验仅报告本机CPU结果。

<a id="refs-ch5"></a>

### 第 5 章：Collective 与分布式布局

- **REF-061**：NVIDIA. *NCCL Documentation*. [官方文档](https://docs.nvidia.com/deeplearning/nccl/index.html)。核对NCCL的多GPU collective与拓扑感知定位。
- **REF-062**：NVIDIA. *NCCL GPU troubleshooting*. [官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)。核对NVLink/PCIe P2P与拓扑检查边界；不作为本机带宽结果。
- **REF-063**：PyTorch. `torch.distributed.broadcast`. [stable文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast)。核对源rank、通信组、就地张量和异步handle语义；未运行多进程通信。
- **REF-064**：PyTorch. `torch.distributed.all_reduce`与`ReduceOp`. [stable文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)。核对SUM/MAX/MIN、AVG后端限制、就地结果与异步handle。
- **REF-065**：Patarasuk, P., & Yuan, X. (2009). *Bandwidth optimal all-reduce algorithms for clusters of workstations*. [DOI](https://doi.org/10.1016/j.jpdc.2008.09.002)。阅读ring reduce-scatter/all-gather成本及浮点归约顺序边界。
- **REF-066**：PyTorch. `torch.distributed.reduce_scatter`. [stable文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)。核对输入/输出大小、拼接或stack布局和异步handle。
- **REF-067**：PyTorch. `torch.distributed.all_gather`. [stable文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)。核对等长分片拼接/stack、不等长列表接口与异步handle。
- **REF-068**：PyTorch. `torch.distributed.all_to_all_single`. [stable文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)。核对等长/不等长split sizes、源—目的交换与异步handle。
- **REF-069**：Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers*. [arXiv:2101.03961v3](https://arxiv.org/abs/2101.03961)。阅读专家路由、容量、负载均衡与溢出Token处理；性能数字不作本机结果。
- **REF-070**：PyTorch. `DistributedDataParallel`. [stable官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。核对梯度同步而非输入切分、参数顺序与初始同步、梯度平均、bucket及`no_sync`语义；本机版本2.5.1+cu124，未运行多进程。
- **REF-071**：PyTorch. *Distributed Data Parallel design note*. [stable官方设计说明](https://docs.pytorch.org/docs/stable/notes/ddp.html)。核对reducer、autograd hook、bucket All-Reduce与反向末尾等待；该文档自述基于较早实现，正文不固定易变内部细节。
- **REF-072**：Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)。阅读§5、§7，核对三阶段分片、状态内存与通信量抽象；论文性能数字不作本机结果。
- **REF-073**：DeepSpeed. *ZeRO documentation*. [0.19.9在线文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)。核对Stage 1/2/3定义及CPU/NVMe offload状态范围；未运行其API。
- **REF-074**：DeepSpeed. *ZeRO-Offload tutorial*. [官方教程](https://www.deepspeed.ai/tutorials/zero-offload/)。核对optimizer memory/compute迁移到host CPU及Stage 1/2配置边界；不采用教程硬件结果。
- **REF-075**：PyTorch. `torch.distributed.fsdp.fully_shard`. [stable 2.14官方文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。核对FSDP2的`DTensor`参数、All-Gather、Reduce-Scatter、reshard、通信分组和state-dict语义；本机2.5.1未运行该在线版本API。
- **REF-076**：PyTorch. *Getting Started with FSDP2*. [官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。核对FSDP2与DDP布局、自底向上按Block分组及预取机制；不采用教程性能数字。
- **REF-077**：Shoeybi, M., et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)。阅读§3，核对Transformer层内并行、MLP列切/行切组合及同步位置；性能与质量数字不作本机结果。
- **REF-078**：PyTorch. *Tensor Parallelism—`torch.distributed.tensor.parallel`*. [main官方文档](https://docs.pytorch.org/docs/main/distributed/tensor.parallel.html)。核对DTensor上的Colwise/Rowwise/Sequence布局和`parallelize_module`；API标为experimental，本机2.5.1未运行。
- **REF-079**：Huang, Y., et al. (2019). *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)。阅读§2与Figure 2，核对连续层分区、micro-batch、fill-drain、同步更新、rematerialization与bubble口径；性能数字不作本机结果。
- **REF-080**：Narayanan, D., et al. (2021). *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)。阅读§2.2与Figure 4，核对同步flush、非交错1F1B、激活内存和异步权重陈旧边界；不采用论文吞吐数字。
- **REF-081**：Korthikanti, V., et al. (2023). *Reducing Activation Recomputation in Large Transformer Models*. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)。阅读Sequence Parallelism部分，核对TP组内序列分片、collective布局转换与激活复制动机；论文性能数字不作本机结果。
- **REF-082**：PyTorch. `SequenceParallel`. [stable 2.14官方文档](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)。核对LayerNorm、Dropout、RMSNorm Python等模块的sequence-sharded输入/输出和参数复制语义；本机2.5.1未运行在线API。
- **REF-083**：Liu, H., Zaharia, M., & Abbeel, P. (2024). *Ring Attention with Blockwise Transformers for Near-Infinite Context*. [arXiv:2310.01889v4](https://arxiv.org/abs/2310.01889)。核对序列跨设备分块、K/V块环形传输与blockwise Attention；论文中的长度、重叠和性能结论不作本机结果。
- **REF-084**：NVIDIA. *Megatron Core Context Parallel Overview*. [latest官方开发者指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)。核对CP与SP差异、远端K/V依赖、ring P2P、反向K/V梯度归约及GQA/MQA通信边界；未运行实现。
- **REF-085**：NVIDIA. *Megatron Core Mixture of Experts*. [0.19.2官方开发者指南](https://docs.nvidia.com/megatron-core/developer-guide/0.19.2/user-guide/features/moe.html)。核对专家并行度、Router/top-k、dispatch-compute-combine、All-to-All dispatcher、负载策略及TP/EP组合边界；未运行实现或采用性能数字。
- **REF-086**：NVIDIA. *Megatron Core Parallelism Strategies Guide*. [latest官方开发者指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。核对DP/TP/PP/CP/EP作用对象与组合入口；示例配置不作为目标集群最优布局。
- **REF-087**：PyTorch. torch.utils.checkpoint. [main官方文档](https://docs.pytorch.org/docs/main/utils/checkpoint.html)。核对保存边界、反向重放、RNG状态和非reentrant实现建议；在线API晚于本机2.5.1，未运行。
- **REF-088**：Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training Deep Nets with Sublinear Memory Cost*. [arXiv:1604.06174v2](https://arxiv.org/abs/1604.06174)。核对以重计算换中间特征内存的基本机制；性能数字不作本机结果。
- **REF-089**：NVIDIA. *Megatron Core TransformerConfig*. [0.19.0 API文档](https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.transformer.transformer_config.html)。核对full/selective、uniform/block重计算配置术语；未运行实现。

<a id="refs-ch6"></a>

### 第 6 章：高性能 Attention 与测量

- **REF-090**：Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。阅读摘要与§2–3，核对标准Attention二次中间矩阵、HBM/SRAM IO-aware动机及精确分块方向；速度数字不作本机结果。
- **REF-091**：NVIDIA. *CUDA Programming Guide—Writing SIMT Kernels*. [官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。核对Kernel线程执行、global memory合并访问与Shared Memory复用；具体事务和资源规则依目标架构，未运行CUDA。
- **REF-092**：NVIDIA. *Nsight Compute Profiling Guide*. [官方指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。核对duration、DRAM bytes/throughput、compute throughput、occupancy及Profiler扰动边界；未运行Nsight，不引用硬件实测值。
- **REF-093**：PyTorch. *Profiling to understand torch.compile performance*. [stable官方文档](https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile)。核对算子、Kernel、launch事件与GPU时间线空隙的诊断；在线版本晚于本机2.5.1，未运行`torch.compile`或采用速度数字。
- **REF-094**：Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)。阅读§3.1–3.3，核对非MatMul操作、序列并行和Warp work partitioning；性能数字不作本机结果。
- **REF-095**：Dao-AILab. *flash-attention README*. [官方实现](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)。核对访问时的causal、变长、MQA/GQA接口及后端/版本边界；未安装或运行。
- **REF-096**：Shah, J., et al. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)。核对Hopper异步流水、Warp Specialization与FP8方向；性能数字不作本机结果。
- **REF-097**：Shah, J., et al. (2026). *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)。核对Blackwell异步MMA、指数/重缩放、Tensor Memory与2-CTA方向；性能数字不作本机结果。
- **REF-098**：PyTorch. `torch.nn.functional.scaled_dot_product_attention`. [main官方文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)。核对SDPA后端选择、`sdpa_kernel`、GQA、dropout与数值边界；在线API晚于本机2.5.1，未运行强制后端。
- **REF-099**：PyTorch. *CUDA semantics*. [stable官方文档](https://docs.pytorch.org/docs/stable/notes/cuda)。核对异步执行、Event/同步计时及allocated/reserved边界；未运行CUDA。
- **REF-100**：PyTorch. `torch.cuda.memory.max_memory_allocated`. [stable官方文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)。核对重置峰值后Tensor allocated bytes的统计口径；未运行CUDA。
- **REF-101**：PyTorch. *Profiler recipe*. [官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe)。核对CPU/CUDA活动、shape与内存诊断；在线版本晚于本机2.5.1，未运行Profiler。

<a id="refs-ch7"></a>

### 第 7 章：推理与服务系统

- **REF-102**：vLLM. *Benchmark CLI—Understanding the Latency Metrics*. [latest官方文档](https://docs.vllm.ai/en/latest/benchmarking/cli/)。核对TTFT/ITL/TPOT的客户端测量点、公式与多Token chunk边界；未运行vLLM。
- **REF-103**：NVIDIA. *GenAI-Perf metrics*. [官方文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)。核对TTFT、ITL、请求延迟及系统Token/请求吞吐的统计单位；未运行工具。
- **REF-104**：Hugging Face Transformers. *Caching*. [v5.6.0官方文档](https://huggingface.co/docs/transformers/v5.6.0/cache_explanation)。核对逐层历史K/V复用、当前K/V追加、past+current mask shape及仅推理边界；未运行API。
- **REF-105**：Hugging Face Transformers. *Cache strategies*. [官方文档](https://huggingface.co/docs/transformers/en/kv_cache)。核对低精度Cache内存/延迟权衡及Sliding/Chunked层的结构性封顶；实现支持随版本变化，未运行API。
- **REF-106**：vLLM. *Automatic Prefix Caching*. [latest官方设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)。核对前缀Block hash、完整Block复用、额外配置与cache salt隔离；未运行vLLM。
- **REF-107**：vLLM. *Scheduler configuration*. [latest官方API文档](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)。核对Prefill按单iteration剩余Token预算分块及多模态调度边界；实现随版本变化，未运行vLLM。
- **REF-108**：NVIDIA Dynamo. *Disaggregated Serving*. [dev官方文档](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving)。核对mixed prefill/decode iteration中chunk大小对ITL干扰形状及Prefill饱和边界；未运行后端。
- **REF-109**：Agrawal, A., et al. (2024). *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)。核对Prefill/Decode特征与chunked-prefill调度动机；不采用论文性能数字作本机结果。
- **REF-110**：Yu, G.-I., et al. (2022). *ORCA: A Distributed Serving System for Transformer-Based Generative Models*. [OSDI 2022论文](https://www.usenix.org/system/files/osdi22-yu.pdf)。核对iteration-level scheduling、轮间重组与selective batching边界；论文性能数字不作本机结果。
- **REF-111**：vLLM. *V1 Scheduler source documentation*. [stable官方源码文档](https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/)。核对请求完成时立即或等待在途step后释放Block的实现语义；实现会演进，未运行vLLM。
- **REF-112**：Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)。阅读§3、§4.1–4.4及§7.1–7.2，核对固定大小逻辑/物理KV块、Block Table、按需增长、引用计数/Copy-on-Write与块大小代价；论文性能数字不作本机结果。
- **REF-113**：Zhong, Y., et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*. [OSDI 2024正式论文](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)。阅读§2.3、§3与§4，核对共置干扰、TTFT/TPOT目标、资源/并行策略耦合与按阶段配置；性能幅度不作本机结果。
- **REF-114**：Patel, P., et al. (2024). *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. [ISCA 2024正式论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)。阅读§III–IV，核对Prompt/Token阶段的计算、内存、延迟与功耗差异，以及独立机器池和容量规划动机；特定GPU与性能结论不外推。
- **REF-115**：Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. [ICML 2023正式论文](https://proceedings.mlr.press/v202/leviathan23a.html)。阅读§2–3、Algorithm 1与Appendix A，核对Draft/Target流程、接受/校正采样、分布保持与期望生成数；论文速度数字不作本机结果。
- **REF-116**：Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)。阅读Algorithm 2及§4–6，核对修改拒绝采样、全接受额外Token、Draft选择和lookahead成本权衡；论文性能数字不作本机结果。
- **REF-117**：Zhang, J., et al. (2023). *Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding*. [arXiv:2309.08168](https://arxiv.org/abs/2309.08168)。核对选择性跳层Draft与完整Target验证的Self-Speculative思路；论文速度数字不作本机结果。
- **REF-118**：Xiao, G., et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. [ICML 2023正式论文](https://proceedings.mlr.press/v202/xiao23c.html)。阅读摘要与§3，核对激活离群值、逐通道等价缩放及W8A8 PTQ机制；论文质量与性能数字不作本机结果。
- **REF-119**：Lin, J., et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. [MLSys 2024正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)。阅读摘要与§3，核对weight-only PTQ、显著权重通道和激活感知缩放；论文质量与性能数字不作本机结果。
- **REF-120**：Frantar, E., et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)。阅读摘要与§2–3，核对基于近似二阶信息的逐层一次性权重量化；论文质量与性能数字不作本机结果。
- **REF-121**：vLLM. *Architecture Overview*. [latest官方文档](https://docs.vllm.ai/en/latest/design/arch_overview/)。核对Engine Core、Token预算Scheduler、waiting/running请求、KV不足时抢占与OpenAI兼容API边界；实现随版本变化，未运行vLLM。
- **REF-122**：SGLang. *Runtime source documentation*. [官方GitHub main](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt)。核对API/Tokenizer、Scheduler/Worker、Detokenizer服务流水线及Radix Cache/cache-aware调度方向；实现随commit变化，未运行SGLang。
- **REF-123**：NVIDIA. *TensorRT-LLM Architecture Overview*. [官方开发者指南](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)。核对PyExecutor后台循环与Scheduler、KVCacheManager、ModelEngine、Sampler职责；实现随版本变化，未运行TensorRT-LLM。

<a id="refs-ch8"></a>

### 第 8 章：后训练、评测与安全

- **REF-124**：Wei, J., et al. (2022). *Finetuned Language Models Are Zero-Shot Learners*. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)。阅读摘要与§2–3，核对多任务自然语言指令微调和未见任务zero-shot评测；性能数字不作本机结果。
- **REF-125**：Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)。阅读摘要、§3与图2，核对人工示范SFT、输出排序、Reward Model和PPO流程；该配方不外推为所有模型固定流程。
- **REF-126**：Lambert, N., et al. (2025). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)。核对SFT、DPO、RLVR及阶段化评测的开放配方；论文结果不作本机实测。
- **REF-127**：Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)。阅读摘要与§4，核对数学任务奖励和GRPO的具体研究语境；不外推到开放式任务。
- **REF-128**：Hugging Face Transformers. *Chat templates*. [main官方文档](https://huggingface.co/docs/transformers/main/chat_templating)。核对role/content序列化、控制Token、训练模板与generation prompt边界；API随版本变化，本机未运行在线版本。
- **REF-129**：Zhou, C., et al. (2023). *LIMA: Less Is More for Alignment*. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)。阅读摘要、§2与§5–6，核对少量精心筛选示范的具体研究；不外推固定样本数，论文结果不作本机实测。
- **REF-130**：Shi, W., et al. (2024). *Instruction Fine-Tuning: Does Prompt Loss Matter?* [arXiv:2401.13586](https://arxiv.org/abs/2401.13586)。核对prompt loss weight的研究问题与实验设置；用于说明mask策略需声明和评测，结果不作本机实测。
- **REF-131**：Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)。阅读§3–4与图1，核对冻结主权重、列向量BA低秩更新、alpha/r、初始化、目标模块与合并；性能数字不作本机结果。
- **REF-132**：Microsoft. *loralib official implementation*. [官方GitHub](https://github.com/microsoft/LoRA)。核对PyTorch存储方向、训练/评估merge切换及冻结非LoRA参数入口；实现随commit变化，本机未安装或运行。
- **REF-133**：Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)。阅读§3–4与附录，核对冻结4-bit Base中反传到LoRA、NF4、double quantization和paged optimizer；论文结果不作本机实测。
- **REF-134**：Dettmers, T., et al. *QLoRA official repository*. [官方GitHub](https://github.com/artidoro/qlora)。核对NF4/FP4、compute dtype与paged optimizer配置入口；实现随commit变化，本机未安装或运行。
- **REF-135**：Hugging Face. *bitsandbytes documentation*. [官方文档](https://huggingface.co/docs/bitsandbytes/index)。核对4-bit量化Base与LoRA训练的当前库入口；API与硬件支持随版本变化，本机未运行。
- **REF-136**：Bai, Y., et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)。核对迭代偏好数据、校准、竞争目标与分布外检测的具体研究路线；结果不外推为普适结论。
- **REF-137**：Gao, L., Schulman, J., & Hilton, J. (2023). *Scaling Laws for Reward Model Overoptimization*. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)。核对proxy Reward Model过优化与gold评价下降的受控机制证据；合成设定和结果不作本机实测。
- **REF-138**：Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)。阅读§4、式(5)–(7)、§5与Appendix A，核对KL正则重参数化、Reference Policy和chosen/rejected log-ratio目标；论文质量与稳定性结果不作本机实测。
- **REF-139**：Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)。阅读§2–3与式(7)，核对Policy ratio与clipped surrogate；原论文实验不外推为LLM训练或本机稳定性保证。
- **REF-140**：Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding*. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)。核对57学科多项选择任务及适用范围；论文分数不作当前模型或本机结果。
- **REF-141**：Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)。核对GSM8K规模、任务和Verifier语境；论文结果不作本机实测。
- **REF-142**：Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code*. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)。阅读§3、式(1)–(2)与Appendix，核对HumanEval、功能正确性及pass@k估计；论文模型分数不作本机实测。
- **REF-143**：Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)。核对Judge设置及位置、冗长、自增强和推理偏差；论文一致率不外推为本书Judge质量。
- **REF-144**：Liang, P., et al. (2023). *Holistic Evaluation of Language Models*. [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)。核对场景—指标矩阵与多指标报告思想；不宣称覆盖所有部署风险。
- **REF-145**：Cao, B., et al. (2024). *On the Worst Prompt Performance of Large Language Models*. [NeurIPS论文页](https://papers.nips.cc/paper_files/paper/2024/hash/7fa5a377b7ffabcce43cd00231bb3f9c-Abstract-Conference.html)。核对case-level语义改写、prompt敏感性与最差表现视角；论文差值不作本机结论。
- **REF-146**：Li, Y., et al. (2024). *An Open-Source Data Contamination Report for Large Language Models*. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.30/)。核对公开污染审计与检测透明性；论文污染率和影响不外推为本书模型结果。
- **REF-147**：Lin, S., Hilton, J., & Evans, O. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)。核对真实性评测任务与范围；论文结果不外推所有事实任务。
- **REF-148**：Greshake, K., et al. (2023). *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)。核对不可信检索数据、间接注入与工具/API风险；演示不外推所有系统攻击率。
- **REF-149**：Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models*. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)。核对对抗后缀绕过的机制证据；不复制攻击载荷，成功率不外推当前系统。
- **REF-150**：Carlini, N., et al. (2021). *Extracting Training Data from Large Language Models*. [arXiv:2012.07805](https://arxiv.org/abs/2012.07805)。核对特定GPT-2设置下的可抽取记忆与隐私风险；样本数不外推其他模型。
- **REF-151**：Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal*. [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)。核对自动红队和鲁棒拒答评测框架；不宣称覆盖未知攻击。
- **REF-152**：Kadavath, S., et al. (2022). *Language Models (Mostly) Know What They Know*. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)。核对P(True)/P(IK)与跨任务校准限制；口头置信不作概率保证。
- **REF-153**：NIST (2024). *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile*. [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)。核对生成式AI多类风险与治理行动；不是产品认证或零风险保证。

<a id="refs-ch9"></a>

### 第 9 章：长上下文、多模态与新架构

- **REF-154**：Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)。核对局部窗口与任务相关全局注意力；性能数字不作本机结果。
- **REF-155**：Zaheer, M., et al. (2020). *Big Bird: Transformers for Longer Sequences*. [arXiv:2007.14062](https://arxiv.org/abs/2007.14062)。核对局部、随机和全局连接组合的稀疏Attention；不外推为任意稀疏模式等价Full Attention。
- **REF-156**：Hsieh, C.-P., et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)。核对检索、multi-hop tracing、aggregation与QA组成的长上下文评测；论文结果不作本机实测。
- **REF-157**：Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)。核对多文档问答/键值检索中的证据位置敏感性；现象不外推为所有模型的必然规律。
- **REF-158**：Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)。核对图像Patch、线性嵌入和位置表示；论文结果不作本机实测。
- **REF-159**：Alayrac, J.-B., et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)。核对视觉编码器、Perceiver Resampler、门控Cross-Attention和图文交错接口；性能数字不作本机结果。
- **REF-160**：Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual Instruction Tuning*. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)。核对视觉编码器—LLM连接、投影接口和视觉指令微调；论文分数不作本机结果。
- **REF-161**：Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)。核对以中间步骤示例引导CoT的机制；论文收益不外推到所有模型或任务。
- **REF-162**：Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)。核对多路径采样与答案边缘化/投票；提升幅度不作本机结果。
- **REF-163**：Lightman, H., et al. (2024). *Let's Verify Step by Step*. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)。核对Outcome与Process监督及其特定实验语境；结果不外推为普遍规律。
- **REF-164**：Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)。核对测试时计算策略与按题目难度分配预算的研究；论文倍率不作本机结果。
- **REF-165**：Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)。核对参数化生成器与显式非参数memory结合的RAG定义；任务结果不作本机实测。
- **REF-166**：Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)。核对query/passage双编码器与Dense Retrieval；相对BM25结果不外推。
- **REF-167**：Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)。核对工具选择、参数与结果利用的训练路线；不替代运行时权限控制。
- **REF-168**：Yao, S., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)。核对交错推理、动作与Observation的研究范式；论文结果不作生产Agent保证。
- **REF-169**：Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)。核对状态空间递推、结构化参数与长序列计算路线；论文结果不作本机实测。
- **REF-170**：Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)。核对输入依赖选择性与硬件感知Selective Scan；吞吐和长度结论不外推。
- **REF-171**：Lieber, O., et al. (2024). *Jamba: A Hybrid Transformer-Mamba Language Model*. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)。核对Transformer、Mamba与MoE交错的混合实例；特定配置不作通用最优或本机结果。

REF-001–044访问日期沿用各条记录，REF-045–056访问日期为2026-11-02，REF-057–084访问日期为2026-11-03，REF-085–106访问日期为2026-11-05，REF-107–109访问日期为2026-11-06，REF-110–111为2026-11-08，REF-112–137为2026-11-09，REF-138–171为2026-11-11；核验范围与限制见来源台账。计算账本的数值按本书行向量/GQA/SwiGLU/绑定嵌入口径独立相加，不能直接套用文献中的其他结构参数量。
