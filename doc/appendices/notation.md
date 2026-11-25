# 统一符号与贯穿配置

本页汇总全书稳定符号、贯穿配置和计算口径，依据编写规范 v1.0；局部符号仍以首次出现的小节定义为准。

[总目录](../index.qmd) · [必要前置知识](prerequisites.md) · [习题答案](exercise-solutions.md)

## 按任务反查

先按正在做的事找变量，再到后面的主题表确认唯一含义与约束。

| 任务 | 先查这些符号 | 还要固定的口径 |
| --- | --- | --- |
| 列模型结构与参数账 | $L,d,H,H_{\mathrm{KV}},d_h,d_{\mathrm{ff}},V,P$ | 是否绑定 Embedding、是否有偏置、局部例子还是贯穿配置 |
| 写 Attention shape 或 KV 容量 | $B,T,H,H_{\mathrm{KV}},d_h,s_{\mathrm{KV}}$ | 轴顺序、当前长度与最大长度、dtype |
| 算训练 global batch 与 Token | $b_{\mathrm{micro}},a_{\mathrm{acc}},p_{\mathrm{DP}},T,D_{\mathrm{tok}}$ | 序列数还是有效 Token 数、mask 与累积窗口 |
| 列显存、通信或性能账 | $P,s,p_{\mathrm{TP}},p_{\mathrm{PP}},p_{\mathrm{CP}}$ | 逻辑 payload、物理峰值和 FLOP/s 不混用 |
| 描述生成策略 | $\tau,k_{\mathrm{samp}},p_{\mathrm{nuc}}$ | 温度与截断顺序、重归一化和停止条件 |
| 描述 LoRA/对齐 | $r,\alpha,\theta,p_\theta,\pi_\theta$ | Base 是否冻结、Reference 是否固定、符号作用域 |

## 模型与实验配置

沿用原目录的假设 Dense Decoder-only 模型，不对应具体商品模型：

| 符号 | 含义 | 贯穿配置 |
| --- | --- | ---: |
| $L$ | Transformer 层数 | 32 |
| $d$ | 隐藏维度 | 4096 |
| $H$ | Query 头数 | 32 |
| $H_{\mathrm{KV}}$ | Key/Value 头数 | 8 |
| $d_h$ | 每头维度，$d=Hd_h$ | 128 |
| $d_{\mathrm{ff}}$ | SwiGLU 中间维度 | 11008 |
| $V$ | 词表大小 | 32000 |
| $T_{\max}$ | 示例最大上下文长度 | 4096 |

固定结构：Pre-RMSNorm、RoPE、GQA、SwiGLU、无偏置线性层、最终 RMSNorm、输入嵌入与 LM Head 权重绑定。RMSNorm 仅有缩放参数。仅 Q/K 使用 RoPE。一般性介绍其他结构时须明确其与贯穿模型的差异。

BF16 是教学训练计算配置，不表示优化器状态、主权重和所有算子均采用 BF16。训练显存示例必须逐项声明权重、梯度、可选 FP32 主权重、Adam 状态精度；禁止固定套用某个“每参数字节数”。默认 KV Cache 示例使用 BF16，每元素 $s=2$ 字节。

手算示例可另用小矩阵，必须标注“局部玩具例子”。实际 Mini-GPT 默认使用 $L=2,d=128,H=4,H_{\mathrm{KV}}=2,d_h=32,d_{\mathrm{ff}}=352,V=256,T_{\max}=128$，配合 UTF-8 字节 tokenizer；使用与贯穿模型相同结构，便于 CPU 小规模验证。分词器专节仍讲解 BPE 等方法。替换实验模型时记录配置、理由与受影响结果，不将实验数字混入贯穿账本。

## 按主题查符号

### 序列、位置与工作负载

| 符号 | 唯一含义与约束 |
| --- | --- |
| $B$ | 当前计算的序列 batch 数；训练和推理语境均需说明 |
| $T$ | 当前序列或缓存长度，不总等于 $T_{\max}$ |
| $x_t$ | 第 $t$ 个 Token ID；正文 Token 索引从 1 起，代码从 0 起 |
| $x_{<t}$ | 位置 $t$ 前的 Token；首位置条件须明确 BOS 或空上下文 |

### 模型、张量与目标

| 符号 | 唯一含义与约束 |
| --- | --- |
| $\theta$ | 全部模型参数 |
| $\mathbf{X}$ | 隐藏状态，形状 $B\times T\times d$ |
| $\mathbf{Q},\mathbf{K},\mathbf{V}$ | 投影后的 Query、Key、Value；粗体 $\mathbf{V}$ 区别于词表大小 $V$ |
| $\mathbf{Z}$ | 输出 logits，形状 $B\times T\times V$ |
| $\mathcal{L}$ | 损失；不使用层数 $L$ 表示损失 |

### 训练、资源与并行

| 符号 | 唯一含义与约束 |
| --- | --- |
| $P$ | 参数总数 |
| $D_{\mathrm{tok}}$ | 训练累计有效 Token 数，不表示隐藏维度 |
| $s$ | 每元素存储字节数，区分精度时用 $s_{\mathrm{w}},s_{\mathrm{g}},s_{\mathrm{KV}}$ 等 |
| $p_{\mathrm{DP}},p_{\mathrm{TP}},p_{\mathrm{PP}},p_{\mathrm{CP}}$ | 各并行度，$p$ 不再单独指概率 |
| $b_{\mathrm{micro}},a_{\mathrm{acc}}$ | 每 DP 副本的 micro-batch 序列数、梯度累积步数 |

### 采样与对齐

| 符号 | 唯一含义与约束 |
| --- | --- |
| $r,\alpha$ | LoRA 秩、缩放超参数 |
| $\tau,k_{\mathrm{samp}},p_{\mathrm{nuc}}$ | 温度、Top-k 截断数、Top-p 阈值 |

模型概率写为 $p_\theta(\cdot)$；条件策略在对齐章可用 $\pi_\theta(y\mid x)$，首次定义其与语言模型概率的关系。其他局部变量可使用，但必须说明作用域，不得覆盖上表含义。算法所需循环索引先定义。RoPE 位置坐标统一用 $u=t-1$，从 0 起。

## 线性层、单位与索引

数学使用行向量：$\mathbf{Y}=\mathbf{X}\mathbf{W}$，$\mathbf{W}\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$。代码的权重存储方向须在对应实现中说明。LoRA 写为 $\Delta\mathbf{W}=(\alpha/r)\mathbf{A}_{\mathrm{LoRA}}\mathbf{B}_{\mathrm{LoRA}}$，矩阵形状依次为 $d_{\mathrm{in}}\times r$ 与 $r\times d_{\mathrm{out}}$。

Q 形状为 $B\times T\times H\times d_h$，K/V 为 $B\times T\times H_{\mathrm{KV}}\times d_h$；GQA 要求 $H$ 可被 $H_{\mathrm{KV}}$ 整除。

一次乘加记 2 FLOPs；FLOPs 是运算数，FLOP/s 是吞吐率。显存以 byte、MiB、GiB 表示，$1\,\mathrm{MiB}=2^{20}$ bytes，$1\,\mathrm{GiB}=2^{30}$ bytes。正文 Token 索引从 1 起，代码与 RoPE 坐标从 0 起，RoPE 位置 $u=t-1$。局部公式的适用条件仍以正文首次定义为准；代入数值前先确认使用的是贯穿配置、Mini-GPT 配置还是明确标注的玩具例子。
