# 必要前置知识

[总目录](../index.qmd) · [符号表](notation.md)

本附录只回顾阅读本书会反复用到的工具，不替代线性代数、概率、微积分或Python课程。全书统一符号、线性层方向与单位见[符号表](notation.md)。

## 五分钟自测

先不查答案，用一句话或一行算式完成下面五项。它们不是入门门槛，而是帮你决定从哪里开始读。

1. `X` 的 shape 为 `[2,8,32]`，右乘数学权重 `[32,96]`，输出 shape 是什么？主要矩阵乘法是多少 FLOPs？
2. 两个 micro-batch 的 NLL 总和与有效 Token 数分别为 `(12,8)` 和 `(9,3)`。合并后的平均 NLL 是多少？
3. FP16 训练使用 loss scaling 时，梯度裁剪应发生在 `unscale` 之前还是之后？
4. 一段 GPU 代码的 Python 墙钟计时没有同步。这个数字能否直接当作 Kernel 用时？
5. 公式算得某中间量为 128 MiB，但实验没有运行。应把它标为解析估算、本机实测还是 `0`？

::: {.callout-note title="自测答案与下一步"}
1. 输出为 `[2,8,96]`；把前两轴合成 $N=16$，主项为 $2\times16\times32\times96=98{,}304$ FLOPs。
2. $(12+9)/(8+3)=21/11\approx1.9091$，不能无权平均两个 local mean。
3. 在 `unscale` 之后、optimizer 更新之前。
4. 不能；GPU 通常异步，应使用事件或在计时边界显式同步。
5. 标为解析估算；未运行状态写 `not_run` 或 `null`，不能写 0。

若五项都能解释，可以直接从[第 1 章](../chapters/01-foundations.md)开始；某项犹豫时，只复习下表指向的段落即可。
:::

## 按需复习路径

| 你遇到的现象 | 先复习本页 | 再到正文验证 |
| --- | --- | --- |
| 矩阵乘不上、头轴换乱或广播报错 | [矩阵、张量与形状](#矩阵张量与形状) | [第 2 章](../chapters/02-transformer.md)的 GQA shape 链 |
| loss 数字能算，但分母与 mask 不确定 | [概率、对数与损失](#概率对数与损失) | [第 1 章](../chapters/01-foundations.md)的下一 Token 目标与[第 4 章](../chapters/04-pretraining.md)的有效 Token 聚合 |
| 不清楚梯度累积、裁剪或成功更新点 | [导数、链式法则与优化](#导数链式法则与优化) | [第 4 章](../chapters/04-pretraining.md)的训练步状态机 |
| 代码能跑但结果/速度不可信 | [Python与张量程序](#python与张量程序) | [第 6 章](../chapters/06-efficient-attention.md)的正确性与计时协议 |
| MiB、FLOPs、峰值显存或证据标签混用 | [单位与证据类型](#单位与证据类型) | [第 3 章](../chapters/03-compute-and-gpu.md)的四本账 |

## 矩阵、张量与形状

矩阵乘法$\mathbf C=\mathbf A\mathbf B$要求$\mathbf A\in\mathbb R^{m\times k}$、$\mathbf B\in\mathbb R^{k\times n}$，结果为$m\times n$；本书主要运算量口径是$2mkn$ FLOPs。线性层采用行向量约定：一批输入`[N,d_in]`右乘数学权重`[d_in,d_out]`得到`[N,d_out]`。PyTorch `Linear.weight`实际存成`[d_out,d_in]`，框架内部计算与数学权重转置等价。

LLM常见张量`[B,T,d]`分别表示batch、序列长度和隐藏宽度。Q变形成`[B,H,T,d_h]`，K/V为`[B,H_KV,T,d_h]`；转置最后两维后，`Q @ K.transpose(-1,-2)`得到query×key分数`[B,H,T_q,T_k]`。广播只允许长度为1或相同的轴自动扩展；遇到shape错误，先逐轴写出含义，不靠试错添加`unsqueeze`。

范数$\lVert\mathbf x\rVert_2=(\sum_i x_i^2)^{1/2}$常用于相对误差。最大绝对误差与相对L2误差回答不同问题：前者看最坏元素，后者看整体能量比例；浮点实现通常需要同时报告。

## 概率、对数与损失

条件概率链式法则为$p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$。自回归语言模型用已有前缀预测下一个Token；训练代码通常把`tokens[:, :-1]`作为输入、`tokens[:, 1:]`作为目标。对数把乘积变成和：$-\log\prod_t p_t=\sum_t-\log p_t$。

Softmax把logits $z_i$变成概率：

$$
p_i=\frac{e^{z_i-c}}{\sum_j e^{z_j-c}},\qquad c=\max_jz_j.
$$

减去共同最大值不改变概率，却避免指数溢出。one-hot目标的交叉熵等于真实类别的负对数概率。含padding或SFT非目标区间时，先汇总有效Token的NLL，再除有效Token总数；不同长度micro-batch的local mean不能无条件再做简单平均。困惑度为平均NLL的指数，只有Tokenizer、数据与上下文口径相同时才可比较。

## 导数、链式法则与优化

若$y=f(u)$、$u=g(x)$，则$\mathrm dy/\mathrm dx=(\mathrm dy/\mathrm du)(\mathrm du/\mathrm dx)$；计算图反向传播把输出梯度沿局部导数逐层累积到参数。梯度表示当前位置的一阶局部变化，不保证有限步长后损失一定下降。

最简单梯度下降为$\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L$。AdamW还维护一阶/二阶矩并把解耦权重衰减应用到参数。混合精度训练可能先缩放loss，反向后再unscale；梯度裁剪必须在unscale之后、optimizer更新之前。梯度累积时各micro-batch应按有效Token数加权，成功更新数与micro-step数分开记录。

## Python与张量程序

读者应能创建虚拟环境、运行脚本、读取异常栈，并理解函数、类、列表/字典、上下文管理器和命令行参数。实验至少固定seed、保存不可变配置、记录Python/框架/硬件版本，并把原始JSON与汇总图分开保存。

调试张量程序的推荐顺序是：检查dtype/device/shape，再检查有限值与mask，最后比较小尺寸参考实现。`torch.no_grad()`或`inference_mode()`关闭梯度记录，不等于模型自动进入`eval()`；`model.eval()`改变Dropout等模块语义，也不等于关闭autograd。GPU操作通常异步，性能计时必须用事件或显式同步；CPU计时不能推出GPU速度。

## 单位与证据类型

byte、MiB、GiB分别按$2^{20}$和$2^{30}$换算；GB若出现则说明十进制。FLOPs是运算数，FLOP/s是速率，二者不能混写。教材把结论分为三类：公式得到的“解析估算”、有明确出处的“文献报告”、在记录环境中实际运行的“本机实测”。缺硬件时写`not_run`或`null`，不能写0，也不能把构造算术或合成trace改称真实性能。

复习的终点不是“读完附录”，而是能重新完成刚才卡住的 shape、分母、调试或证据判断。准备好后可从[第 1 章](../chapters/01-foundations.md)开始；需要查询全局记号时返回[符号表](notation.md)，需要核对习题推导时使用[答案与提示](exercise-solutions.md)。
