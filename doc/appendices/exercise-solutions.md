# 习题答案与提示

本附录收录第 1–10 章的 40 道习题答案与提示；正文题目均链接到对应章节锚点。

[总目录](../index.qmd) · [必要前置知识](prerequisites.md) · [符号表](notation.md)

::: {.callout-tip title="把答案当作诊断工具"}
1. **先独立作答：** 写出假设、公式、单位和结论边界，再打开本页核对。
2. **再比较路径：** 不只看末位数字；先检查分母、dtype、逻辑量/物理量和“解析/实测”标签是否一致。
3. **最后定位误差：** 概念或单位卡住时回看[必要前置知识](prerequisites.md)，符号混淆时查[符号表](notation.md)，再重新完成原题。
:::

开放性题目的其他结论若假设与证据充分，也可能成立。以下导航中的主题词说明每组答案主要检查什么。

| 章节 | 快速定位 | 主要检查 |
| --- | --- | --- |
| 第 1 章 | [答案](#solutions-1) | 环节边界、Token 成本、损失与采样 |
| 第 2 章 | [答案](#solutions-2) | GQA shape、因果性、归一化与参数 |
| 第 3 章 | [答案](#solutions-3) | 参数、FLOPs、GPU 映射与 Roofline |
| 第 4 章 | [答案](#solutions-4) | Token 归一化、混合精度、训练预算与恢复 |
| 第 5 章 | [答案](#solutions-5) | collective、并行布局与 pipeline |
| 第 6 章 | [答案](#solutions-6) | Tile、Online Softmax 与测量协议 |
| 第 7 章 | [答案](#solutions-7) | KV、PD 分离、推测解码与服务指标 |
| 第 8 章 | [答案](#solutions-8) | SFT、LoRA、DPO 与安全评测 |
| 第 9 章 | [答案](#solutions-9) | 长上下文、多模态、Agent 与混合架构 |
| 第 10 章 | [答案](#solutions-10) | 项目账本、证据层级与 Gate 状态 |

<a id="solutions-1"></a>

## 第1章答案与提示

1. 四项依次是预训练、后训练、推理、服务。“生成了文本”只能说明用过前向计算；是单请求推理还是经过排队和批处理的服务，还需查看请求调度边界。
2. 长度比为$1200/750=1.6$，所以线性主项约为1.6倍，完整方阵Attention位置对为$1.6^2=2.56$倍。Kernel形状、因果三角是否跳过、数据搬运、并行利用率和启动开销都会让墙钟比例偏离这两个理论主项。
3. 序列概率为$0.8\times0.5\times0.25=0.1$；总NLL为$-\log0.1=\log10\approx2.302585$，平均损失为$\log10/3\approx0.767528$，困惑度为$10^{1/3}\approx2.154435$。掩掉中间目标后，只汇总其余两个NLL，分母从3改为2；不能仍除3。
4. 温度1时，最短核集保留前两项，因为$0.60+0.25=0.85\ge0.8$；重归一化为$(12/17,5/17)$。温度0.5时概率正比于原概率的平方，第一项为$0.36/(0.36+0.0625+0.01+0.0025)=0.36/0.435\approx0.827586$，已达阈值，因而只保留第一项并重归一化为1。EOS按Token ID停止；文本停止串可能横跨Token或流式块，须保留尾部做匹配，并预先约定是否返回停止文本。

<a id="solutions-2"></a>

## 第2章答案与提示

1. Q为`[2,32,8,128]`，K/V各为`[2,8,8,128]`，在头映射后的Attention分数为`[2,32,8,8]`，合并后输出为`[2,8,4096]`。分组比$g=H/H_{\mathrm{KV}}=4$，每个KV头服务4个Query头。
2. 可见对数为$1+2+3+4=10$。改动第4个Token不应改变前三个位置的输出；否则实现泄露了未来信息。RoPE编码位置不等于限制可见性，不能取消因果掩码。
3. LayerNorm的均值为3.5，方差为0.25，输出$(-1,1)$。RMS为$\sqrt{(3^2+4^2)/2}=\sqrt{12.5}$，无仿射RMSNorm输出约$(0.848528,1.131371)$；乘$\boldsymbol\gamma$后为$(1.697056,0.565685)$。顺序Block先得$\mathbf U=\mathbf X+\operatorname{Attention}(\operatorname{Norm}(\mathbf X))$，MLP须读取残差相加后的$\operatorname{Norm}(\mathbf U)$。
4. Q/O各有$d^2=16{,}777{,}216$个参数；K/V各有$d(H_{\mathrm{KV}}d_h)=4096\times1024=4{,}194{,}304$个，GQA合计$41{,}943{,}040$。MHA的四个投影均为$d\times d$，合计$67{,}108{,}864$，GQA减少$25{,}165{,}824$。这是结构账本；质量、Kernel效率与墙钟时间还需在固定工作负载上验证。

<a id="solutions-3"></a>

## 第3章答案与提示

1. 解除绑定增加$Vd=32000\times4096=131{,}072{,}000$个参数。每参数新增$2+2+4+4=12$ bytes，共1,572,864,000 bytes，约1.464844 GiB。这里没有FP32主权重，也没有激活或工作空间。
2. 所有含$BT$但不额外含$T$的投影、MLP和全位置输出项变为2倍；$BT^2$ Attention项变为4倍。真实时间还受是否跳过因果上三角、Kernel形状、数据搬运、启动和利用率影响。
3. $\lceil2050/256\rceil=9$个Block；前8个覆盖2048个元素，最后一个只有2个有效线程。1024线程会减少Block数量，并可能增加寄存器/共享内存压力或边界浪费；应结合设备上限与测量选择。
4. 拐点$120/3=40$ FLOPs/byte。A的带宽屋顶为30 TFLOP/s，故上界30、带宽受限；B的带宽屋顶240 TFLOP/s，故上界120、计算受限。A的bytes减半使强度变为20，上界升至60 TFLOP/s；这是Roofline预测上界，不保证墙钟恰好加速2倍。

<a id="solutions-4"></a>

## 第4章答案与提示

1. 正确loss为$(12+9)/(8+3)=21/11\approx1.909091$。无权平均为$(12/8+9/3)/2=(1.5+3)/2=2.25$，它让只有3个有效Token的第二批与8个Token的第一批拥有相同批级权重。
2. 正确顺序先得未缩放范数$512/1024=0.5$，低于阈值，裁剪因子为1。忘记反缩放会按$1/512$裁剪；之后即使再除1024，有效梯度也被额外缩小512倍，不再等价于原loss。
3. $C=6\times5{,}802{,}037{,}248\times2\times10^{10}=6.9624446976\times10^{20}$ FLOPs。有效聚合吞吐为$32\times250\times10^{12}\times0.35=2.8\times10^{15}$ FLOP/s，耗时约248,658.739秒，即2.878天。未覆盖项可包括checkpoint/评估/故障停机、长序列Attention偏离$6PD$、激活重计算、padding实际工作、通信与数据等待。
4. 至少还需scheduler与loss scaler状态、各随机源RNG状态、sampler/数据shard与样本游标、数据/分词器/代码/依赖及并行拓扑版本。对照测试可让确定性小模型连续跑20步，另一条在第8步保存退出后续跑，比较第9步batch ID、学习率、loss scale及第20步模型/optimizer状态。缺optimizer、数据游标或不兼容状态而只加载权重时应称为热启动。

<a id="solutions-5"></a>

## 第5章答案与提示

1. 应用SUM Reduce-Scatter：四个64 MiB输入先逐元素求和，再让每rank保留16 MiB的不重叠分片。若每rank都需完整64 MiB和，则用SUM All-Reduce。两者都不是仅切分而不归约的Scatter。
2. $B_{\mathrm{global}}=2\times8\times4=64$条序列；最大输入Token数为$64\times2048=131{,}072$。TP切一份模型的层内张量，PP切同一份模型的层，均不制造独立样本。变长或mask训练还要汇总全DP组有效目标Token数，并用NLL总和除总有效数，不能无权平均rank loss。
3. bubble占比为$(8-1)/(16+8-1)=7/23\approx30.435\%$，相对理想时间开销为$7/16=43.75\%$。$m=32$时占比为$7/39\approx17.949\%$。增大$m$可能增加同时存活激活、调度buffer或更新延迟，并可能改变micro-batch shape与Kernel效率，仍须做显存和墙钟测量。
4. $4096/16=256$、$32/16=2$、$11008/16=688$、$32/1=32$均为整数，但$H_{\mathrm{KV}}=8$不能按16个TP rank各分至少一个KV头；朴素头分片失败。可改为TP=8、PP=2、DP=8，乘积仍为128，且每TP rank一个KV头、每stage 16层。若重放比例$q=0.5$且$C_b\approx2C_f$，训练主项算术增幅约$q/3=1/6\approx16.67\%$；这不是墙钟减速保证。

<a id="solutions-6"></a>

## 第6章答案与提示

1. 完整概率矩阵为$4096^2\times2=33{,}554{,}432$ bytes，即32 MiB。局部Tile为$64^2\times2=8192$ bytes，即8 KiB；元素数比例为$4096^2/64^2=4096$。Tile式只比较一个局部score与完整矩阵，不含Q/K/V/O、统计量、反向和临时缓冲；算法仍遍历所有稠密score，所以不能把4096当作峰值节省或FLOPs加速。
2. $m_A=2,\ell_A=e^{-2}+1\approx1.135335,z_A=e^{-2}+2\approx2.135335$；$m_B=3,\ell_B=e^{-2}+1\approx1.135335,z_B=4e^{-2}+8\approx8.541341$。共同$m=3$，$\ell=e^{-1}\ell_A+\ell_B\approx1.553002$，$z=e^{-1}z_A+z_B\approx9.326887$，$o=z/\ell\approx6.005716$。块内输出的分母与最大值尺度不同，直接平均没有全局Softmax依据。
3. 至少应：固定并记录实际后端并把fallback单列；用相同shape/dtype/mask/dropout/grad工作量；预热编译、allocator与Kernel选择；用CUDA Event或前后同步；做足够重复并报告median/P10/P90和原始毫秒；先验证输出/梯度容限；在独立、同等待遇的进程中重置峰值并报告allocated与reserved；把Profiler诊断与无Profiler计时分开；记录硬件/驱动/PyTorch版本。`nvidia-smi`进程占用不等于Tensor allocated峰值。
4. $N_{\mathrm{TB,fwd}}\approx2\times32\times\lceil8192/128\rceil=4096$；$g=32/8=4$。GQA让4个Query头共享一组K/V，但32个Query头仍各自产生score、Softmax统计量和输出。causal可跳过完整未来Tile，却产生三角边界与不均匀有效Key数；实际任务合并、边界mask、occupancy和Kernel调度决定launch与时间，不能由面积简单宣称减半。

<a id="solutions-7"></a>

## 第7章答案与提示

1. 逻辑Token数为$513+1024+2049=3586$，逻辑payload为$3586\times128$ KiB = 448.25 MiB。块数为$\lceil513/256\rceil+\lceil1024/256\rceil+\lceil2049/256\rceil=3+4+9=16$，物理payload为$16\times256\times128$ KiB = 512 MiB，末块浪费63.75 MiB。块表、引用、对齐、未分配池、workspace、权重和allocator保留池另计，所以不能推出reserved显存。
2. $(O-1)(28-18)>45$给出$10(O-1)>45$，最小整数$O=6$。2048 Token的逻辑交接payload为$2048\times128$ KiB = 256 MiB。该判断仍未计两池排队、模型副本、Batch变化和故障。
3. $\mathbb E[N]=1+0.75+0.75^2+0.75^3=2.734375$。基线生成这些Token约需$2.734375\times11=30.078125$ ms；推测轮成本为$3\times1.5+13+1=18.5$ ms，估计速度约$1.626\times$。未覆盖项可包括接受事件相关性、Target验证随$K$/Batch的非线性、采样与KV回滚开销、TTFT/尾延迟、Draft/Target后处理一致性及暂存显存。
4. offered load为$720/120=6$ requests/s，完成吞吐$660/120=5.5$ requests/s，输出吞吐$79{,}200/120=660$ tokens/s，SLO goodput$594/120=4.95$ requests/s。60条可能在途、拒绝、超时或取消；若被静默删去，会掩盖过载。长度分布决定工作量，P95/P99揭示均值/吞吐看不到的尾延迟，比较还须固定采样、资源和窗口/drain口径。

<a id="solutions-8"></a>

## 第8章答案与提示

1. 正确窗口loss为$(12+9)/(6+3)=21/9=7/3\approx2.333333$；batch mean无权平均为$(12/6+9/3)/2=(2+3)/2=2.5$。前者按全窗口assistant有效Token归一，符合F-SFT-LOSS；后者让只有3个目标Token的batch获得相同批级权重。Prompt Token不进入该assistant-only分子/分母，但仍作为因果上下文参与Attention。
2. 参数量为$8(4096+1024)=40{,}960$。每参数$2+2+4+4=12$ bytes，payload为491,520 bytes，即0.46875 MiB。把冻结Base改为4-bit存储不改变LoRA参数数目及上述LoRA状态口径；它只压缩Base码payload，实际训练仍含量化元数据、反量化/MatMul、激活、LoRA分支、workspace和allocator，故不能推出峰值或速度。
3. $a^+=-4.0-(-4.5)=0.5$，$a^-=-3.0-(-2.8)=-0.2$，所以$\Delta=0.7$。$\beta\Delta=0.14$，chosen成对概率$\sigma(0.14)\approx0.534943$，loss$=-\log\sigma(0.14)\approx0.625596$。DPO比较相对Reference的变化；尽管Policy的chosen绝对log probability仍低于rejected，它把chosen相对提高、rejected相对降低，故margin为正。
4. $\operatorname{ASR}=10/200=5\%$，危险拒答率$170/200=85\%$，良性误拒率$30/300=10\%$，良性有用率$240/300=80\%$。四项不互补，剩余输出类别也须解释。新版本必须同时报告ASR、误拒、有用性、质量切片、置信区间及固定负载下的吞吐；ASR下降不能抵消误拒翻倍，吞吐提升也不能证明内容质量或权限安全。

<a id="solutions-9"></a>

## 第9章答案与提示

1. Full为$16384\times16385/2=134{,}225{,}920$对。窗口为$2048\times2049/2+(16384-2048)\times2048=31{,}458{,}304$对，比例约23.437%。贯穿配置128 KiB/Token，所以逻辑KV为$16384\times128$ KiB = 2 GiB。它们只描述可见图与容量；训练位置、证据位置、任务难度、Tokenizer、输出预算和质量阈值共同决定有效上下文。
2. 每边$448/14=32$个Patch，故$N_v=1024$、总Token为2048。文本单独可见对为$1024\times1025/2=524{,}800$，拼接后为$2048\times2049/2=2{,}098{,}176$，约3.998倍；视觉前缀KV为$1024\times128$ KiB = 128 MiB。未计成本可包括视觉编码器/连接器、激活、workspace、图像预处理和物理填充中的任意两项。
3. 理想覆盖为$1-(1-0.25)^6=1-0.75^6\approx82.2021\%$；Decode工作量为$6\times100=600$ Token。串行延迟为$90+110+30+150+20+10=410$ ms。覆盖假设候选独立且还需Verifier正确选择；延迟未说明工具授权、结果正确、重试、副作用或质量，因此两者都不是任务成功率。
4. 8个Attention层的KV为$4\times8/32=1$ GiB。24个SSM层状态为$24\times8192\times16\times2=6{,}291{,}456$ bytes = 6 MiB，理想和为1030 MiB。模型仍有Attention层KV；8 MiB是32个SSM层的另一构造口径，本题只有24层。实际还含卷积状态、权重、workspace、对齐与可能的MoE通信，质量和速度也须实测。

<a id="solutions-10"></a>

## 第10章答案与提示

1. 绑定Embedding为$Vd=2{,}097{,}152$。每层为$2d^2+2dH_{\mathrm{KV}}d_h+3dd_{\mathrm{ff}}+2d=754{,}176$，四层加final RMSNorm后总计$5{,}114{,}112$参数。KV为$2BLTH_{\mathrm{KV}}d_hs=2{,}097{,}152$ bytes = 2 MiB。物理显存还可能含分页末块浪费/元数据、allocator保留/碎片、workspace或其他临时Tensor中的任意两类。
2. 128 MiB是按shape与dtype得到的解析逻辑中间量；RSS多110 MiB是该CPU进程与测量口径下的观测，但不是Tensor专属峰值，更不是GPU allocated/reserved。可以说该配置的理论score容量和CPU RSS观测；不能声称FlashAttention节省GPU显存或绘制GPU Flash加速/峰值对比图。
3. offered load为$720/120=6$ requests/s，完成吞吐为$660/120=5.5$ requests/s，output throughput为$79{,}200/120=660$ tokens/s，SLO goodput为$594/120=4.95$ requests/s。还需给60条未完成请求是在途、超时、拒绝还是取消，以及窗口末队列、drain时间和到达/完成随时间变化。
4. Gate 1可以在源码、配置与实际权重shape一致时PASS；Gate 2可在参数精确核对且其余资源假设明确时PASS。Gate 3只有解析设计通过，GPU正确性/性能子项为`not_run`；Gate 4因未部署而`not_run`。最终报告只能写“结构与解析阶段完成，目标训练/服务未运行”，不能标为端到端实验完成。

完成核对后，若错误集中在通用数学、单位或 Python 表达，先走[必要前置知识](prerequisites.md)的按需复习路径；若只是变量含义或口径混淆，直接查[符号表](notation.md)。修正后应回到原题重新计算，而不是以本页答案替换自己的推导。
