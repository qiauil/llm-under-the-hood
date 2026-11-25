# 拆开大模型：从 Token 到 GPU 与推理服务

<img src="doc/assets/book-cover.png" alt="《拆开大模型》封面，分层展示 Token、Transformer、GPU 与推理服务器" width="320">

一个 Token 是怎么算出来的？训练时，参数、显存和通信开销怎样分摊到多张 GPU？上线后，推理服务又怎样把生成的 Token 稳定交付给用户？

《拆开大模型》沿着这些问题，从 Transformer 的公式和张量形状讲到预训练、分布式训练、GPU Kernel 优化与推理服务，并延伸到后训练、评测和综合项目。这本中文教材由 Qiauil 与 ChatGPT 共同创作，包含十章正文、四个附录、可运行的示例脚本和配套实验记录。

## 开始阅读

- [读者导览](doc/index.qmd)提供系统入门、性能工程和专题学习三条路线。
- 想系统学习大语言模型，可以从[第 1 章 大语言模型基础](doc/chapters/01-foundations.md)开始。
- 关注训练或推理性能，可以先读[第 3 章 模型计算复杂度与 GPU 计算基础](doc/chapters/03-compute-and-gpu.md)，再读第 5–7 章。
- 关注后训练和模型评测，可以先补齐第 1–2 章的模型基础，再读第 8–9 章。

## 内容概览

| 章 | 主题 |
| --- | --- |
| 1 | [大语言模型基础](doc/chapters/01-foundations.md) |
| 2 | [Transformer 模型结构](doc/chapters/02-transformer.md) |
| 3 | [模型计算复杂度与 GPU 计算基础](doc/chapters/03-compute-and-gpu.md) |
| 4 | [预训练原理与训练流程](doc/chapters/04-pretraining.md) |
| 5 | [大规模分布式训练](doc/chapters/05-distributed-training.md) |
| 6 | [高性能 Attention 与 GPU Kernel 优化](doc/chapters/06-efficient-attention.md) |
| 7 | [大语言模型推理与高性能服务系统](doc/chapters/07-inference-serving.md) |
| 8 | [微调、对齐与评测](doc/chapters/08-post-training-evaluation.md) |
| 9 | [前沿模型技术专题](doc/chapters/09-frontiers.md) |
| 10 | [综合实践与课程项目](doc/chapters/10-projects.md) |

阅读时可按需查阅[必要前置知识](doc/appendices/prerequisites.md)、[统一符号表](doc/appendices/notation.md)、[习题答案与提示](doc/appendices/exercise-solutions.md)和[参考资料](doc/appendices/references.md)。实践时可结合[示例脚本](doc/examples/)和[实验记录](doc/experiments/)，按各章的说明运行示例并核对结果。

## 本地阅读与 PDF

按[安装指南](https://quarto.org/docs/get-started/)安装 Quarto 后，在仓库根目录运行以下命令，在浏览器中阅读。

```bash
quarto preview doc
```

如需离线阅读，可生成 PDF。首次构建且尚未安装 TeX 时，先运行 `quarto install tinytex`。

```bash
quarto render doc --to pdf
```

生成的 PDF 位于 `doc/_book/`。

## 许可证

- 教材正文、附录和原创插图采用 [CC BY 4.0](LICENSE)。
- `doc/examples/` 下的原创示例代码和实验脚本采用 [MIT](doc/examples/LICENSE)。

第三方材料遵循其各自的许可条款。
