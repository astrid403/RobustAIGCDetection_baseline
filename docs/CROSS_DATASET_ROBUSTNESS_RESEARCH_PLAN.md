# 跨数据集 AIGC 图像检测鲁棒性研究计划

## 0. 文档边界

- 状态：计划外、只读研究例外；不属于 Part 11，也不改变 Part 11–12。
- 基线快照：分支 `feature/defactify-external-eval`，审阅时 HEAD
  `cedc2a18d956948acfed87d575d41d4b93689d1d`。
- 本文只提出下一研究分支的候选方案，不授权实现、训练、重新评估或
  Defactify 访问。
- `docs/FINAL_RESULTS_FREEZE.md` 和
  `artifacts/number_freeze_v2_manifest.json` 中的 Protocol v2 模型、数字、
  图表、结论与哈希保持不变。
- 如果后续获批实施，应从当前冻结版本新建独立研究分支，并建立新协议和
  新运行目录；不得覆盖现有 Part 11–12 流程或复用已有正式结果目录。

## 1. 执行摘要

当前结果不是“A1 无效”，而是一个明确的域泛化反例：将 CLIP ViT-B/32
的全局图像表示从 final 换成 penultimate，能改善 GenImage 内部的未见生成器
和 JPEG/resize/blur 鲁棒性，但不能解决跨数据集的场景、采集链、真实图像
来源和生成器族变化。A1 相对 B2 的 AUROC 差异为：

- GenImage unseen：`+0.004123`，95% CI `[0.002493, 0.006033]`；
- Defactify full：`-0.013021`，95% CI `[-0.016487, -0.009541]`；
- Defactify balanced：`-0.013711`，95% CI `[-0.017987, -0.009504]`。

这说明仅改变相邻 CLIP 层，或把 final 与 penultimate 全局向量拼接，主要是在
同一个语义特征族内重新分配信息，并没有引入真正互补、跨域稳定的取证信号。

最值得尝试的三个方案是：

1. **主方案：S1，语义去偏的局部 NPR 分支 + 冻结 B2 的验证集级后融合。**
   用 patch shuffle/crop 抑制场景语义捷径，用 Neighboring Pixel
   Relationships（NPR）突出上采样造成的局部像素依赖；B2 只作为冻结的语义
   专家，二者在 GenImage 开发协议内做受约束的概率融合。这不是
   final+penultimate fusion。
2. **S2，RINE-lite 多中间层 CLS 表示 + 学习式层权重 + 监督对比约束。**
   使用多个相隔较远的 CLIP block，而非最后两层拼接；冻结 encoder，仅训练
   小型投影、层重要性和分类头。
3. **S3，SPAI-lite 真实频谱自监督异常分数 + 轻量判别头。**
   只用真实训练图像学习频谱重建的一致模式，使未知生成器作为异常被检测；
   它是最直接的跨生成器不变性假设，但工程和验证风险高于 S1/S2。

推荐先做 S1；备用方案为 S2。S3 作为第二阶段研究储备，不建议在第一个短周期
中直接替代主方案。

## 2. 当前仓库与实验事实

### 2.1 已实现模型

| 代号 | 实现 | 可训练参数 | 关键信号 |
|---|---|---:|---|
| B0 | ImageNet ResNet50 端到端 | 大部分 backbone | RGB 纹理与语义 |
| B1 | frozen CLIP final + linear | 513 | 单个全局 final 向量 |
| B2 | frozen CLIP final + 512-hidden MLP | 263,169 | 单个全局 final 向量 |
| A1 | frozen CLIP penultimate + 同规格 MLP | 263,169 | 单个全局 penultimate 向量 |
| P | normalized final+penultimate concat + MLP | 262,657 | 两个相邻全局向量 |

CLIP encoder 是 OpenAI ViT-B/32，输入 224，final/penultimate 均为归一化
512-D 全局 embedding。P 仅对两个向量分别 L2 normalize 后拼接，没有 patch
token、频域、残差、注意力、门控或互补损失。

### 2.2 数据与训练约束

- GenImage train：3,000 张，来自 Stable Diffusion V1.5、ADM、BigGAN；
- GenImage validation：1,200 张，同三个生成器；
- GenImage unseen：4,000 张，来自 GLIDE、Midjourney、VQDM、Wukong；
- 正式训练：10 epochs、batch 32、AdamW、`lr=1e-3`、BCE、
  validation-AUROC 选 checkpoint、阈值固定 0.5；
- 训练期 CLIP 图像预处理来自 `open_clip`，没有显式 JPEG、resize、blur、
  noise 或强语义破坏增强；
- 正式硬件记录为 RTX 5060 Laptop GPU，8,151 MiB；冻结 CLIP 特征后的头部
  训练约一分钟，说明特征缓存是当前仓库最有价值的低成本路径。

### 2.3 已确认的结果边界

- A1 在内部未见生成器的 AUROC/AUPRC 提升有统计支持；
- A1 在 JPEG 70、resize 0.5、Gaussian blur 1.0 下的 AUROC 均高于 B2；
- A1 在 Defactify full/balanced 的 AUROC/AUPRC 退化有统计支持；
- A1 仅在 Defactify 的 DALL-E 3 子组改善，其他四个生成器下降；
- P 没有超过 A1，证明简单增加同族全局特征维度不是答案；
- 所以新的方案必须改变“学习信号或数据不变性”，不能只换最后一个 CLIP
  block 或再次做相邻层拼接。

## 3. 跨数据集退化的可能原因

### 3.1 语义与数据来源捷径

CLIP 的设计目标是图文语义对齐。当前分类头看到的是整图全局向量，且训练集
只有三个生成器。模型很容易把场景类别、构图、ImageNet 来源、分辨率或编码
格式与真假标签相关联，而非学习生成过程的稳定痕迹。

GenImage 的 real/fake 往往在同一生成器目录和固定收集流程内构造；Defactify
则改变了真实图来源、生成器集合、场景分布、类别比例和存储后端。NeurIPS
2024 的 *Breaking Semantic Artifacts* 专门证明，跨场景退化可来自 real 与
fake 两侧的“semantic artifacts”，patch shuffle 加局部分类能够改善跨场景
与开放世界泛化。[论文原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html)
和[官方代码](https://github.com/Zig-HS/FakeImageDetection)均已公开。

### 3.2 全局池化丢失局部取证线索

ViT-B/32 在 224 输入下只有较粗的 patch 网格，最后又池化成 512-D 全局向量。
上采样邻域、局部噪声、边缘振铃、细粒度频谱等取证线索可能在 patch embedding、
深层语义聚合和全局池化中被削弱。A1 略早一层所以在同域退化下更稳，但仍然是
单个全局语义向量。

CVPR 2024 的 NPR 将生成网络上采样造成的局部像素依赖显式化，并在 GenImage
官方训练示例中支持单生成器训练、多个未知生成器测试。
[官方代码](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)。

### 3.3 生成器多样性不足与经验风险最小化

当前 3,000 张训练数据同时覆盖 GAN（BigGAN）、像素空间 diffusion（ADM）和
latent diffusion（SD1.5），方向是对的，但每个生成器每类约 500 张，且普通
BCE 把所有样本混合优化。模型可能由最容易的生成器或场景主导，没有显式优化：

- leave-one-generator-out 风险；
- worst-generator 风险；
- 同图不同退化的一致性；
- 类内跨生成器紧致性。

因此很高的同源 validation AUROC（约 0.997）并不代表跨域模型选择可靠；
validation 与 train 共享生成器，使生成器/场景捷径不会被惩罚。

### 3.4 “鲁棒于退化”不等于“鲁棒于数据域”

JPEG、resize、blur 主要改变图像传输链；Defactify 还改变内容语义、真实图像
相机/网络来源、生成器年代、类别比例和潜在后处理。A1 的退化鲁棒性提升说明
其特征对这些固定扰动更稳定，但没有证据表明其去除了数据集身份。

### 3.5 训练目标只要求可分，不要求可迁移

BCE + validation AUROC 会奖励训练域内任何有效线索。当前没有 paired
consistency、supervised contrastive、domain/group DRO 或 real-distribution
one-class objective。RINE 使用多 block CLS、可学习层重要性和监督对比损失，
强调类内紧致；其论文报告在 20 个测试集上的平均改善，并且冻结 CLIP、训练
周期短。[论文原文](https://arxiv.org/abs/2402.19091)；
[官方代码](https://github.com/mever-team/rine)。

### 3.6 单一证据源的不可避免盲区

语义特征可跨压缩，但会受场景偏差影响；像素/频域线索更贴近生成过程，却可能
被强 JPEG、resize 或新型生成器抹去。因此多专家只有在“信号互补、验证协议
隔离、融合自由度受限”时才有价值。P 的失败不否定多特征融合，它只否定了
两个高度相关的相邻全局 CLIP 表示的简单拼接。

## 4. 文献与官方实现结论

### 4.1 直接相关的高质量工作

| 工作 | 核心思想 | 对本项目的启示 | 适配判断 |
|---|---|---|---|
| UniversalFakeDetect, CVPR 2023 | 使用未为真假任务过拟合的大规模 CLIP 特征；线性/近邻分类 | 支持保留 B2 作为语义专家，但项目中的 B1 说明线性头不足 | 已有基础 |
| RINE, 2024 | 多个 CLIP block 的 CLS，经投影、层权重和监督对比学习 | 不同于 P 的“两层拼接”；能学习真正的层选择 | 高 |
| NPR, CVPR 2024 | 邻域像素关系突出上采样局部依赖 | 与全局 CLIP 互补、输入变换简单 | 高 |
| Breaking Semantic Artifacts, NeurIPS 2024 | patch shuffle 破坏语义捷径，patch-based classifier | 直接对应 GenImage→Defactify 的跨场景问题 | 高 |
| AIDE, ICLR 2025 | CLIP 语义 + 高频/低频 patch 低层特征 | 支持“语义专家 + 局部取证专家”，而非 CLIP 层堆叠 | 中 |
| SPAI, CVPR 2025 | 真实频谱是较稳定模式；masked spectral reconstruction | one-class 思路避免拟合少数生成器 | 中 |
| DIRE, ICCV 2023 | diffusion reconstruction error | 未见 diffusion 检测依据强，但推理昂贵且偏 diffusion | 低/中 |
| LaRE², CVPR 2024 | latent reconstruction error + feature refinement | 比像素 DIRE 高效，但仍需 diffusion/VAE 依赖 | 中低 |
| DRCT, ICML 2024 Spotlight | 原图/重建图的对比训练，大规模 DRCT-2M | generator diversity 很强，但数据与训练规模不适合首轮 | 低 |
| FatFormer, CVPR 2024 | forgery-aware adapter + language-guided alignment + wavelet | 完整方法较重、依赖旧环境及额外 wavelet 包 | 中低 |

主要来源：

- [UniversalFakeDetect 论文](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html)
  与[作者项目页/官方代码入口](https://utkarshojha.github.io/universal-fake-detection/)；
- [NPR 官方代码](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)；
- [RINE 论文](https://arxiv.org/abs/2402.19091)与
  [RINE 官方代码](https://github.com/mever-team/rine)；
- [Breaking Semantic Artifacts 论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html)
  与[官方代码](https://github.com/Zig-HS/FakeImageDetection)；
- [AIDE 论文](https://arxiv.org/abs/2406.19435)；
- [SPAI 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html)
  与[项目页](https://mever-team.github.io/spai/)；
- [DIRE 论文与官方代码入口](https://arxiv.org/abs/2303.09295)；
- [LaRE² 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_LaRE2_Latent_Reconstruction_Error_Based_Method_for_Diffusion-Generated_Image_Detection_CVPR_2024_paper.html)；
- [DRCT 论文](https://proceedings.mlr.press/v235/chen24ay.html)与
  [官方代码](https://github.com/beibuwandeluori/DRCT)；
- [FatFormer 官方代码](https://github.com/Michel-liu/FatFormer)。

### 4.2 对 DINO/DINOv2 的判断

DINO/DINOv2 是有吸引力的独立预训练视觉特征，可作为“换预训练目标”的
backbone 对照；但本轮检索到的直接跨生成器检测证据与成熟官方实现，明显弱于
CLIP-RINE、NPR、语义破坏和频谱建模。简单把 CLIP 512-D 换成 DINO 全局 CLS
仍可能重复“全局语义向量 + 小头”的问题。因此：

- 不把 DINO 全局特征替换列入前三；
- 若后续资源允许，可把 frozen DINOv2-small patch-token statistics
  （CLS + patch mean/std，而非只用 CLS）作为 S1/S2 的独立 backbone 消融；
- 不允许根据 Defactify 表现选择 CLIP 或 DINO。

### 4.3 对重建误差路线的判断

DIRE、LaRE²、DRCT、FIRE 都说明重建误差可能跨 diffusion generator，但它们
通常需要 Stable Diffusion/VAE/UNet、多步反演、重建数据或更复杂训练。当前
仓库没有 `diffusers`、预训练扩散权重管理或重建缓存协议。直接引入会扩大：

- 依赖和下载体积；
- GPU 显存与推理时间；
- 缓存/哈希/版本控制面；
- “只检测 diffusion，牺牲 BigGAN 等 GAN”风险。

因此首轮不选 diffusion reconstruction；只有 S1/S2 均失败且用户接受更高成本
时，才评估 LaRE²/FIRE，而不是从 DIRE 的完整多步像素重建开始。

## 5. 三个最值得尝试的方案

## 5.1 S1（推荐）：Semantic-Broken NPR + Frozen-CLIP Late Fusion

### 核心设计

建立两个有明确分工的专家：

1. **语义专家**：冻结的 B2 final-CLIP MLP。首个实验可直接使用同一新协议下
   重新训练的 B2 control；结构保持简单。
2. **局部取证专家**：
   - 对 RGB 计算 NPR/邻域差分表示；
   - 训练时以固定概率做 patch shuffle，使标签不依赖完整场景布局；
   - 使用轻量 ResNet18 或 4-stage 小 CNN，输出整图 logit；
   - clean 与 JPEG/resize/blur 视图共享标签，可加 logit consistency。
3. **后融合**：只在 GenImage 开发验证上拟合一个受约束标量
   `p = α p_clip + (1-α) p_npr`，例如预注册
   `α ∈ {0, 0.25, 0.5, 0.75, 1}`；不训练高容量联合融合器。

首个最小版本只做 NPR + patch shuffle + 固定 `α=0.5`，先证明两专家错误互补；
一致性损失和 α 搜索属于预注册消融，不能看到外部测试后追加。

### 理论依据

- NPR 针对生成器上采样形成的局部像素依赖，减少场景内容主导；
- patch shuffle 明确破坏全局语义和布局相关性，逼迫取证专家依赖局部证据；
- CLIP 分支保留在强压缩或局部痕迹消失时仍有价值的高层先验；
- 受约束 late fusion 避免高容量网络重新学习数据集捷径，并可解释每个专家的
  独立贡献。

### 与 P 的本质区别

P 融合的是同一个 CLIP encoder 的相邻层、同一种预处理、同一个全局池化空间。
S1 融合的是 RGB 语义 embedding 与显式邻域差分/局部伪影，两者输入域、归纳
偏置和失败方式不同；融合发生在校准后的 logits，而不是特征拼接。

### 实现难度与成本

- 难度：中；
- 新依赖：首选零新增；NPR 可用 PyTorch tensor slicing 实现；
- 显存：ResNet18、224、batch 32 预计可在 8GB 内；必要时 batch 16；
- 时间：3,000 张数据的单 seed pilot 预计分钟级到十几分钟级，而非小时级；
- 推理：约为 B2 CLIP + 一个 ResNet18；可缓存 CLIP，NPR 分支需逐图推理。

### 主要风险

- JPEG/resize 会削弱邻域差分；
- patch shuffle 过强会制造训练伪影；
- NPR 更擅长含上采样操作的生成器，对某些 autoregressive/新 diffusion
  generator 不一定稳定；
- 融合可能只是平均两个相关错误，收益有限。

### 预期收益

不预承诺具体外部数字。合理预期是：相对 B2，GenImage 的 leave-one-generator-
out（LOGO）平均/最差 AUROC 提升，且 NPR 与 CLIP 的错误一致率显著低于
B2/A1；若这些条件成立，跨数据集 AUROC 有机会获得约 1–3 个百分点的实际
改善。该范围是研究目标，不是结果声明。

## 5.2 S2（备用）：RINE-lite 多中间层 CLS + 层权重 + SupCon

### 核心设计

- 冻结 OpenAI CLIP ViT-B/32；
- 一次 forward 取相隔较远的 block CLS，例如 `{3, 6, 9, 12}`；
- 每层共享小投影（如 768→128），用跨层 softmax importance 做加权和；
- 分类头为 128→1；
- 损失为 BCE + 小权重 supervised contrastive；
- 可缓存 `[N, 4, 768]`，缓存元数据必须记录 block IDs、提取版本、preprocess
  和 split hash。

block 集合必须在实验前预注册，不能逐层用 GenImage unseen 或 Defactify 搜索。
首选 `{3,6,9,12}` 是覆盖浅/中/深语义阶段的结构性选择，不是结果驱动选择。

### 理论依据

RINE 证明多个中间 block 的 CLS 及学习式重要性比只取最终特征更适合 SID，
且监督对比损失可增强类内紧致。它与 A1/P 的差别在于：

- 覆盖四个远距离阶段，而非最后两个高度相关阶段；
- 先做共享低维投影，再按维度学习层权重；
- 加入表示空间约束，而非仅靠 BCE；
- encoder 冻结，仍保持低训练成本。

### 实现难度与成本

- 难度：中低；
- 可训练参数：目标 `<1M`，显著低于完整 RINE 的约 6.3M；
- 缓存：约为当前单层缓存的 4–6 倍，3,000/1,200 张规模仍可接受；
- 特征提取：一次 ViT forward 保存四层，不应做四次独立 forward；
- 训练：预计分钟级；batch 可在缓存上使用 64–128。

### 主要风险

- 当前 A1 的 Defactify 退化提示 CLIP layer choice 本身可能不是根因；
- 小训练集可能使层重要性过拟合；
- SupCon 对 batch 类别/生成器组成敏感；
- 若实现仍只利用 CLS，可能继续丢失局部 patch 痕迹。

### 预期收益

预期首先改善 LOGO worst-generator 与 seed 稳定性，而非追求同源 validation
上限。若多层重要性在 seeds/LOGO folds 中稳定、且优于 B2/A1/P，可合理期待
较低风险的跨域增益；否则应明确判为“同族 CLIP 特征仍不够”并停止。

## 5.3 S3：SPAI-lite 真实频谱自监督异常检测

### 核心设计

- 只用 GenImage train 中的真实图；
- 对 log-magnitude FFT 做固定大小的径向/局部频谱表示；
- 训练小型 masked spectral autoencoder 重建被遮蔽频谱；
- 推理特征包括多频带重建余弦相似度、MAE 和径向统计；
- 首版用 logistic regression/小 MLP 输出真假分数；
- 不与 CLIP 联合训练；只有独立通过开发门槛后，才允许预注册 0.5/0.5
  score ensemble 消融。

完整 SPAI 包含 any-resolution spectral context attention；“lite”版本先验证
one-class 频谱假设，不宣称复现论文完整 SOTA。

### 理论依据

SPAI 的核心假设是：真实图像频谱分布比不同生成器的伪影更稳定。学习真实分布
再把 fake 当 OOD，可降低对已见 generator fingerprint 的依赖。这直接针对
generator diversity 不足，也避免用 fake generator ID 做模型选择。

### 实现难度与成本

- 难度：中高；
- 显存：小型频谱 autoencoder 可控制在 8GB 内；
- 训练：预计几十分钟级，但需额外处理任意分辨率、FFT 数值范围和 mask；
- 工程：需新的频谱缓存、重建指标与单类训练路径。

### 主要风险

- 数据集的 JPEG 格式、原始尺寸和采集流程本身会形成频谱捷径；
- resize/JPEG 可能破坏真假频谱差异；
- 仅约 1,500 张真实训练图，真实分布覆盖不足；
- lite 实现可能无法继承完整 SPAI 的收益；
- 频谱异常分数的阈值校准较难，但不能在外部测试上校准。

### 预期收益

若真实频谱模型在 LOGO 和多退化验证中稳定，它可能比判别式 fake fingerprint
更适合未知生成器；但首轮成功概率低于 S1/S2。其最大科研价值是提供独立的
one-class 对照，即使失败也能判断当前跨域问题是否主要来自格式/频谱偏移。

## 6. 推荐顺序

### 主方案：S1

选择 S1 的理由：

1. 直接对应当前最可能的失败机制：语义捷径 + 全局池化；
2. NPR 和 patch shuffle 均有公开论文/官方代码与跨生成器或跨场景证据；
3. 与 B2 信号互补，明确避开已经失败的 final+penultimate fusion；
4. 无需扩散模型、大规模新数据或重依赖；
5. 可以先独立验证取证分支，再决定是否融合，失败定位清晰；
6. 8GB GPU 和 3,000 张训练规模下可快速闭环。

### 备用方案：S2

如果 S1 的 NPR 分支在 JPEG/resize 下崩溃，或与 B2 错误高度相关，则转 S2。
S2 最大限度复用当前已验证的 CLIP feature/cache 基础设施，工程风险最低；
但由于 A1/P 已暴露“CLIP 层变化未必跨域”的风险，故不作为主方案。

### 暂不首选

- 直接换 DINO 全局 CLS；
- 再做 final+penultimate concat/gating/attention；
- 完整 FatFormer/AIDE；
- DIRE/DRCT/FIRE 全重建管线；
- 依靠大量新生成器数据暴力扩充；
- 看过 Defactify 后调 α、block、增强强度、阈值或 checkpoint。

## 7. 公平训练和评估协议

## 7.1 数据角色

后续新分支先冻结一份 Protocol v3，建议角色如下：

| 数据 | 角色 | 禁止用途 |
|---|---|---|
| GenImage 三个训练生成器 | 模型训练；构造 LOGO folds | 不报告成外部测试 |
| GenImage 同生成器 validation | epoch/checkpoint；阈值仅在确有必要时校准 | 不单独证明泛化 |
| 三生成器 LOGO | 方案/消融选择的主依据 | 不与训练 fold 混样本 |
| GenImage unseen（已看过） | 冻结候选后的内部诊断 | 不再用于架构/超参选择 |
| 可选新开发外部集 | 仅当预先命名、固定版本和角色时做一次开发验证 | 不可与 Defactify 混用 |
| Defactify full | 最终一次性盲测 | 任何调参、选模、融合或阈值选择 |
| Defactify balanced | 从 full predictions 按冻结 ID 派生 | 第二次推理或调参 |

由于现有团队已看过 GenImage unseen 与 Defactify，严格意义上二者都不能再作为
“未见开发集”。新方案必须仅由论文、训练数据和 LOGO 证据设计；Defactify
只能回答“预注册方案是否成功”，不能决定下一变体。

## 7.2 LOGO 选择协议

对 `{SD1.5, ADM, BigGAN}` 做三折 leave-one-generator-out：

- 每折：两生成器 train，一个生成器 validation；
- real 与 fake 均按 generator/source 规则进入相同 fold，避免只 hold out fake；
- 同一原图或近重复必须 group 到同一 fold；
- 主选择指标：三折 AUROC 平均值；
- 安全指标：三折 AUROC 最小值；
- 次指标：AUPRC、balanced accuracy、real/fake recall；
- 退化指标：clean、JPEG 70、resize 0.5、blur 1.0 的平均及 worst AUROC；
- 每个方案先 seed 42 pilot，通过后才做 seeds 42/43/44。

若 real 图无法可靠按 generator/source 配对，必须先设计 source-aware group split；
不能把共享来源 real 随机分散到 train/validation。

## 7.3 比较矩阵

最小公平矩阵：

| 模型 | 目的 |
|---|---|
| B2-v3 | 同数据、同协议的主 baseline |
| A1-v3 | 验证已知 layer trade-off 是否复现；非必需部署候选 |
| S1-NPR-only | 检验局部取证信号独立价值 |
| S1-CLIP+NPR | 检验互补融合 |
| S1 无 patch shuffle | 语义去偏消融 |
| S1 无 degradation consistency | 退化一致性消融（若主版使用） |

不要同时开展大量 backbone/head 搜索。S2 仅在 S1 No-Go 后进入下一独立、预注册
阶段；不能把所有结果放进同一个 winner-takes-all 搜索。

## 7.4 预算与选择纪律

- 相同训练样本、fold、seed、图像尺寸和 checkpoint 规则；
- 每个候选同等 epoch 或同等 optimizer-step budget；
- 主指标预注册为 LOGO mean AUROC，tie-break 为 worst-fold AUROC；
- 阈值 0.5；若模型分数不可比，需要 calibration，只能在 GenImage validation
  拟合一次并冻结；
- 融合 α 只在 LOGO out-of-fold predictions 上确定；
- 正式训练固定 seeds `[42,43,44]`，报告所有 seed、均值和样本标准差；
- 不删除失败 seed，不用最佳 seed；
- 外部统计使用 matched sample IDs 的 paired hierarchical bootstrap；
- 除 AUROC/AUPRC 外必须报告 balanced accuracy、macro-F1、两类 recall、
  per-generator 和 worst-generator；
- 保存 config、commit、seed、命令、退出状态、checkpoint hash、prediction hash、
  split hash 和 feature-cache contract。

## 7.5 防泄漏要求

- 禁止读取 Defactify predictions 来分析新方案应如何设计；
- 禁止用已知 Defactify generator 子组差异选择 NPR 权重或 CLIP block；
- 禁止以 Defactify full 或 balanced 选择 checkpoint/threshold/calibration；
- 看到外部结果后，只有两种动作：报告成功；或报告失败并结束该预注册路线；
- 若要开展下一路线，必须新写方案与新实验 ID，且不能声称 Defactify 仍是盲测。

## 8. 分阶段实施计划

### Phase 0：新分支与 Protocol v3 冻结

只做文档、split audit 和代码设计：

- 从冻结提交新建独立研究分支；
- 固定数据角色、LOGO grouping、候选架构、增强概率、损失权重、seed、预算；
- 记录当前 Defactify 已暴露，定义其为“一次性外部确认集”而非选模集；
- 定义新 cache schema，禁止读取 v2 cache 作为错误 feature mode。

停止点：协议、split hash 和候选规格由用户批准。

### Phase 1：S1 最小实现与正确性测试

- 实现 NPR tensor transform；
- 实现 deterministic patch shuffle（训练随机、验证关闭或按预注册策略）；
- 实现小型 NPR expert；
- 实现 out-of-fold late fusion；
- 单图和 tiny batch 测试 shape、label、determinism、无路径泄漏；
- 只做 synthetic tensor / 极小本地图像 smoke，不训练正式模型。

停止点：正确性 Gate。

### Phase 2：单 seed LOGO pilot

- B2-v3、NPR-only、CLIP+NPR 使用完全相同 folds；
- 一次性跑 seed 42；
- 输出 fold-level、mean、worst、退化和错误互补指标；
- 不访问 GenImage unseen 或 Defactify。

停止点：Development Go/No-Go Gate。

### Phase 3：受控消融与三 seed 确认

仅当 Phase 2 Go：

- 跑预注册的 no-shuffle 和 no-consistency 消融；
- 冻结最终 S1 规格；
- B2-v3 与 S1 跑 seeds 42/43/44；
- 检查 seed 方差、fold 权重稳定性和 expert error complementarity。

停止点：Model Freeze Gate。

### Phase 4：内部诊断与一次性外部评估

- 先对已冻结模型运行 GenImage unseen，明确它不是选模集；
- 用户再次批准后，Defactify full 每个 seed 只推理一次；
- balanced 从 full predictions 按 manifest 派生；
- 计算 per-generator、worst-generator、paired bootstrap；
- 无论结果好坏，停止，不调整方案。

停止点：External Scientific Go/No-Go；不是调参 gate。

### Phase 5：若 S1 Development No-Go

- 不运行外部测试；
- 记录 S1 失败原因；
- 单独预注册 S2 block IDs、projection、SupCon 权重和预算；
- 重复 Phase 1–4；
- S2 仍失败则停止，是否开展 S3 由用户另行批准。

## 9. 预计需要修改或新增的代码文件

以下仅是后续实施范围，本轮不修改。

### S1

建议新增：

- `models/npr_detector.py`：NPR 表示与轻量 detector；
- `models/late_fusion.py`：受约束 score fusion/calibration；
- `training/group_samplers.py`：class/generator-balanced batch；
- `training/objectives.py`：clean/degraded consistency（若获批）；
- `data_pipeline/make_cross_dataset_splits.py`：source-aware LOGO manifests；
- `evaluation/evaluate_cross_dataset.py`：统一模型/ensemble 输出；
- `analysis/cross_dataset_analysis.py`：fold、worst-generator、互补错误和 CI；
- `tests/test_npr_detector.py`；
- `tests/test_cross_dataset_protocol.py`；
- `configs/research_v3/` 下的 smoke/pilot/final configs。

预计修改：

- `training/train.py`：注册 NPR/paired-view 训练路径；
- `training/engine.py`：支持双视图或 group-aware metrics；
- `data_pipeline/csv_image_dataset.py`：在不破坏旧 API 的前提下返回 group metadata；
- `evaluation/metrics.py`：如缺少 fold/worst-group 聚合，仅扩展不改旧定义；
- `models/__init__.py`。

### S2

建议新增：

- `models/rine_lite_detector.py`；
- `training/contrastive_losses.py`；
- `tests/test_rine_lite_features.py`。

预计修改：

- `models/clip_mlp_detector.py`：增加显式多 block CLS extraction API；
- `data_pipeline/fingerprints.py`：block IDs 与 extraction version 进入 cache signature；
- `training/train.py`、`evaluation/evaluate.py`：注册新模型类型。

### S3

建议新增：

- `models/spectral_autoencoder.py`；
- `features/spectral_features.py`；
- `training/train_spectral_oneclass.py`；
- `evaluation/evaluate_spectral.py`；
- `tests/test_spectral_features.py`。

旧的正式 configs、checkpoints、predictions、冻结 CSV、final assets 和 manifests
一律不改。

## 10. 明确的 Go/No-Go 标准

## 10.1 Correctness Gate

**Go** 必须全部满足：

- NPR/多层/频谱 feature contract 有 shape、finite、determinism 测试；
- train/validation sample IDs、近重复 group 和 generator/source 无泄漏；
- cache signature 能区分模型、feature mode、block、transform、split；
- fusion 只使用 out-of-fold validation predictions；
- 参数量、FLOPs/时间和显存可记录；
- smoke 不读取 Defactify。

任一不满足即 **No-Go/repair**，不得进入 pilot。

## 10.2 S1 Development Gate（不看 GenImage unseen/Defactify）

相对同协议 B2-v3，seed-42 三折 LOGO：

**Go**：

- mean AUROC 至少 `+0.005`；
- worst-fold AUROC 不低于 B2-v3，容忍度最多 `-0.002`；
- 三个 fold 至少两个改善；
- JPEG/resize/blur 的平均 AUROC 不低于 B2-v3，或 clean 的增益足以形成预注册
  的明确 trade-off；
- CLIP 与 NPR 错误有可测互补性：fusion 优于两个单专家中的较强者至少
  `+0.002` AUROC；
- 无类别坍塌，real/fake recall 均不低于 0.70。

**No-Go**：

- mean 改善 `<0.005`；
- 收益只来自一个 fold；
- worst-fold 下降超过 0.002；
- fusion 不如强单专家；
- 结果依赖 patch-shuffle artifact 或数据路径/格式泄漏。

阈值是预注册的工程决策标准，不是对外结果保证。

## 10.3 Three-seed Model Freeze Gate

**Go**：

- 三 seed LOGO mean AUROC 的 S1−B2 平均差至少 `+0.005`；
- paired 95% CI 下界 `>0`，或在样本量不足时三个 seed 差值均为正；
- worst-fold 平均不下降超过 0.002；
- α、expert 权重和 fold 排名跨 seed 稳定；
- 资源成本不超过 B2 推理时间的约 2.5 倍、峰值显存低于 8GB；
- 所有配置、哈希、seed 与失败记录完整。

否则 **No-Go S1**，不得访问 Defactify；按批准流程转 S2 或停止。

## 10.4 External Scientific Gate

此 gate 只判断科研假设是否成立，不用于选模：

**Go / 支持跨数据集提升**：

- Defactify full AUROC 相对 B2-v3 平均提高至少 `+0.010`；
- paired hierarchical bootstrap 95% CI 下界 `>0`；
- balanced AUROC 同方向；
- 五个 generator 中至少四个 AUROC 不下降超过 0.005；
- worst-generator AUROC 不低于 B2-v3；
- balanced accuracy 或两类 recall 不出现严重代价。

**Partial Go / 有限提升**：

- full/balanced AUROC 正向但 CI 跨零，或只改善部分 generator；
- 必须表述为趋势/子组 trade-off，不能声称实现稳定跨数据集提升。

**No-Go / 受控负结果**：

- full 或 balanced AUROC 明确下降；
- 收益集中在单一 generator；
- worst-generator 明显下降；
- 依赖阈值变化才能显示优势。

No-Go 后不得调 α、换 block、加增强、重选 checkpoint 或重跑 Defactify。

## 11. 最小实验闭环与停止原则

最短且信息量最大的闭环是：

1. 冻结 Protocol v3 和 LOGO；
2. 实现 S1 的 NPR-only 与固定 0.5 late fusion；
3. seed-42 LOGO 一次 pilot；
4. 若通过，做预注册消融和三 seed；
5. 冻结模型；
6. GenImage unseen 内部诊断；
7. 用户批准后一次性 Defactify；
8. 报告成功、有限提升或负结果，然后停止。

该流程优先回答三个可证伪问题：

- 抑制语义布局后，跨生成器验证是否改善？
- NPR 与 CLIP 的错误是否真正互补？
- 开发域改善在完全冻结后是否迁移到外部数据集？

若任一关键问题为否，不继续堆叠模块。这样可在短时间内得到清晰、可解释且没有
测试集泄漏的研究结论。

## 12. 后续逐任务执行总则

本节把上述研究路线拆成可以逐次批准的原子任务。它是未来独立研究分支的执行
清单，不改变当前 `docs/codex_plan/MASTER_PLAN.md`、`STATUS.md` 或 Part 11–12。

### 12.1 每次执行前的强制读取顺序

未来每次收到“执行下一个 Task”或“执行 Task XX”的请求时，执行者必须在修改
文件或运行实验前完整读取：

1. 仓库根目录 `AGENTS.md`；
2. `docs/codex_plan/MASTER_PLAN.md`；
3. `docs/codex_plan/STATUS.md`；
4. `STATUS.md` 指向的当前 Part 文件；
5. `docs/FINAL_RESULTS_FREEZE.md`；
6. `artifacts/number_freeze_v2_manifest.json`；
7. 本文 `docs/CROSS_DATASET_ROBUSTNESS_RESEARCH_PLAN.md`；
8. 本文中当前 Task 的完整小节；
9. 当前 Task 直接涉及的代码、配置、测试和上一个 Task 的交接产物。

随后必须核对：

- 当前分支、HEAD、tracked changes 和全部 untracked files；
- 当前 Task 的前置 Task 是否已经完成并获用户批准；
- 是否存在正在运行的训练/评估进程；
- 当前 Task 允许修改的文件是否与用户已有改动冲突；
- 所需数据、checkpoint、cache 和磁盘空间是否存在；
- 本 Task 是否会访问 GenImage unseen 或 Defactify。

发现冲突、前置条件不满足、目标分支错误或冻结资产可能被覆盖时，立即停止并
报告；不得猜测或自动修复。

### 12.2 单任务纪律

- 每个用户请求最多执行一个 Task；不得自动开始下一个 Task。
- 没有用户明确批准，不得从 Gate 自动进入后续 Task。
- 每个 Task 只能修改其“允许文件”列出的文件。
- smoke、pilot、ablation、final 必须使用不同 experiment name 和目录。
- 不覆盖已有运行目录，不删除失败运行，不隐藏失败 seed。
- 大型数据、checkpoint、cache、prediction 和日志不进入 Git。
- 默认不 commit、push、merge；只有用户明确要求时才进行。
- Task 状态只在未来独立研究分支中更新；当前冻结分支保持 Part 11 状态。
- Defactify 相关 Task 必须单独得到即时批准，早期批准不能自动沿用。

### 12.3 任务状态枚举

- `pending`：尚未开始；
- `in_progress`：当前请求正在执行；
- `blocked`：存在明确阻断条件；
- `awaiting_approval`：Task 已完成，位于用户批准 Gate；
- `completed`：证据完整且已获所需批准；
- `skipped_by_gate`：由于前一 Gate 的 No-Go，按计划不执行；
- `cancelled`：用户明确取消。

### 12.4 总任务状态表

本轮仅建立计划，所有任务初始为 `pending`。

| Task | 名称 | 初始状态 | 是否运行训练 | 是否可访问 GenImage unseen | 是否可访问 Defactify |
|---|---|---|---:|---:|---:|
| 01 | 创建独立研究分支与边界快照 | completed | 否 | 否 | 否 |
| 02 | 数据审计与 LOGO 协议设计 | completed | 否 | 否 | 否 |
| 03 | Protocol v3 与 S1 规格冻结 | completed | 否 | 否 | 否 |
| 04 | NPR 特征变换实现与单元测试 | completed | 否 | 否 | 否 |
| 05 | 语义破坏和退化视图管线 | completed | 否 | 否 | 否 |
| 06 | NPR detector 与训练路径 smoke | completed | 仅 tiny smoke | 否 | 否 |
| 07 | OOF 融合、指标和 provenance | completed | 否 | 否 | 否 |
| 08 | Seed-42 三折 LOGO pilot | completed (No-Go) | 是 | 否 | 否 |
| 09 | S1 预注册消融 | skipped_by_gate | 是 | 否 | 否 |
| 10 | 三 seed 确认与模型冻结 | skipped_by_gate | 是 | 否 | 否 |
| 11 | 冻结模型的 GenImage unseen 诊断 | skipped_by_gate | 否 | 是 | 否 |
| 12 | 一次性 Defactify 外部评估 | skipped_by_gate | 否 | 否 | 是，需即时批准 |
| 13 | 统计、结论与研究冻结 | skipped_by_gate | 否 | 否 | 只读既有预测 |

主路线依赖关系：

```text
Task 01
  -> Task 02
    -> Task 03 [Protocol/S1 specification approval gate]
      -> Task 04
        -> Task 05
          -> Task 06
            -> Task 07 [correctness approval gate]
              -> Task 08 [development Go/No-Go]
                -> Task 09
                  -> Task 10 [model-freeze approval gate]
                    -> Task 11 [internal diagnostic stop]
                      -> Task 12 [separate Defactify approval]
                        -> Task 13 [research number/claim freeze]
```

如果 Task 08 为 No-Go：Task 09–13 标记为 `skipped_by_gate`，停止 S1；是否进入
S2 备用链必须由用户另行批准。不能在同一请求中自动切换 S2。

## 13. 主方案 S1 的详细 Task

### Task 01 — 创建独立研究分支与边界快照

**目标**

在不触碰冻结结果的前提下建立后续研究的隔离边界，并记录可恢复的起点。

**前置条件**

- 用户明确批准开始实施，而不仅是批准本计划；
- 用户明确给出或批准研究分支名；
- 当前冻结分支 tracked worktree 无未解释改动。

**允许动作**

- 只读核验冻结 manifest 的路径和 SHA256；
- 从用户指定的冻结提交创建独立分支；
- 新增研究状态/边界文档，例如
  `docs/cross_dataset_plan/RESEARCH_STATUS.md`；
- 记录 start commit、分支、冻结资产清单和禁止覆盖目录。

**禁止动作**

- 不修改模型、数据、训练、评估或配置；
- 不运行训练、inference、GenImage unseen 或 Defactify；
- 不移动、删除或纳入用户已有 untracked files；
- 不修改原 `STATUS.md` 的 Part 11 状态。

**产物**

- 独立研究分支；
- 研究状态文件；
- 冻结边界核验记录。

**验收**

- 原 Protocol v2 freeze manifest 中所有已存在文件 hash 匹配；
- 研究分支起点准确；
- `git diff` 只含获批的研究文档；
- 用户已有 untracked files 列表前后不变。

**停止条件**

完成后停止，等待 Task 02 批准。

### Task 02 — 数据审计与 source-aware LOGO 协议设计

**目标**

在不训练的情况下确认三生成器 LOGO 是否可无泄漏实施，并冻结 split 设计依据。

**前置条件**

- Task 01 completed；
- 仅使用 GenImage train/validation inventory 和 manifest。

**允许动作**

- 统计 generator、label、source、format、尺寸、路径族和样本数；
- 审计 sample ID/path overlap；
- 设计 `{SD1.5, ADM, BigGAN}` 三折 LOGO；
- 对文件 hash 或 perceptual hash 审计写只读脚本/报告，但不得修改原图；
- 设计 real 样本如何按 source/group 进入相同 fold；
- 输出 proposed manifests 到新的 research 路径；此 Task 不把它们用于训练。

**重点检查**

- fake 的 generator holdout 是否同时造成 real source 泄漏；
- 同一真实源图是否被不同生成器目录复制；
- train/validation 是否存在路径级、SHA256 或近重复；
- label 与 JPEG/PNG、尺寸、目录名是否高度相关；
- 每 fold 是否类别平衡且样本量足够。

**建议允许文件**

- 新增 `analysis/audit_cross_dataset_splits.py`；
- 新增 `tests/test_cross_dataset_split_audit.py`；
- 新增 `outputs/research_v3/audits/` 下的轻量 CSV/JSON；
- 新增 `docs/cross_dataset_plan/DATA_AUDIT.md`。

**禁止动作**

- 不读取 Defactify 数据、manifest、prediction 或 subgroup results；
- 不读取 GenImage unseen 图像用于设计；
- 不训练模型；
- 不更改已有 v2 split。

**验收**

- 三折 train/validation sample IDs 完全不重叠；
- group/duplicate audit 通过或明确记录无法解决的边界；
- 每折 generator、source、label、format、size 分布完整；
- split 生成确定性测试通过；
- 所有 proposed manifest 均有 SHA256。

**No-Go**

若 real source 无法可靠分组、存在无法消除的大量近重复，或某 fold 单类/样本过少，
停止并请求用户决定是否更换开发协议；不得自行改为随机 split。

### Task 03 — Protocol v3 与 S1 规格冻结

**目标**

在任何实现或训练前，把 S1 的全部自由度和选择规则写死。

**必须冻结的内容**

- S1 输入：RGB CLIP 分支 + NPR 分支；
- NPR 精确定义、边界 padding 和数值范围；
- NPR backbone：首选 ResNet18 或预注册小 CNN，二选一后不得在 pilot 中切换；
- patch grid、shuffle 概率和是否保留原 patch；
- JPEG/resize/blur 参数与应用概率；
- consistency loss 是否启用、权重；
- optimizer、learning rate、batch、epoch/step budget；
- checkpoint metric；
- OOF fusion 的候选 α 集合与 tie-break；
- seeds `[42,43,44]`；
- LOGO 主/次指标和 Task 08/10 Gate；
- experiment naming、输出目录、cache schema；
- GenImage unseen 与 Defactify 数据角色。

**建议最小冻结规格**

- 首版使用 NPR + patch shuffle；
- fixed α=0.5 是首个融合结果；
- α grid 只能作为预注册对照；
- consistency 可先设为消融而非主版，减少变量；
- 阈值固定 0.5；
- 不新增外部训练数据。

**允许文件**

- 新增 `docs/cross_dataset_plan/PROTOCOL_V3.md`；
- 新增 `configs/research_v3/s1_contract.yaml`；
- 新增机器可读 schema/manifest；
- 新增只验证 config schema 的测试。

**禁止动作**

- 不实现模型；
- 不训练或抽取正式特征；
- 不查看 GenImage unseen/Defactify；
- 不以预计结果为理由留未约束的自动搜索空间。

**验收与 Gate**

- YAML 与文档逐字段一致；
- 没有“根据表现调整”等开放条款；
- 用户明确批准 Protocol v3 和 S1 contract。

批准前状态为 `awaiting_approval`，不得开始 Task 04。

### Task 04 — NPR 特征变换实现与单元测试

**目标**

只实现并验证 NPR 输入表示，不实现完整训练。

**实现范围**

- 新增 `models/npr_detector.py` 中纯函数/模块形式的 NPR transform；
- 支持 `[C,H,W]` 与 `[B,C,H,W]`；
- 明确定义邻域方向、边界裁剪/padding、dtype、device 和输出范围；
- 不在 transform 中读取路径、标签或 generator metadata。

**测试**

- 常量图、水平/垂直梯度、棋盘格的解析期望值；
- CPU/GPU（若可用）一致性容差；
- batch 与逐图结果一致；
- 输入不被原地修改；
- shape、finite、determinism、autograd 行为；
- RGB 转换和 normalize 顺序符合 contract。

**允许文件**

- `models/npr_detector.py`；
- `models/__init__.py`；
- `tests/test_npr_detector.py`；
- 必要的研究文档注释。

**禁止动作**

- 不修改旧 detector 行为；
- 不训练；
- 不使用 GenImage unseen 或 Defactify；
- 不引入未获批依赖。

**验收**

- 新测试和全部旧轻量测试通过；
- tensor contract 文档化；
- 旧 B0/B1/B2/A1/P API 不变。

### Task 05 — 语义破坏与退化视图管线

**目标**

实现可复现、可审计的 patch shuffle 和 paired degradation，但不进行正式训练。

**实现范围**

- patch shuffle 只作用于 NPR 分支；
- RNG 由 global seed、epoch、sample ID 派生；
- validation clean view 不随机；
- paired degraded view 使用 Protocol v3 固定的 JPEG/resize/blur；
- 保证 clean/degraded 标签和 sample ID 一一对应。

**允许文件**

- 新增 `data_pipeline/research_transforms.py`；
- 必要时扩展 `data_pipeline/csv_image_dataset.py`，但保持旧返回 API；
- 新增 `tests/test_research_transforms.py`；
- 新增 tiny fixtures，不能复制真实数据。

**测试**

- 相同 seed/sample/epoch 结果相同；
- 改变 epoch 或允许的 seed 时结果改变；
- patch 数量、排列和边界正确；
- paired view 不换标签、不换 sample ID；
- shuffle 不跨 batch 样本；
- evaluation 路径默认不启用随机 shuffle。

**验收**

- 所有新旧测试通过；
- 没有全局 RNG 隐式漂移；
- transform config 完整进入 provenance。

### Task 06 — NPR detector 与训练路径 smoke

**目标**

组装 NPR expert、loss 和训练入口，只运行足以证明代码可工作的 tiny smoke。

**实现范围**

- 构建 Protocol v3 冻结的轻量 NPR backbone；
- 注册 BCE 主损失；
- 只有 contract 已批准时才加入 consistency loss；
- 加入 class/generator-balanced sampler；
- train/validation 输出完整指标；
- checkpoint/config/log 进入独立 `smoke_..._v3` 目录。

**允许文件**

- `models/npr_detector.py`；
- `training/group_samplers.py`；
- `training/objectives.py`；
- `training/train.py`；
- `training/engine.py`；
- 研究 smoke config；
- 对应 tests。

**允许运行**

- synthetic tensor forward/backward；
- 极小本地样本 smoke，最多一个极短 epoch；
- 全部轻量单元测试。

**禁止运行**

- 不运行 3-fold pilot；
- 不运行完整 GenImage training；
- 不访问 GenImage unseen/Defactify；
- 不根据 smoke 指标改超参，smoke 只判断功能。

**验收**

- forward/backward、save/load、resume、metric 和 sampler tests 通过；
- 无 NaN/Inf；
- smoke 同一 seed 可复现；
- 显存低于 8GB；
- 不覆盖任何 v2 输出。

### Task 07 — OOF 融合、指标与 provenance

**目标**

在 pilot 前完成不泄漏的 out-of-fold score fusion 和研究记录设施。

**实现范围**

- 输入必须是按 sample ID 对齐的 OOF predictions；
- 检查每个 sample 只在其 held-out fold 出现一次；
- 实现 fixed α=0.5；
- 若 contract 允许，按预注册 grid 选择 α；
- tie-break 必须固定；
- 生成 fold mean/worst、两类 recall、error overlap 和互补性指标；
- registry 记录 config/split/checkpoint/prediction hash。

**允许文件**

- `models/late_fusion.py`；
- `evaluation/evaluate_cross_dataset.py`；
- `analysis/cross_dataset_analysis.py`；
- `tests/test_oof_fusion.py`；
- `tests/test_research_registry.py`。

**禁止动作**

- 不使用 in-fold train predictions 选择 α；
- 不读取 GenImage unseen 或 Defactify predictions；
- 不运行正式训练。

**正确性 Gate**

- 人工 toy OOF 数据的 α、metric、tie-break 可解析验证；
- sample ID 错位、重复、缺失时必须 hard fail；
- fusion 结果可从输入 predictions 独立重算；
- provenance hash 完整。

完成后停止并等待用户批准 Task 08。

### Task 08 — Seed-42 三折 LOGO pilot

**目标**

用一次预注册 pilot 判断 S1 是否值得继续。

**运行矩阵**

- B2-v3：3 folds × seed 42；
- NPR-only：3 folds × seed 42；
- CLIP+NPR fixed α=0.5：由 OOF predictions 派生，不做新训练；
- 只有 contract 已批准时才报告 α-grid 对照。

**执行顺序**

1. 运行前复核所有 split/config hash；
2. 先完成 B2-v3 三折；
3. 再完成 NPR-only 三折；
4. 检查每个运行 exit status 和 artifact；
5. 生成 OOF fusion；
6. 计算 clean 与预注册 degradation 指标；
7. 按本文 10.2 自动形成 Go/No-Go 建议。

**禁止动作**

- 不跑 A1/P 额外变体；
- 不改变 epoch、LR、shuffle、backbone 或 α；
- 不访问 GenImage unseen/Defactify；
- 不重跑失败 job 来隐藏失败；只允许记录原因后按修复规则处理。

**产物**

- 6 个训练运行的 registry；
- OOF predictions/hash；
- fold/mean/worst 指标；
- corruption 指标；
- error complementarity 报告；
- `S1_SEED42_DECISION.md`。

**Gate**

- 满足 10.2：状态 `awaiting_approval`，建议 Go；
- 不满足：S1 No-Go，Task 09–13 不执行；
- 正确性问题：Blocked，修复后只重跑受影响范围并保留失败记录。

用户必须明确批准 Go，才能开始 Task 09。

### Task 09 — S1 预注册消融

**目标**

确认收益来自语义去偏/互补信号，而不是偶然容量或增强。

**最小消融**

- S1 主版；
- no patch shuffle；
- 若主版有 consistency：no consistency；
- NPR-only；
- B2-v3；
- fixed α=0.5 与 contract 内 α-grid 的差别。

**预算**

- 仅 seed 42；
- 使用完全相同三折和 optimizer-step budget；
- 不增加新 backbone/head；
- 不新增未预注册增强。

**主要判定**

- 主版必须优于 no-shuffle，或至少显示更好的 worst-fold/跨退化表现；
- fusion 必须优于强单专家；
- 参数量差异和训练时间必须报告；
- 若主版收益无法归因，停止，不进入三 seed。

**产物**

- `S1_ABLATION.csv`；
- `S1_ABLATION_DECISION.md`；
- 完整 registry 与 hash。

**停止条件**

完成后停止。只有消融支持主假设且用户批准，才执行 Task 10。

### Task 10 — 三 seed 确认与模型冻结

**目标**

在不访问内部/外部测试的情况下确认稳定性并冻结最终模型。

**运行矩阵**

- B2-v3：seeds 42/43/44 × 3 LOGO folds；
- 最终 S1：seeds 42/43/44 × 3 LOGO folds；
- seed 42 已有且 contract/hash 完全一致时可复用，不得因指标好坏决定是否复用。

**分析**

- 每 seed、每 fold；
- mean ± sample SD；
- paired CI；
- worst-fold；
- α 稳定性；
- clean/corruption；
- 推理时间、峰值显存、参数量。

**Model Freeze Gate**

严格使用本文 10.3。通过后冻结：

- architecture、transform、loss、optimizer、epoch；
- α、threshold、calibration；
- checkpoint selection；
- seeds；
- 全量 GenImage train/validation 的最终训练方式；
- external evaluation commands。

**产物**

- `S1_THREE_SEED_RESULTS.csv`；
- `MODEL_FREEZE_V3.md`；
- `model_freeze_v3_manifest.json`；
- 最终训练 config snapshots 和 hashes。

**停止条件**

记录 Go/No-Go 后停止，等待用户批准模型冻结。未批准不得运行 Task 11。

### Task 11 — 冻结模型的 GenImage unseen 内部诊断

**目标**

对已经冻结的 B2-v3 与 S1 做一次内部 held-out-generator 诊断。

**前提**

- Task 10 的模型、checkpoint、threshold、α 和 commands 已冻结；
- 用户明确批准 Task 11；
- 任何结果都不能触发模型调整。

**运行**

- 每模型每 seed 对 GenImage unseen 推理一次；
- 同一模型的所有派生指标来自同一 prediction；
- 运行 clean 及预注册的 JPEG/resize/blur；
- 输出 per-generator、worst-generator 和 paired metrics。

**禁止动作**

- 不重选 checkpoint、threshold 或 α；
- 不根据结果决定换增强/block/backbone；
- 不访问 Defactify；
- 不把 GenImage unseen 重新称为盲测。

**产物**

- prediction hashes；
- `GENIMAGE_UNSEEN_DIAGNOSTIC.csv`；
- paired CI 和 generator/corruption summaries；
- 是否满足内部目标的只读结论。

**停止条件**

无论结果如何都停止。Task 12 需要新的、单独的用户批准。

### Task 12 — 一次性 Defactify 外部评估

**目标**

用冻结模型一次性检验跨数据集假设，不做模型选择。

**即时批准要求**

Task 12 开始前必须再次向用户展示：

- frozen model/config/checkpoint hashes；
- 要执行的精确命令；
- seeds 与运行数；
- full→balanced 派生规则；
- 失败重试策略；
- 明确声明结果不会触发调参。

只有用户在该时点明确批准后才可执行。

**运行矩阵**

- B2-v3 seeds 42/43/44：Defactify full；
- S1 seeds 42/43/44：Defactify full；
- balanced 只从相应 full prediction 按冻结 sample IDs 派生；
- 不为 balanced 做第二次 inference。

**禁止动作**

- 不读取 Defactify train/validation；
- 不选 checkpoint/threshold/α；
- 不排除 seed；
- 不因首个 seed 结果不好而停止或改命令；
- 不删除/覆盖失败日志；
- 不追加新模型。

**产物**

- 6 个 full prediction hashes；
- 6 个 full 和 6 个 balanced metric rows；
- per-generator；
- 完整 registry；
- failure/retry audit。

**停止条件**

矩阵完整或出现不可修复的正确性问题后停止。不得自动开始统计解释以外的新实验。

### Task 13 — 统计、结论与研究冻结

**目标**

从 Task 10–12 的冻结 predictions 生成可复核结论，不再运行模型。

**分析范围**

- seeds mean ± sample SD；
- S1−B2 paired deltas；
- matched sample IDs 的 hierarchical bootstrap；
- full/balanced/per-generator/worst-generator；
- clean/corruption；
- error overlap；
- efficiency；
- 本文 10.4 External Scientific Gate。

**结论枚举**

- `SUPPORTED_CROSS_DATASET_GAIN`；
- `PARTIAL_TRADE_OFF`；
- `CONTROLLED_NEGATIVE_RESULT`；
- `INVALID_DUE_TO_CORRECTNESS`。

**允许文件**

- 轻量汇总 CSV/JSON；
- 分析脚本和测试；
- `docs/cross_dataset_plan/FINAL_RESEARCH_RESULT.md`；
- `artifacts/research_v3_number_freeze_manifest.json`；
- 不超过预注册数量的主图表。

**禁止动作**

- 不重新 inference；
- 不修改任何 prediction；
- 不调整 threshold/α；
- 不选择性排除 seed/generator；
- 不把 partial/negative result 写成全面提升。

**验收**

- 所有指标能从 prediction 独立重算；
- 所有 manifest hash 匹配；
- 结论严格匹配 10.4；
- 限制包括数据规模、已暴露测试集、near-duplicate audit 边界和环境成本；
- 用户批准后才标记研究数字/结论冻结。

完成后停止，不自动实现 S2 或 S3。

## 14. S1 No-Go 后的备用任务链

备用链只有在 S1 Task 08 或 Task 09 No-Go、且用户明确批准后才启用。

### Task F01 — S1 失败审计与 S2 预注册

- 状态：`awaiting_approval`；合同见
  `configs/research_v3/S2_CONTRACT.yaml`，说明见
  `docs/cross_dataset_plan/S2_PREREGISTRATION.md`；
- 只读分析 S1 LOGO 证据；
- 判断失败属于局部信号弱、退化敏感、融合不互补或实现/数据问题；
- 禁止读取 Defactify；
- 冻结 S2 block IDs `{3,6,9,12}`、projection、TIE、SupCon 权重和预算；
- 输出 `S2_CONTRACT.yaml` 并等待批准。

### Task F02 — 多 block CLIP feature contract

- 状态：`awaiting_approval`；实现与验证完成，停在 F03 审批 Gate；
- 实现一次 forward 抽取多 block CLS；
- cache signature 包含 block IDs、open_clip 版本、preprocess、split hash；
- 测试 block 顺序、shape、determinism、single-forward 等价性；
- 不训练。

### Task F03 — RINE-lite head 与 SupCon smoke

- 实现共享投影、softmax layer importance、128-D 聚合和分类头；
- 实现 SupCon；
- 只做 tiny smoke；
- 不访问 GenImage unseen/Defactify。

### Task F04 — S2 Seed-42 LOGO Gate

- B2-v3 与 S2 三折 seed 42；
- 使用与 S1 相同开发 Gate；
- No-Go 则停止；Go 需用户批准。

### Task F05 — S2 三 seed、冻结与评估

- 复用 Task 10–13 的纪律；
- 每个阶段仍需单独请求；
- 不因为 S1 的外部结果选择 S2。

S3 频谱路线不在自动备用链内。若 S1/S2 均 No-Go，必须新开一轮研究计划批准，
再决定是否承担 one-class spectral reconstruction 的更高成本。

## 15. 每个 Task 的标准交接格式

未来每个 Task 完成时使用以下格式，并确保最终答复自包含：

```text
Cross-dataset Task XX completion summary

Task:
Status:
Branch:
Start commit:
Current commit:
Protocol/contract hash:
Inputs read:
Commands executed:
Training/inference executed:
GenImage unseen accessed:
Defactify accessed:
Files changed:
Files added:
Artifacts generated:
Tests:
Key evidence:
Gate decision:
Failures/retries:
Known issues:
Frozen v2 assets unchanged:
User untracked files preserved:
Ready for next Task:
Required user approval:
```

如果 Task 位于 Gate，`Ready for next Task` 必须为 `no`，直到用户明确批准。

## 16. 本轮计划写入的停止点

本轮仅完成详细任务拆分：

- 未创建研究分支；
- 未创建 Protocol v3；
- 未修改代码或配置；
- 未运行训练、inference、GenImage unseen 或 Defactify；
- 未修改任何 Protocol v2 冻结资产；
- 未改变 Part 11–12 状态；
- 未 commit、push 或 merge。

下一步只能在用户明确批准后从 **Task 01** 开始，一次执行一个 Task。
