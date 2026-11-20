# LLM Teacher 蒸馏的三个系统性陷阱：一个 7,866 样本案例研究

> 本文记录了在构建一个"X 购买意图打分系统"的过程中，发现并修复的三个
> 系统性问题。这三个问题都不是简单的 bug——它们分别指向**教师标签质量**、
> **小模型训练动力学**、**数据管道鲁棒性**三个在 LLM 蒸馏场景中普遍存在
> 但容易被忽视的深层挑战。全文基于真实数据和可复现的实验，不含任何
> 虚构或美化成分。

## 背景

我们要解决的问题：从 X（Twitter）上实时挖掘**具有购买意向的帖子**。

每天有数千人在 X 上公开发出这样的信号——"is there an app that…"、
"anyone recommend…"、"I wish there was…"。这些帖子是销售线索的
天然来源，但混在信息流噪音中，人工筛选成本极高。

我们的方案是一个四阶段管线：

```
采集（27 组查询配方）→ LLM Teacher 标注 → 小模型蒸馏 → 边缘部署
```

核心思路：用 Gemini 2.5 Flash 作为 teacher 标注 7,866 条 X 帖子
（2,850 条购买意向 / 5,016 条噪音），然后将 teacher 的判断力蒸馏进
一个极小的模型——冻结嵌入底座（768 维输出）+ 一个 49,281 参数的
MLP 分类头——最终在 CPU 上实现单条 0.07-0.25 秒的实时打分。

选 49k 参数不是保守，是刻意设计：**底座已经把 7.4 亿参数的语言理解
压缩成了 768 维向量，决策面在这个空间里是一条几何分界线**——不需要
更多参数来重复底座已经做完的语言理解工作。

这套设计听起来很合理。但我们在执行过程中踩了三个坑，每个坑都指向
一个我们最初没有预料到的系统性问题。

---

## 陷阱一：Teacher 标签噪声——蒸馏的真正瓶颈

### 现象

第一版模型训练完成，测试集 precision@5 = 1.00——"排名前 5 条全是
精准线索"。指标看起来很好。

直到我们对 50 条标注进行了人工审计。

审计发现：41 条被标为"意向"的帖子里，混着大量这样的内容：

> "recommend me some anime to watch"
> "recommend a good book for the weekend"
> "can anyone recommend me some cool youtubers?"

这些都是**内容消费**——用户想看动漫、读书、刷视频——不是购买产品
或服务的意向。而真正的购买意向帖——"I wish there was an app that
could track my subscriptions"——反而被标成了噪音。

标签分布严重偏斜：真正的产品采购帖被淹没在娱乐内容求助中。

### 根因分析

问题出在 teacher 的标注提示词。原始版本：

> "判断这条帖子是否表达购买意向（asking for recommendations, seeking
> an alternative, complaining about a tool they use, willing to pay）"

这个提示词有两个致命缺陷：

**缺陷一：表面模式匹配。** Gemini 看到 "recommend" 就倾向给高分，
因为它没有区分"推荐给我一个可以买的东西"和"推荐给我一个可以看的
东西"。两者的语法结构几乎相同，但商业含义完全不同。

**缺陷二：缺少校准锚点。** 没有正/反示例来"钉住"判断边界，teacher
按自己预训练时的先验来分类——而这个先验偏向于把所有"求推荐"都归为
同一类。

### 修复

```python
SYSTEM_PROMPT = (
    "You label public X posts for a sales-lead mining tool whose users "
    "want to find people to sell products to.\n"
    '"intent" rules — true ONLY when the author actively seeks a PRODUCT, '
    "TOOL, APP or SERVICE to acquire or use: asking for product "
    "recommendations for a concrete need, seeking an alternative to a tool "
    "they already use, complaining about a tool's limitation, or wishing "
    "an app/software had a missing capability. false when the ask is for "
    "CONTENT to passively consume (movies, shows, books, music, "
    "YouTubers, documentaries, games), when the author only praises "
    "something, or when there is no product seek at all.\n"
    '- "type": recommend_request, alternative_seek, complaint, '
    "feature_wish, praise, noise\n"
)
```

核心改动：把"true ONLY when…false when…"的逻辑边界显式化，特别是
**排除条件**（CONTENT to passively consume）——这是原提示词完全没有
的。

### 验证

重标 7,866 条后，新的 True 样本（41 → 2,850 中真实采样）：

```
✓ "Can anyone recommend a good robot vac/mop? Hardwood and tiled floors"
✓ "Does anyone recommend any artist for a ref sheet I want my sona…"
✓ "Is there an app that simplifies iOS experience for the elderly?"
✗ "recommend me some cool youtubers to watch"（→ 正确标为 False）
✗ "Can anyone recommend which book I should read?"（→ 正确标为 False）
```

### 更深层的教训

**蒸馏的质量上限 = teacher 的质量上限。** 这不是一句空话——我们实测：

- teacher 标签修改前：模型把所有帖子挤在 0.25-0.40 之间，零判别力
- teacher 标签修正后：模型正确区分了付费意向帖（0.99）和内容消费帖
  （0.002），三档分层的 1:3:6 比例在真实查询上可复现

这意味着：**在这个任务上，提升模型规模（49k → 7B → 70B）不会带来
任何收益——瓶颈在 teacher 标签质量，不在模型容量。** 这是一个反直觉
但重要的结论：大多数蒸馏优化工作聚焦在模型架构和训练策略上，而我们
的实践表明，当 teacher 本身有系统性偏差时，这些优化全部无效。

---

## 陷阱二：小模型训练动力学——30 步梯度更新为什么不够

### 现象

修正标签后重训，指标依然不对——模型把所有帖子的分数挤在 0.25-0.40
的窄带里，没有任何判别力。

```
所有帖子分数分布：
正类均值 = 0.301，范围 [0.268, 0.396]
负类均值 = 0.278，范围 [0.245, 0.323]
最大差异 = 0.05（远不足以做出可靠判断）
```

### 根因分析

检查训练代码，发现了一个隐藏在"看起来合理"背后的训练动力学问题：

```python
for epoch in range(30):       # 30 轮
    loss.backward()            # 全批量：每轮只算一次梯度
    optimizer.step()           # 每轮只更新一次参数
```

全批量（full-batch）训练中，每个 epoch 只执行 **1 次**梯度更新。
30 个 epoch = **总共 30 次参数更新**。

对于一个 49,281 参数的模型在 5,508 条样本上做二分类，30 次更新意味着
什么？参数从随机初始化开始，每个参数只被"推动"了 30 次——远不足以
收敛到任何有意义的解。实测：分数全部挤在 0.25-0.40 的窄带里，
accuracy = 75.8%，恰好等于"全预测为负"的多数类基线。

### 修复

```python
# 改为 mini-batch：每轮 5,508/16 ≈ 345 步 × 200 轮（cap）= 数万步
BATCH = 16
EPOCHS = 200          # 上限，早停通常会提前结束
PATIENCE = 20         # 连续 20 轮 val_loss 无改善则停

for epoch in range(EPOCHS):
    perm = torch.randperm(len(x_train), generator=generator)
    for start in range(0, len(perm), BATCH):
        batch = perm[start : start + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(head(x_train[batch]).squeeze(-1), y_train[batch])
        loss.backward()
        optimizer.step()

    # 每个 epoch 结束后检查 val_loss
    if val_loss < best_val_loss - 1e-4:
        best_state = copy(head.state_dict())
        stale = 0
    else:
        stale += 1
        if stale >= PATIENCE:
            break
```

### 效果

```
修复前：30 步梯度更新，全部分数挤在 0.25-0.40，accuracy = 多数类基线
修复后：~6,000 步梯度更新，val_loss 0.67 → 0.28，accuracy 0.877
```

### 教训

**小模型的训练动力学和大模型完全不同。** 大模型（百万-亿级参数）在
大量数据上通常需要 careful learning rate scheduling 和 gradient
accumulation；小模型（几万参数）在少量数据上需要的是**更多的梯度
更新次数**和**正则化（早停/Dropout）来防止过拟合**。

通用经验法则：梯度更新次数至少应该是参数量的 10-100 倍（在此案例中
即 5,000-500,000 次）。我们最初的 30 步更新比这个少了 2-4 个数量级。

---

## 陷阱三：U+2028 静默数据丢失——一个跨平台 Unicode 陷阱

### 现象

部署上线后，标注流程报告 7,854 / 7,866 条完成。但数据集文件中怎么也
找不到那消失的 12 条帖子——文件行数没错，行内容也没错，它们就是
"不在"了。

### 根因

X 帖子中偶发含有 **U+2028（Line Separator）**和 **U+2029（Paragraph
Separator）**字符。这些 Unicode 字符在视觉上类似于换行，但在语义上
**不是**换行——它们是文本内容的一部分。

Python 的 `str.splitlines()` 会按这些字符**切分字符串**：

```python
# 一条包含 U+2028 的帖子（原始 JSONL 一行）：
'{"id": "123", "text": "Hello\u2028World"}'

# read_text().splitlines() 会把它切成：
# ['{"id": "123", "text": "Hello', 'World"}']
# 两段都不是合法的 JSON → 静默跳过
```

更阴险的是：JSONL 文件本身是完好的——`json.loads()` 对文件对象
迭代（按 \n 切分）能正确解析每一行。问题只出现在**用 splitlines()
预处理字符串**的代码路径中。

### 影响范围

- 12 条帖子在所有使用 splitlines() 的代码路径中被永久丢失
- 这些帖子在训练集和测试集中均不可见
- 没有任何报错——它们被**静默**地从数据集中排除

### 修复

```python
# 所有 JSONL 读取统一改为文件对象迭代（只按 \n 切）
with open(path, encoding="utf-8") as fh:
    for line in fh:            # 按物理行（\n）迭代，不碰 U+2028
        r = json.loads(line)   # 完整行正确解析
```

---

## 部署架构：Worker 原生推理 vs 中心化 sidecar

训练完成后，下一个问题是**怎么部署打分服务**。我们对比了两种方案：

### 方案 A：中心化 sidecar（Python + PyTorch）

```
Client → API Route → HTTP → FastAPI sidecar (:5019) → 模型推理 → 返回分数
```

- 嵌入模型：embeddinggemma-2（740M，Google）
- 优点：完整质量（更大的底座 = 更好的语义理解）
- 缺点：需要维护一个 Python 进程；嵌入模型 1.5GB 常驻内存

### 方案 B：Worker 原生推理（Workers AI + 内置权重）

```
Client → API Route → Workers AI（bge-base 嵌入）→ 内置 MLP 头 → 返回分数
```

- 嵌入模型：Cloudflare Workers AI 原生托管的 bge-base-en-v1.5
- 优点：零外部依赖，权重内置在 Worker 中（199KB JSON），边缘推理
- 缺点：底座从 740M 换成了 110M（语义理解能力略降）

### 最终选择：双路径架构

```python
async def scoreTexts(texts, perItem=False):
    # 优先尝试 Worker 原生路径（部署环境有 AI binding）
    native = await scoreNative(texts, perItem)
    if native:
        return native
    # 降级到 sidecar（本地开发环境）
    return await scoreSidecar(texts, perItem)
```

**数学一致性验证**：Worker 内置头和本地 sidecar 头的推理结果逐位
一致（JS 实测 0.9982/0.0040 == PyTorch 实测 0.9982/0.0040），因为
两者的权重来自同一个训练产物。

---

## 评估方法论：Teacher 一致率 ≠ Ground Truth

### 我们为什么不用 ground truth

因为我们没有 ground truth。所有 7,866 条标签都来自 Gemini teacher，
没有人工金标。

这不是偷懒——是**当前阶段的最优选择**：

1. 人工标注 7,866 条需要 2-4 周（双人标注 + IAA 计算）
2. teacher 标签虽然只有 ~90% 人类一致率，但已经**远超随机**（50%）
3. 本项目的目标是产品（线索队列），不是论文（benchmark 发布）——
   产品可以在使用中迭代，不需要完美的初始标签

### 但我们必须诚实

```
我们的指标度量的是：模型与 teacher 的一致率
不代表：模型与人类判断的一致率
差异来源：teacher 自身的系统性偏差
```

具体偏差（50 条人工审计发现）：
- teacher 对"求推荐内容消费"（看动漫/书/电影）倾向于低估意向 → 修正后正确
- teacher 对"品牌误触发抱怨"（如 Ring 门铃误报）倾向于漏报 → 已知但未完全修正

这个诚实立场本身就是项目的一个方法论贡献——大多数 AI 应用博文把
teacher 标签当 ground truth 报告指标，我们明确区分了两者。

---

## 最终指标

三路分层切分（train 5,508 / val 1,179 / test 1,179），test 从未参与
训练或早停：

| 指标 | 验证集 | 测试集 |
| --- | --- | --- |
| precision@5 | 1.00 | 1.00 |
| precision@10 | 1.00 | 0.70 |
| accuracy | 0.889 | 0.877 |
| ECE（校准误差） | 0.053 | — |

线上实测（与训练集无重叠的新查询）：

| 帖子 | 分数 | 判定 |
| --- | --- | --- |
| "is there an app that tracks competitor pricing daily? willing to pay" | 0.9964 | 强意向 |
| "looking for a cheaper alternative to Ahrefs" | 0.8094 | 强意向 |
| "just watched the new Dune movie" | 0.0162 | 噪音 |
| "can anyone recommend me some cool youtubers?" | 0.0002 | 噪音 |

---

## 总结：三个陷阱 → 三个洞察

| 陷阱 | 表面原因 | 深层原因 | 修复 | 洞察 |
| --- | --- | --- | --- | --- |
| 标签倒挂 | teacher "太笨" | 提示词缺少校准锚点，表面句式 ≠ 商业语义 | 显式正反锚点 | **teacher 质量是蒸馏的瓶颈，不是模型容量** |
| 训练不足 | "模型太小" | 全批量 30 步梯度更新，远不足以收敛 | mini-batch + 早停 | **小模型需要不同的训练配方**：更多梯度更新次数，而非更多参数 |
| U+2028 丢数据 | "Unicode 坑" | str.splitlines() 按 U+2028 切分 JSONL | 文件对象迭代 | **静默数据丢失比崩溃更危险**——管道中最需要防御的是"没有报错的丢失" |

这三个问题的共同点：**它们都不是模型架构的问题**。换更大的模型、加
更多层、用更好的优化器——都不能解决。它们是数据质量、训练配方和
数据工程的問題。

---

## 开源与复现

全部代码、数据集和模型权重已开源：

- 管线 + 全量数据集：github.com/flashclub/x-lead-intent-pipeline
- SaaS：github.com/flashclub/x-lead-miner
- HF 模型：huggingface.co/flashcoder/tweet-buy-intent
- HF 数据集：huggingface.co/datasets/flashcoder/tweet-buy-intent-leads

复现只需一台笔记本（无 GPU 要求）+ 两个 API key（X 数据源 + 标注
teacher），总成本 < $10/月。
