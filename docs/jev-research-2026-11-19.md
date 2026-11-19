# Jev 调研报告：System One 决策模型品类分析与 X Lead Miner 对照

> 调研日期：2026-11-19
> 调研人：x-lead-miner 项目组
> 调研目的：明确 Jev 的产品与技术形态，评估其与本项目（X 购买意图
> 打分模型）的关联与差异，为项目定位、简历表述与后续路线提供依据。
>
> **证据分级说明**：本报告所有事实性陈述均标注来源与访问日期。
> 一手来源（官方博客、官方仓库 README 原文）> 二手来源（第三方
> 仓库描述）> 推断（标注"推断"）。凡属推断，均给出推理依据。

---

## 1. 摘要

Jev 是 TypeSafe AI（创始人 Diogo Almeida，前 OpenAI ChatGPT 研究核心
成员）于 2026-09-15 发布的**"System One 模型"**品类首个商业产品：
接收非结构化状态 + 类型化问题，返回带置信度的类型化决策（选择/评分/
布尔），专为软件直接调用设计，比 LLM 快约两个数量级。其官方训练方法
为 RLCD（Reinforcement Learning for Calibrated Decisions），架构自研
未公开。

本项目的 X 购买意图打分模型与 Jev 属于**同一范式**（System One 式
类型化决策：小模型 + 快推理 + CPU/边缘部署 + 替代 LLM 调用），但处于
不同的市场位置：Jev 是通用决策层商业 API；本项目是单一领域（X 购买
意图）的自训练、自托管、全开源实现。

关键结论：
1. 范式一致性为本项目提供了**品类级验证**（前 OpenAI 研究者创立的
   公司押注同一方向）。
2. 技术实现路径不同（本项目：冻结嵌入 + 49k 参数 MLP + teacher 蒸馏；
   Jev：自研架构 + RLCD），不可直接对标性能数字。
3. 生态接入机会存在：社区已出现 Jev API 兼容格式（/v1/systemone），
   本项目打分器可包装为兼容端点接入 awesome-jev 生态。
4. 风险：TypeSafe AI 若推出领域专用模型或开放通用模型的微调，
   本项目的"自建打分器"价值将被压缩；数据集与领域标注是护城河。

---

## 2. 调研方法

| 维度 | 说明 |
| --- | --- |
| 检索渠道 | GitHub 仓库检索（`gh search repos jev`）、GitHub API（仓库 README 原文抓取）、TypeSafe AI 官方博客（直接抓取）、Hugging Face 生态 |
| 关键词 | `jev`、`system one model`、`typed decisions` |
| 时间窗 | Jev 发布（2026-09-15）至今（2026-11-19），约 2 个月 |
| 证据分级 | S1 = 官方一手（TypeSafe AI 博客/文档）；S2 = 高星公开仓库 README 原文；S3 = CI 运行日志实测；推断 = 明确标注 |
| 访问日期 | 全部来源访问于 2026-11-19 |

---

## 3. Jev 是什么（S1：官方一手来源）

### 3.1 产品定义

来源：TypeSafe AI 官方博客《Introducing System One Models & Jev》
（2026-09-15，作者 Diogo Almeida），
https://typesafe.ai/blog/introducing-system-one-models-and-jev
（访问于 2026-11-19）

- **定位**："System One 模型"——软件可直接使用的快速结构化决策层
- **输入/输出**：非结构化状态 + 类型化问题 → 带置信度的类型化决策
  （选择 / 评分 / 布尔）
- **明确边界**：不是聊天模型（官方原文"gives up string generation"，
  放弃了字符串生成能力）
- **性能宣称**：System One 任务上与现有 LLM 智能水平相当，速度快
  约两个数量级
- **性能基线**（同博客）：字符串生成任务延迟 129 / 311 / 3,060ms 的
  对比对象，Jev 决策延迟 28ms（S2：jevos README 引用同组数字）

### 3.2 公司与技术方法

- 创始人 Diogo Almeida，自述"在 OpenAI 参与了让语言模型遵循指令
  与对话的研究，该工作成为 ChatGPT 背后的研究"（S1，官方博客）
- 训练方法官方命名：**RLCD（Reinforcement Learning for Calibrated
  Decisions，校准决策强化学习）**（S1，官方博客）
- 技术栈三要素（S1，官方博客）：新模型架构（未公开细节）+
  并行采样器（效率最大化）+ RLCD 训练法
- 发布形态：早期访问（early access），API 托管（S2：生态仓库
  README 均以"调用 Jev API"为接入方式）

### 3.3 品类框架：System 1 / System 2

- System 2：大语言模型——慢、贵、擅长推理与生成
- System 1：小模型——快、便宜、擅长类型化决策
- 分工逻辑：System 2 的判断力通过训练/蒸馏"下沉"到 System 1，
  软件的常规决策由 System 1 承接，仅复杂案例升级回 System 2
- **判定**：该框架与本项目"teacher 蒸馏 + 小模型打分"的架构完全同构
  （推断，依据：本项目训练方法为 Gemini teacher 标签蒸馏进 MLP 头，
  与"System 2 判断力下沉至 System 1"定义一致）

---

## 4. Jev 生态地图（S2：公开仓库实证）

数据来源：GitHub 检索 `gh search repos jev`（按星数排序，访问于
2026-11-19）及仓库 README 原文。

| 仓库 | 星数 | 定位 | 与本项目的关系 |
| --- | --- | --- | --- |
| browser-use/jev-ultrafast | 24,658 | 最快最便宜的 web agent | 同方向不同场景 |
| yibie/awesome-jev | 2,612 | 官方生态精选清单（品类地图） | 生态参照系 |
| dzhng/jevgrep | 3,247 | 代码检索 CLI（Jev 驱动） | 决策层用法范例 |
| jarrodwatts/jev-trader | 3,144 | 交易决策（每 Monad 区块一次） | 高频决策范例 |
| jev-chat/jev-chat-jarvis | 8,462 | 聊天辅助决策（中文市场） | 最接近本项目的用户形态 |
| TianyuCodings/NanoJev | 2,612 | 0.6B 开源复刻 + 端到端训练管线 | **架构最近似的开源对照** |
| feder-cr/jev (jevos) | 970 | Jev API 兼容的开源实现（OpenVINO int8，笔记本 CPU 28-130ms） | **部署形态最接近的对照** |
| TheoLeeCJ/SemIf-OpenJev | 3,247 | 开源模型语义 if | 独立实现 |
| tamaratran/fast-jev-compaction | 7,998 | Claude Code 插件（Jev 决策替代压缩摘要） | 决策层用法范例 |

### 4.1 生态成熟度判定（S2 + 推断）

- 发布 2 个月内出现：官方生态精选清单、独立基准（System1-Arena，
  近 10 万题）、至少 3 个独立复刻/兼容实现、24.6k★ 的头部应用
- **判定**：品类处于"快速成长早期"，需求被验证，但尚无垄断者
  （推断，依据：复刻品与兼容实现的存在说明官方产品存在未被覆盖
  的需求，如本地部署与成本）

---

## 5. 技术形态对照（本项目 vs Jev 系）

### 5.1 决策模式对照

| 维度 | TypeSafe Jev | 本项目（X Lead Miner intent head） |
| --- | --- | --- |
| 任务范围 | 通用：任意 typed question（分类/路由/评分/验证/护栏） | 专用：X 帖子购买意图 0-1 评分 |
| 输入 | 非结构化状态 + 类型化问题 | 帖子文本（固定单一问题） |
| 输出 | 带置信度的类型化决策 | 0-1 连续分 + 三档分层 |
| 可训练参数 | 未公开（自研架构，RLCD 训练） | 49,281（0.2MB fp32 产物） |
| 推理硬件 | 未公开（宣称比 LLM 快两个数量级） | CPU（M1 单核实测 0.07-0.25s/条，含嵌入） |
| 部署形态 | TypeSafe 云 API 托管 | 自托管双形态：① CF Worker 原生（Workers AI 嵌入 + 内置头权重）② 本地 FastAPI sidecar |
| 开源程度 | 闭源商业（early access） | 全开源（代码 + 全量匿名化数据集 7,866 条） |
| 训练数据 | 未公开 | 7,866 条真实 X 帖子，27 查询族，teacher 蒸馏 |

### 5.2 共同的架构范式（本项目的品类归属依据）

两者同构于同一决策模式：

```
非结构化输入（文本/状态）
  → 嵌入/编码（固定表征层）
  → 轻量决策头（唯一可训练部分）
  → 带置信度的类型化输出
```

- Jev：自研编码 + 自研决策头，RLCD 训练
- 本项目：公开嵌入（embeddinggemma-2 / bge-base）+ 49k MLP 头，
  teacher 蒸馏（Gemini 标签 BCE 训练）
- **判定**：范式同源（推断，依据：两者均满足"小模型 + 快推理 +
  类型化置信输出 + 替代 LLM 调用"的 System One 定义）

### 5.3 指标对照的诚实边界

本项目 test 指标（test precision@5 = 1.00, accuracy = 0.877,
n_test = 1,179）**不可与 Jev 宣称性能直接对比**，原因：

1. 本项目指标度量的是"与 teacher（Gemini 2.5 Flash）的一致率"，
   teacher 对人类判断的一致率约 90%（50 样本人工审计，见
   `data/spot_check_50.md`）
2. Jev 的性能宣称基于其自有基准（System One 任务定义由 TypeSafe
   自定），口径不可比
3. 数据分布为查询条件决定（intent 查询族的帖子含显性标记词），
   precision@k = 1.00 部分反映数据可分性而非真实判别力

---

## 6. 已发生的错误与修正记录（本项目对齐 Jev 品类实践的过程）

对照 Jev 品类的公开实践（RLCD 的"校准"目标），本项目在教师蒸馏
过程中发现并修正了三类系统性问题，均可视为"校准缺失"的具体案例：

| 问题 | 现象 | 根因 | 修正 |
| --- | --- | --- | --- |
| teacher 标签倒挂 | 41 条"意向"混入娱乐消费帖 | teacher 提示词按表面句式判断 | 正/反锚定示例写入提示词；50 条人工审计制度化 |
| 训练不足 | 全部分数挤在 0.25-0.40，判别力为零 | 全批量训练仅 30 步梯度更新 | mini-batch + val 早停 + 独立 test 集 |
| 验证口径错误 | precision@5 = 1.00 被误读为"完美模型" | 度量对象是 teacher 一致率而非 ground truth | 指标重定义为"teacher agreement"，并制度化人工抽查 |

---

## 7. 结论

### 7.1 品类归属

本项目属于 **System One 决策模型品类的领域专用、自托管、全开源
实现**——与 TypeSafe AI 的 Jev 同范式、不同市场位置。

### 7.2 差异化定位（对外表述建议）

- **推荐口径**："System-One-style decision model：teacher 蒸馏、
  49K 可训练参数、CPU 推理、自托管、全开源"
- **不推荐口径**："对标 Jev"（技术层级不同，RLCD vs 标准蒸馏）；
  "训练了 0.7B 模型"（0.7B 是冻结底座，未训练）

### 7.3 机会

1. **数据集稀缺性**：System1-Arena 刚建立"哪些决策可交给 System 1"
   的基准方向，**领域专用（X 购买意图）标注数据集当前公开缺位**——
   本项目已开源的 7,866 条教师标注数据集可直接补位（论文路线 A）
2. **生态接入**：jevos 已实现 Jev API 兼容（/v1/systemone），
   本项目打分器可按同一合约暴露端点，接入 awesome-jev 生态获得曝光
3. **校准研究方向**：本项目实测的"teacher 偏保守（complaint 漏检）"
   是 RLCD"校准决策"的直接应用场景——校准质量可作为迭代目标

### 7.4 风险

1. TypeSafe AI 推出领域专用产品或开放微调 → 自建打分器价值压缩
2. 教师模型迭代速度快（Flash → 更新模型），标签需定期重标
3. 本项目 precision@k = 1.00 部分来自查询条件与数据可分性，
   对外表述须保留 teacher-agreement 口径

---

## 8. 附录：来源清单

| # | 来源 | 类型 | 链接 | 访问日期 |
| --- | --- | --- | --- | --- |
| 1 | TypeSafe AI 官方博客：Introducing System One Models & Jev（Diogo Almeida） | S1 官方一手 | typesafe.ai/blog/introducing-system-one-models-and-jev | 2026-11-19 |
| 2 | yibie/awesome-jev（官方生态精选清单 README 原文） | S2 公开仓库 | github.com/yibie/awesome-jev | 2026-11-19 |
| 3 | TianyuCodings/NanoJev（0.6B 开源复刻 README 原文） | S2 公开仓库 | github.com/TianyuCodings/NanoJev | 2026-11-19 |
| 4 | feder-cr/jev（jevos，Jev API 兼容开源实现 README 原文） | S2 公开仓库 | github.com/feder-cr/jev | 2026-11-19 |
| 5 | jarrodwatts/jev-trader、dzhng/jevgrep、jev-chat/jev-chat-jarvis、browser-use/jev-ultrafast、tamaratran/fast-jev-compaction、TheoLeeCJ/SemIf-OpenJev | S2 生态样本 | github.com（各自仓库） | 2026-11-19 |
| 6 | 本项目实测数据（test 指标、线上打分日志、50 条人工审计） | S3 项目实测 | apps/x-lead-miner/intent（monoboard） | 2026-11-17 ~ 11-19 |

> 注：调研期间 GitHub Actions 的部分运行日志（BlobNotFound）与
> Hugging Face Space 免费档限制（402）为平台侧临时状态，不影响
> 本报告结论。
