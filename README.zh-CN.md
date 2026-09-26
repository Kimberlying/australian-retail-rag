# Australian Retail RAG

[![English](https://img.shields.io/badge/lang-English-lightgrey.svg)](README.md)
[![简体中文](https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue.svg)](README.zh-CN.md)
[![CI](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml)

一个面向澳大利亚零售行业、以评估为核心构建的检索增强生成（RAG）系统。数据来源是澳大利亚上市公司的公开资料，以及明确标注为合成数据的零售文档。

这是一个独立的作品集项目，**不是 Coles 的系统、客户项目，也不代表 Coles 的背书**。合成文档中的 "Harbourline Retail" 是一家虚构公司。

## 亮点

- **评估框架：** 55 道题的黄金测试集（golden set），覆盖检索指标（Hit@k、Recall@k、MRR、nDCG、Precision@k）和生成指标（答案准确率、拒答准确率、引用有效性、LLM-as-judge 忠实度）。检索或拒答质量一旦下降，CI 就会失败。
- **混合检索：** BM25 与向量检索（`bge-small-en-v1.5`，通过 ONNX 在本地运行）用 Reciprocal Rank Fusion（RRF）融合。它是在 TF-IDF、BM25、纯向量检索之间经过实测对比选出来的，详见下方的[检索器对比](#检索器对比)。
- **经过校准的证据门槛（evidence gate）：** 离题问题在调用 LLM 之前就会被拒答。阈值基于向量相似度，根据评估报告中的阈值扫描结果确定。
- **基于 Claude 的有据生成：** 证据以 XML 形式传给模型，有助于抵御 prompt injection。回答附带引用，拒答可以被程序识别。已启用服务端拒答回退（refusal fallback）；没有 API key 或 API 调用失败时，系统会降级为本地检索预览。
- **生产级工程实践：** 类型化配置（pydantic-settings）、带 request id 的结构化 JSON 日志、启动时只加载一次索引并区分存活/就绪探针的 FastAPI 服务、多阶段构建且以非 root 用户运行的 Docker 镜像、`uv` 锁文件、ruff、strict mypy、带覆盖率的 pytest、pre-commit，以及 GitHub Actions。

## 架构

```text
 data/documents/*.md|pdf ──► ingest ──► 分块 ──► chunks.json + embeddings.npz（bge-small，已缓存）
                                                                     │
 问题 ──┬─► BM25（精确词、数字）────┐
        └─► 向量余弦（同义改写）────┴─► RRF 融合 ──► top-k
                                                                     │
                          证据门槛：最佳向量相似度 < 0.52 ？──┬── 是 ──► 拒答（不调用 LLM）
                                                              │
                                                              └── 否 ──► Claude（XML 包裹的上下文）
                                                                     │   └── 出错 ─► 本地预览
                                                                     ▼
                                                   回答 + 引用 + 延迟 + token 用量

 evals/golden_set.jsonl ──► retail-rag eval ──► 指标 ──► reports/eval.{md,json} ──► CI 门槛
```

## 快速开始

```bash
make install          # uv sync --all-extras + 安装 pre-commit hooks
make ingest           # 构建 data/index/index.json
uv run retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?"
```

第一次执行 `ingest` 会下载向量模型（约 70 MB），并把分块向量缓存在索引旁边。如果需要完全离线运行，设置 `RAG_RETRIEVER=bm25`；如果要使用预先下载好的模型目录，设置 `RAG_EMBEDDING_MODEL_PATH`。

没有 `ANTHROPIC_API_KEY` 时，查询只返回本地检索预览。要获得基于证据的 Claude 回答：

```bash
cp .env.example .env   # 填入 ANTHROPIC_API_KEY；切勿提交 .env
uv run retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?" --json
```

### API

```bash
make serve                                   # 或：docker compose up --build
curl localhost:8000/health                   # 存活探针
curl localhost:8000/ready                    # 就绪探针（索引是否已加载）
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"How quickly must Class A recall stock be removed?"}'
```

`/query` 的响应包含 `answer`、`citations`、`generated_by`（`claude` / `local` / `local_fallback`）、`refused` 和 `latency_ms`。每个响应都带有 `x-request-id` 响应头，日志里也会出现同一个 id。

## 评估

```bash
make eval                                    # 与 CI 相同的质量门槛
uv run retail-rag eval --k 6 --chunk-size 600 --stem k6_c600   # 做实验
uv run retail-rag eval --judge --stem claude  # 需要 ANTHROPIC_API_KEY：生成 + LLM 评分
```

| 指标 | 衡量什么 |
|---|---|
| Hit rate@k / Recall@k | 正确的证据有没有送到模型面前？ |
| MRR / nDCG@k | 正确的证据是否排在靠前的位置？ |
| 答案准确率 | 回答是否包含预期的事实？ |
| 拒答准确率 | 对语料无法回答的问题，系统是否会拒答？ |
| 引用有效性 | 回答里引用的每个文件是否都在检索结果中？否则就是编造的引用。 |
| 忠实度 | LLM-as-judge（结构化输出）：回答中的每个论断是否都有证据支持？ |

黄金测试集的格式说明见 [`evals/README.md`](evals/README.md)，各检索器的基线报告提交在 [`evals/baselines/`](evals/baselines/)。

### 检索器对比

所有实验都使用同样的 55 道题、`k=4`、本地模式，以及各检索器校准后的门槛。完整报告在 [`evals/baselines/`](evals/baselines/)，执行 `make compare` 可以重新生成。

| 检索器 | Hit@4 | Recall@4 | MRR | nDCG@4 | 改写问题 Hit@4 | 拒答准确率 | 误拒 | 通过率 | p50 延迟 |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF（基线） | 0.955 | 0.932 | 0.917 | 0.954 | 0.778 | 0.000 | 0 | 0.764 | 0.1 ms |
| BM25 | 0.977 | 0.955 | 0.932 | 0.967 | 0.889 | 0.182 | 0 | 0.818 | 0.03 ms |
| 向量检索（bge-small） | 0.955 | 0.955 | 0.884 | 0.922 | 0.889 | 0.273 | 0 | 0.818 | 2.9 ms |
| **混合检索（BM25 + 向量，RRF）** | **0.977** | **0.955** | **0.932** | **0.969** | **0.889** | **0.273** | **0** | **0.836** | 2.5 ms |

结论：

1. **关键词检索应付不了同义改写。** 用 TF-IDF 时，"freezer" 匹配不到 "frozen"。向量检索能处理改写，但会把精确事实排得更靠后（MRR 0.884）。混合检索兼顾两者的长处，因此设为默认。
2. **单独的 BM25 就是比 TF-IDF 更强的基线。** 仅靠去停用词和文档长度归一化，不用任何模型，就把改写问题的命中率从 0.78 提到了 0.89。这说明之前是 TF-IDF 这个基线偏弱，而不是关键词检索本身不行。
3. **相似度阈值能区分离题问题，但区分不了"硬负例"。** [阈值扫描](evals/baselines/hybrid_local.md#evidence-gate-threshold-sweep)显示：
   - 明显无关的问题（法国首都、写诗请求、prompt injection）得分 ≤ 0.49，而所有可回答的问题得分 ≥ 0.56。门槛设在 0.52，能在不调用 LLM、零误拒的情况下拒掉这 3 题。
   - 近领域问题会通过门槛，因为它们本身就是语料恰好没覆盖的零售或金融问题："RBA 现金利率"得分 0.63，"Harbourline 在西澳的门店数"得分 0.65。
   - "Coles **FY24** 营收"得分高达 **0.78**，比大多数真实问题还高：它的话题完全相关，只是年份不对。而 TF-IDF 连离题问题都分不开，"What is the capital of France?" 仅凭停用词就得了 0.335。
   - 因此拒答是分层的：低成本的检索门槛负责离题问题；近领域、实体错误、时间错误的问题交给生成模型的拒答约定处理，并在 Claude 模式下用 `refusal_accuracy` 衡量。
4. **唯一剩下的可回答漏检**是一个改写问题："someone stealing" 与 "shoplifter"。下一步的改进手段是 cross-encoder reranker。

测试集只有 55 题，一道题大约相当于任一指标的 2 个百分点。低于这个幅度的差异应视为噪声；在调整融合权重之前，应先扩充测试集。

## 工程实践

| 关注点 | 实现方式 |
|---|---|
| 依赖管理 | `uv` + 提交到仓库的 `uv.lock`；可选依赖组 `api`、`llm`、`pdf`、`embeddings`、`dev` |
| 配置 | 带校验的 `pydantic-settings`（`src/retail_rag/config.py`）；密钥以 `SecretStr` 保存 |
| 代码质量 | `ruff`（lint + 格式化 + bandit 安全规则）、`mypy --strict`、`pre-commit` |
| 测试 | 91 个 pytest 测试，覆盖单元、API、CLI、评估和检索器。Claude 和向量模型均为模拟实现，测试离线、结果确定。另有一个集成测试在 CI 中调用真实的 bge-small 模型。CI 覆盖率门槛为 85% |
| CI | lint → 测试（py3.11/3.12）→ 评估门槛（报告显示在 job summary）→ Docker 构建 + 冒烟测试；另有手动触发的 Claude 实测评估任务 |
| 容器 | 多阶段构建、非 root 用户、索引内置于镜像、对 `/ready` 做 `HEALTHCHECK`、JSON 日志 |
| 可观测性 | 带 `request_id` 的结构化日志、分阶段延迟、token 用量、`generated_by`、`refused` |

```bash
make check   # lint + 类型检查 + 测试 + 评估门槛：与 CI 完全一致
```

## 仓库结构

```text
src/retail_rag/
  config.py          从环境变量 / .env 读取的类型化配置
  logging_config.py  文本或 JSON 结构化日志
  chunking.py        确定性分块
  retrieval/         TF-IDF、BM25、向量、混合（RRF）检索器；向量模型；工厂方法
  ingest.py          markdown/文本/PDF 导入
  generation.py      Claude 生成（XML 上下文、拒答约定、降级）
  pipeline.py        流程编排、分数门槛、延迟/用量统计
  api.py             FastAPI 服务（启动时加载索引、request id、探针）
  cli.py             ingest / query / serve / eval
  evaluation/        测试集 schema、指标、运行器、报告、LLM 评分
evals/               黄金测试集 + 提交到仓库的基线报告
data/documents/      公开资料快照 + 合成文档
tests/               pytest 测试集
```

## 数据与声明边界

- Coles 的数据快照附有官方来源链接。公开演示前请先刷新数据。
- 所有 Harbourline 文档均为合成数据。除非明确获得许可，交易数据必须保持为合成数据。
- 不要把本仓库描述为 Coles 的部署或客户项目。
- 不要提交 API key、私有文档、个人数据或前雇主的数据。

## 路线图

1. ~~评估框架：黄金测试集 + CI 质量门槛~~ ✅
2. ~~工程基础：uv、ruff、mypy、Docker、CI、结构化日志~~ ✅
3. ~~向量检索与 BM25/向量混合检索（RRF），并与 TF-IDF 基线对比~~ ✅
4. Cross-encoder reranker，以及在同一检索接口下接入 pgvector（HNSW）
5. 对真实年报做版面感知的 PDF 解析，支持按公司、年份、页码等元数据过滤
6. 在 Postgres 中生成合成的订单和库存数据，加一个只读 SQL 工具，以及 RAG / SQL / 混合路由
7. 链路追踪（Langfuse/OpenTelemetry）、prompt caching、流式响应、认证与限流
