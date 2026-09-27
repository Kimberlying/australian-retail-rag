# Australian Retail RAG

[![English](https://img.shields.io/badge/lang-English-lightgrey.svg)](README.md)
[![简体中文](https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue.svg)](README.zh-CN.md)
[![CI](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml)

一个面向澳大利亚零售行业、以评估为核心构建的问答系统。它既能从文档中回答（检索增强生成，RAG），也能从运营数据库中回答（text-to-SQL），并会把每个问题路由到合适的数据源。数据来源是澳大利亚上市公司的公开资料，以及明确标注为合成数据的零售文档和数据。

这是一个独立的作品集项目，**不是 Coles 的系统、客户项目，也不代表 Coles 的背书**。合成文档和数据中的 "Harbourline Retail" 是一家虚构公司。

## 亮点

- **评估优先：** 230 道题的黄金测试集（golden set），覆盖检索、拒答、路由和 SQL 回答；另有 38 道题的留出（held-out）路由测试集。下面每一个设计选择都有实测的前后对比，质量下降时 CI 会失败。
- **混合检索 + 可选重排序：** BM25 与向量检索（`bge-small-en-v1.5`，本地 ONNX 运行）用 Reciprocal Rank Fusion（RRF）融合，可选 cross-encoder 重排序（`ms-marco-MiniLM-L-6-v2`）。向量部分可以在内存中运行，也可以放在 **Postgres + pgvector（HNSW）** 中，两者共用同一个检索接口。
- **分层拒答：** 自查询（self-query）**元数据过滤**在不调用 LLM 的情况下拒答公司错误和时间段错误的问题（例如 "Woolworths EBIT"、"Coles FY24 营收"）；经过校准的**证据门槛**拒答离题问题；剩下的交给**生成模型的拒答约定**。每次拒答都会记录是哪一层触发的。
- **文档 + 数据：** 基于规则的路由器把问题发往文档、一个带固定随机种子的**合成运营数据库**（订单、库存、损耗），或两者兼用。Claude 在 tool-use 循环中编写 SQL，通过一个多层防护的**只读 SQL 工具**执行。
- **按页解析 PDF：** 一份合成的 FY25 年报按页解析，并去除每页重复的页眉页脚；引用附带页码；公司和财年等元数据来自文档头部的 front matter 或旁挂的 sidecar 文件。
- **生产级服务：** 基于 server-sent events 的流式回答、稳定前缀的 prompt caching、常量时间比较的 API key 认证、按客户端限流、带 GenAI 属性的 OpenTelemetry 链路追踪（兼容 Langfuse）、结构化 JSON 日志，以及存活和就绪探针。
- **工程基础：** `uv` 锁文件、ruff、strict mypy、186 个测试（覆盖率 92%，CI 中包含真实向量模型和真实 Postgres 的集成测试）、多阶段构建且以非 root 用户运行的 Docker 镜像、GitHub Actions。

## 架构

```text
 data/documents/*.md|pdf ─► ingest ─► 按页 ─► 分块（+ company、fiscal_year、page）
                              │                  │
                              │                  ├─► BM25 索引（内存）
                              │                  └─► 向量：numpy（默认）或 pgvector HNSW
                              └─► 固定种子的合成 SQLite：门店、商品、订单、库存、损耗

 问题 ─► 路由器 ─┬─ rag ────► 元数据过滤 ─► 混合检索（BM25 + 向量，RRF）
                 │                 │ 无匹配            └─► 可选 cross-encoder 重排序
                 │                 ▼                                 │
                 │          拒答（不调用 LLM）   证据门槛：最佳相似度 < 0.575 ？
                 │                                    │ 是：拒答（不调用 LLM）  │ 否
                 │                                    ▼                         ▼
                 │                                               Claude（XML 证据，流式输出）
                 ├─ sql ────► Claude tool-use 循环 ─► run_sql（只读）─► 回答 + 展示 SQL
                 └─ hybrid ─► 检索相关政策 ─► SQL agent 把政策应用到数据上

 每个环节 ─► OpenTelemetry spans + JSON 日志（request id、路由、拒答原因、token、延迟）
 evals/*.jsonl ─► retail-rag eval ─► 指标 + 报告 ─► CI 质量门槛
```

## 快速开始

```bash
make install          # uv sync --all-extras + 安装 pre-commit hooks
make ingest           # 构建文档索引 + 合成运营数据库
uv run retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?"
uv run retail-rag query "How quickly must Class A recall stock be removed?" --stream
```

第一次执行 `ingest` 会下载向量模型（约 70 MB），并把分块向量缓存在索引旁边。如果需要完全离线运行，设置 `RAG_RETRIEVER=bm25`；如果要使用预先下载好的模型目录，设置 `RAG_EMBEDDING_MODEL_PATH`。

没有 `ANTHROPIC_API_KEY` 时，文档类问题返回本地检索预览，数据库类问题会说明需要 Claude 来编写 SQL。要获得 Claude 的回答：

```bash
cp .env.example .env   # 填入 ANTHROPIC_API_KEY；切勿提交 .env
uv run retail-rag query "Which store had the highest sales revenue in July 2026?" --json
```

### API

```bash
make serve                                   # 或：docker compose up --build
curl localhost:8000/health                   # 存活探针
curl localhost:8000/ready                    # 就绪探针：索引、SQL、认证、追踪状态
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -H 'x-api-key: <key>' -d '{"question":"How quickly must Class A recall stock be removed?"}'
curl -N -X POST localhost:8000/query/stream -H 'content-type: application/json' \
  -d '{"question":"What happens to chilled stock after a cold chain breach?"}'
```

`/query` 返回 `answer`、`citations`（来源、页码、相关度）、`route`、`sql_queries`、`generated_by`、`refused`、`refusal_reason`、`filters`、`latency_ms` 和 `usage`（包括 prompt cache 读取量）。`/query/stream` 以 server-sent events 依次发送 `meta`（路由和引用）、若干 `token` 增量，最后是带完整结果的 `done`。

- **认证：** 设置 `RAG_API_KEYS=key1,key2`，客户端通过 `x-api-key` 或 `Authorization: Bearer` 发送 key。未配置 key 时认证关闭，仅用于本地开发。
- **限流：** 每个 API key（未开启认证时按客户端地址）一个令牌桶，用 `RAG_RATE_LIMIT_PER_MINUTE` 和 `RAG_RATE_LIMIT_BURST` 配置。超出后返回 `429` 和 `Retry-After`。认证失败的请求也会消耗调用方的额度，所以猜 key 会被限速。
- **链路追踪：** `RAG_TRACING=otlp` 通过标准的 `OTEL_EXPORTER_OTLP_*` 变量把 spans 导出到 Jaeger、Tempo 或 Langfuse（见 `.env.example`）。
- **pgvector：** 执行 `docker compose --profile pgvector up`，然后设置 `RAG_VECTOR_STORE=pgvector` 和 `RAG_DATABASE_URL`。

## 评估

```bash
make eval                                             # 与 CI 相同的质量门槛
make compare                                          # 所有检索器 + 消融实验 -> evals/baselines/
uv run retail-rag eval --reranker cross-encoder       # 试用重排序
uv run retail-rag eval --no-metadata-filters          # 消融实验
uv run retail-rag eval --judge --stem claude          # 需要 ANTHROPIC_API_KEY：生成、SQL、LLM 评分
```

| 指标 | 衡量什么 |
|---|---|
| Hit rate@k / Recall@k | 正确的证据有没有送到模型面前？ |
| MRR / nDCG@k | 正确的证据是否排在靠前的位置？ |
| 拒答准确率 / 误拒率 | 语料无法回答的问题是否拒答，能回答的问题是否没有被误拒？ |
| 拒答分层 | 每道无法回答的题是被哪一层拒掉的：元数据过滤、证据门槛，还是模型 |
| 路由准确率 | 在黄金测试集上，以及在从未用于调参的留出集上，RAG / SQL / hybrid 路由是否正确 |
| SQL 执行准确率 | agent 最后执行的查询是否返回与标准 SQL 相同的结果行？（Claude 模式） |
| 答案准确率、引用有效性、忠实度 | 生成质量；忠实度由 LLM 以结构化输出评分（Claude 模式） |

黄金测试集的格式说明见 [`evals/README.md`](evals/README.md)，基线报告提交在 [`evals/baselines/`](evals/baselines/)。

### 结果

除特别说明外，所有实验都使用同样的 230 道题、`k=4`、本地模式（不生成回答），并开启元数据过滤。完整报告在 [`evals/baselines/`](evals/baselines/)。

| 配置 | Hit@4 | Recall@4 | MRR | nDCG@4 | 改写问题 Hit@4 | 拒答准确率 | 误拒率 | 通过率 |
|---|---|---|---|---|---|---|---|---|
| TF-IDF（基线） | 0.933 | 0.922 | 0.870 | 0.876 | 0.577 | 0.269 | 0.000 | 0.861 |
| BM25 | 0.950 | 0.939 | 0.892 | 0.896 | 0.692 | 0.346 | 0.006 | 0.883 |
| 向量检索（bge-small） | 0.956 | 0.944 | 0.853 | 0.868 | 0.923 | 0.500 | 0.000 | 0.904 |
| **混合检索（BM25 + 向量，RRF）：默认** | **0.978** | **0.964** | **0.900** | **0.908** | **0.885** | **0.538** | **0.000** | **0.926** |
| 混合检索，关闭元数据过滤 | 0.978 | 0.964 | 0.898 | 0.907 | 0.885 | 0.269 | 0.000 | 0.896 |
| 混合检索 + cross-encoder 重排序 | 0.994 | 0.986 | 0.940 | 0.945 | 0.962 | 0.500 | 0.000 | 0.935 |

- **路由：** 黄金测试集上 0.996，**留出集上 0.947**。规则是对照黄金测试集写的，所以留出集的数字才是客观的。
- **延迟（GitHub Actions runner，2 vCPU）：** 混合检索 p50 为 7.7 ms；加上重排序后 p50 为 458 ms，时间花在给 20 个"问题–分块"对打分上。

### 结论

1. **元数据过滤几乎零成本地让拒答准确率翻倍。** 关闭过滤时，混合检索拒掉 0.269 的无法回答问题；开启后为 0.538，检索质量不变，且零误拒。过滤能拦住相似度拦不住的情况："Coles FY24 营收"和"Coles FY25 营收"的向量几乎一样，但语料里没有 Coles 的 FY24 文档，所以过滤后没有任何可用证据。
2. **重排序是提升检索最有效的手段，但有延迟代价。** 它把 MRR 从 0.900 提到 0.940，把改写问题命中率从 0.885 提到 0.962，并修复了剩下 4 个检索漏检中的 3 个，例如 "never picks up their online order" 对应 "uncollected orders are cancelled"。代价是每次查询约 450 ms 的 CPU 时间，所以默认关闭。当 Claude 生成本身已经占据大部分延迟，或者有 GPU 时，值得开启。
3. **重排序也会影响证据门槛。** 门槛看的是实际返回证据的向量相似度，而重排序返回的是不同的证据。因此有一个离题问题（"RBA 现金利率"）通过了门槛，拒答准确率因而是 0.500 而不是 0.538。门槛需要按配置分别校准，每份报告里的阈值扫描表都展示了这个权衡。
4. **语料变大后，门槛的余量在缩小。** 7 份文档时，离题问题最高得分 0.49，可回答问题最低 0.56；12 份文档时分别为 0.56 和 0.59。门槛因此从 0.52 重新校准为 0.575。这就是评估报告包含阈值扫描、并且每次语料变化都要重新检查门槛的原因。
5. **领域内的无法回答问题需要模型来判断。** "Harbourline 的股价"、"CEO 的名字"这类问题得分 0.68–0.78，比很多真实答案还高。经过两层低成本拒答后，26 道无法回答的题里还有 12 道要交给 Claude 的拒答约定处理，这部分在 Claude 模式下衡量。
6. **关键词检索应付不了同义改写。** 改写问题命中率：TF-IDF 0.58、BM25 0.69、向量 0.92、加重排序 0.96。混合检索（0.885）在改写问题上比纯向量检索略低，换来整体最好的排序质量。
7. **发现并修复了一个指标 bug。** 之前的 nDCG 会给每个匹配同一标签的分块都记分，所以两个包含相同数字的分块会让 nDCG 超过 1。现在每个标签只记一次分，并加了回归测试。本 README 早期版本中的 nDCG 数值因这个 bug 偏高。

测试集有 230 题，一道题大约相当于任一指标的 0.4 个百分点（在 180 道可回答的文档问题上约 0.6 个百分点，在 26 道无法回答的问题上约 3.8 个百分点）。低于这个幅度的差异应视为噪声。

## 设计决策

| 决策 | 原因 | 代价 |
|---|---|---|
| BM25 + 向量混合检索，用 RRF 融合 | RRF 融合的是排名，BM25（无上界）和余弦相似度（[0, 1]）之间不需要分数归一化或调参 | 改写问题上略低于纯向量检索 |
| 重排序默认关闭 | 在 CPU 上检索延迟约增加 60 倍，换来 MRR 提升 4 个点 | 延迟允许时设置 `RAG_RERANKER=cross-encoder` 开启 |
| 门槛基于向量相似度，而不是重排序分数 | 余弦相似度是经过校准且稳定的；cross-encoder 的 logits 不是 | 门槛结果仍取决于返回了哪些证据 |
| 公司过滤严格；财年过滤保留无日期文档 | 政策文档没有时效性："FY25 缺货率与政策目标相比如何"既需要 FY25 年报，也需要无日期的政策文档 | Harbourline 年份错误的问题仍会走到门槛和模型那一层 |
| 基于规则的路由器 | 即时、免费、确定性、可解释（每个决策都列出触发的规则） | 留出集准确率 0.947；升级方向是 LLM 分类器 |
| SQL 工具多层防护 | 模型写的 SQL 视为不可信输入：`mode=ro` 只读连接、SQLite authorizer 白名单（只允许 SELECT、读取、函数）、每次只执行一条语句、时间上限、行数上限 | 刻意只读，不支持写回操作 |
| 带容差的执行准确率 | 多出的列、行顺序、0.5% 以内的舍入差异不应算错 | 可能接受"结果对但理由错"的查询 |
| 对稳定前缀做 prompt caching | SQL agent 约 1.1k token 的 schema 提示会被缓存，tool-use 循环每一步都复用它 | RAG 系统提示（约 200 token）低于 Claude Opus 5 的 512 token 最小值，暂时不会被缓存；`usage.cache_read_input_tokens` 可以直接看到这一点 |
| 进程内限流 | 不需要额外基础设施，单副本部署时正确 | 多副本时需要把状态移到 Redis 或网关 |

## 工程实践

| 关注点 | 实现方式 |
|---|---|
| 依赖管理 | `uv` + 提交到仓库的 `uv.lock`；可选依赖组 `api`、`llm`、`pdf`、`embeddings`、`pgvector`、`otel`、`dev` |
| 配置 | 带校验的 `pydantic-settings`（`src/retail_rag/config.py`）；密钥以 `SecretStr` 保存 |
| 代码质量 | `ruff`（lint + 格式化 + bandit 安全规则）、`mypy --strict`、`pre-commit` |
| 测试 | 186 个 pytest 测试：单元、API、CLI、评估、检索器、SQL 安全、路由、流式输出、链路追踪、认证。Claude 用脚本化的假客户端替代，测试离线、结果确定。CI 中还用真实的 bge-small 模型做集成测试，并用 Postgres service container 跑 pgvector 测试。覆盖率门槛 85%（当前 92%） |
| CI | lint → 测试（py3.11/3.12，带 Postgres + pgvector）→ 评估门槛（检索、拒答、路由、留出集路由）+ 一次仅供参考的重排序评估 → Docker 构建 + 冒烟测试；另有手动触发的 Claude 实测评估任务 |
| 容器 | 多阶段构建、非 root 用户、索引和 SQL 数据库内置于镜像、对 `/ready` 做 `HEALTHCHECK`、JSON 日志 |
| 安全 | API key 认证（常量时间比较摘要）、限流、XML 转义证据防 prompt injection、带 authorizer 的只读 SQL、日志中不出现密钥（只记录指纹） |
| 可观测性 | OpenTelemetry spans（`rag.query` → `rag.route` / `rag.retrieve` / `gen_ai.generate` / `rag.sql_agent` → `rag.sql.execute`），附带 GenAI 的 token 和缓存属性；结构化日志包含 `request_id`、路由、拒答原因、延迟和用量 |

```bash
make check   # lint + 类型检查 + 测试 + 评估门槛：与 CI 完全一致
```

## 仓库结构

```text
src/retail_rag/
  config.py          从环境变量 / .env 读取的类型化配置
  ingest.py          front matter + sidecar 元数据，按页解析 PDF
  chunking.py        确定性分块
  filters.py         自查询元数据过滤（公司、财年）
  retrieval/         TF-IDF、BM25、向量、pgvector、混合（RRF）、cross-encoder 重排序
  router.py          基于规则的 RAG / SQL / hybrid 路由器
  sql/               合成数据库、只读 SQL 工具、Claude text-to-SQL agent
  generation.py      Claude 生成：XML 上下文、拒答约定、缓存、流式输出
  pipeline.py        流程编排与分层拒答
  api.py, security.py  FastAPI 服务、认证、限流、SSE 流式输出
  tracing.py         OpenTelemetry spans（关闭时为空操作）
  cli.py             ingest / query / serve / eval
  evaluation/        测试集 schema、指标、运行器、报告、LLM 评分、留出集路由
evals/               黄金测试集、留出路由集、提交到仓库的基线报告
data/documents/      公开资料快照、合成政策文档、合成年报 PDF
scripts/             可复现生成合成年报 PDF 的脚本
tests/               pytest 测试集
```

## 数据与声明边界

- Coles 的数据快照附有官方来源链接。公开演示前请先刷新数据。
- 所有 Harbourline 文档、年报和运营数据库均为合成数据；其中的数量级仅作示意，彼此之间没有对账。除非明确获得许可，交易数据必须保持为合成数据。
- 不要把本仓库描述为 Coles 的部署或客户项目。
- 不要提交 API key、私有文档、个人数据或前雇主的数据。

## 路线图

1. ~~评估框架：黄金测试集 + CI 质量门槛~~ ✅
2. ~~工程基础：uv、ruff、mypy、Docker、CI、结构化日志~~ ✅
3. ~~向量检索与 BM25/向量混合检索（RRF）~~ ✅
4. ~~Cross-encoder 重排序，以及在同一检索接口下接入 pgvector（HNSW）~~ ✅
5. ~~按页解析 PDF，支持按公司、财年、页码等元数据过滤~~ ✅
6. ~~合成订单和库存数据、只读 SQL 工具，以及 RAG / SQL / 混合路由~~ ✅
7. ~~链路追踪（OpenTelemetry/Langfuse）、prompt caching、流式响应、认证与限流~~ ✅
8. 在 CI 中运行 Claude 模式评估（需要仓库 secret `ANTHROPIC_API_KEY`），并公布答案准确率、忠实度和 SQL 执行准确率
9. 规则不确定时回退到基于 LLM 的路由器；多副本部署时用 Redis 实现限流
10. 针对真实的多栏年报做版面感知的表格抽取
