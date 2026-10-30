# 方法

目标不是「跑一遍 benchmark 拿个分」，而是让两个权重之间的差异**可以被归因**。
下面每一条都是为了消掉一个混淆变量。

## 1. 协议取 checkpoint 自带的 generation_config

`temperature=1.0 / top_p=0.95 / top_k=20`，来自被测模型自己的
`generation_config.json`，不是 evalscope 或某篇论文的默认值。理由：这是这份权重
发布时预期的采样分布，也是官方卡分数的口径。用别的协议（比如 `temperature=0` 贪心）
测出来的绝对分数没有可比对象。

代价是自己实现 HTTP 客户端，不能直接用 `evalscope eval` 跑全部基准——但换来的是
采样参数、prompt、预算、并发全部可控。

## 2. top_k 必须验证真的下发了

这是最容易踩的坑：vLLM 的 OpenAI 兼容层默认只接受 `temperature / top_p / seed`，
`top_k` 这类参数要被搬进 `extra_body` 才生效。如果它被静默丢弃，两侧就都在用
top_k=0 采样，协议形同虚设。

所以先用**首 token 分布反证**，而不是相信文档：

```bash
for tk in 1 0; do
  for s in $(seq 1 20); do
    curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with just: OK\"}],
           \"max_tokens\":1,\"temperature\":5.0,\"top_p\":1.0,\"top_k\":$tk,\"seed\":$s,
           \"logprobs\":true,\"top_logprobs\":1}" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["logprobs"]["content"][0]["token"])'
  done | sort -u | wc -l
done
```

判据：`top_k=1` 时 20 次请求只应出现 **1** 种首 token；`top_k=0` 时应出现 **20** 种互不相同的。
实测两侧都满足，说明参数确实到位。

（顺带确认：evalscope 1.14.0 在构造请求时会自己把 `top_k` 挪进 `extra_body`，
源码在 `openai` api 插件的参数归一化函数里，所以走 evalscope 的 LiveCodeBench 也没丢。）

## 3. prompt 必须逐字节相同

两侧唯一的变量应该是量化。所以：

- 所有 prompt 构造只在 `scripts/protocol.py` 实现一次，两侧 import 同一份。
- GPQA 的选项顺序用 `sha256(question)` 作为种子打乱——固定、可复现、两侧一致。
  不能用 Python 全局 `random`，否则两次运行选项排列不同。
- MathVision 图片以 base64 内联进 messages，不依赖外部 URL。
- 验证方式：把两侧实际发出的 prompt 取哈希逐个比对。实测 GPQA 198/198、
  IFBench 252/252、MathVision 40/40 全部一致。

## 4. 输出预算 = 上下文长度 − prompt

不设固定 `max_tokens`。若写死 261000，遇到 `261145 + prompt > 262144` 会被服务端
直接 400；若设小了，长推理题会被截断，判分器把「没答完」记成「答错」，制造虚假差异。

`protocol.py::max_tokens_for` 的做法：prompt token 数按 `chars/2` **高估**
（真实 Qwen 分词器约 2–4 chars/token），因此预算永远落在服务端上限内，
对真实 prompt 仍接近 260k。

结果：全部 2 万余次生成的截断数 **0**。

## 5. 判分器必须固定

全部复用 evalscope 自带评分函数，不引入 LLM judge：

| 基准 | 判分函数 |
|---|---|
| GPQA-Diamond | `evalscope.utils.multi_choices.parse_answers` |
| IFBench | `evalscope.benchmarks.ifbench.evaluation_lib.process_results`（prompt_level_strict） |
| MathVision | `evalscope.metrics.math.parser.extract_answer` + `math_equal` |
| LiveCodeBench v6 | Docker 沙箱跑单测 |

判分器本身不能成为变量。LLM judge 会引入第二个模型的偏好，且不可复现。

## 6. 判据是配对检验，不是绝对准确率差

这是整套方法的核心。同一配置换种子重跑整轮，分数就会漂 1–2 分——和量化造成的差异
同量级。所以比绝对差值没有信息量。

逐题配对 McNemar 只看**两侧不一致的题**，问的是「把 W8A8 对的题换成 BF16 会怎样，
反过来呢」。整轮抖动（两侧同时答对或同时答错）被完全抵消，剩下的是真正的方向性偏移。

用精确二项检验，不做正态近似（不一致对数常常只有十几，近似不成立）。

多重比较：一次测 7 组（4 基准 × 轮次），Bonferroni 阈值 = 0.05/7 ≈ 0.0071。
本测试唯一低于常规 0.05 的 LiveCodeBench（p=0.013）在**校正后不显著**，
所以定性为「强烈提示、待复现」，不能写成结论。

## 7. PPL 哨兵：比准确率先亮的灯

基准准确率是离散的，分辨率被题数锁死——198 题时一格就是 0.5 分。量化造成的损伤
常常小于这一格，于是三项指标全在噪声里，看不出任何东西。

NLL 是连续量。做法是用 vLLM 的 `prompt_logprobs` 对固定语料做 teacher forcing，
算每 token 的平均负对数似然，两侧比 Δ。不需要本地加载权重，纯 HTTP。

读法比数值重要：

| 形状 | 含义 |
|---|---|
| Δ 大 + 逐 chunk 全变差 | 真信号 |
| Δ 小 + chunk 方向对半 | 噪声底 |

本测试代码域 +8.4%、10/10 全变差 → 真信号；散文域 +0.6%、6/10 → 噪声底。
**只看总 Δ 的绝对值会误判**，必须看逐 chunk 的方向是否单调。

## 8. LiveCodeBench 走 evalscope

它的判分需要 Docker 沙箱跑单测，自己实现不现实。两个坑：

- 数据集放本地（`--dataset-args '{"live_code_bench":{"local_path":...}}'`），不走网络。
- 用 `--use-cache` 续跑时必须**显式加 `--rerun-review`**：评估指纹（`--limit` 不同）
  变了，否则会复用上一次的判分结果，拿到错的分数。

```bash
evalscope eval --model <名称> --api-url http://127.0.0.1:<端口>/v1 --api-key EMPTY \
  --eval-type openai_api --datasets live_code_bench \
  --dataset-args '{"live_code_bench":{"local_path":"/path/to/lcb-local","subset_list":["default"]}}' \
  --generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"max_tokens":259000}' \
  --timeout 7200 --repeats 3 --eval-batch-size 16 --use-cache --rerun-review
```

注意 `--limit 200` 会取到数据集**前段的简单题**，pass@1 虚高到 98%；必须跑全量 1055。

## 9. 其它踩过的坑

- **一次性 `pool.submit` 全部 future 会卡住**。1000+ 题规模下必须用滚动窗口
  （`wait(..., FIRST_COMPLETED)`，保持固定数量在飞），见 `run_eval.py::run`。
- **MathVision 抽样要按真实难度分布分层**。原做法是每个 level 各抽 40 题，
  level 1 被超采 2.5 倍，系统偏难。当前 399 题按数据集真实 level 占比分配。
- **引擎版本必须两侧一致**。`vllm-ascend v0.30.0rc2` 每 40–50 分钟必挂
  （TP 集合通信死锁），换到 `v0.23.0` 后 70+ 小时无故障。版本不同会引入
  「精度差异到底来自量化还是引擎」的混淆，所以必须同镜像。
- **`pkill -f <脚本名>` 会杀掉自己的 SSH 命令**（命令行里含同样字符串），用 PID。
