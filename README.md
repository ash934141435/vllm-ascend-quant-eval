# Qwen3.8-27B W8A8 vs BF16 精度评测（Ascend NPU）

> 在 8×910B4 上，用官方采样协议对同一模型的两份权重做同机 A/B：一份 Ascend W8A8-dynamic
> 量化权重，一份 BF16。回答两个问题：
> 1. 量化有没有**可归因的**精度损失？
> 2. 这份量化权重在官方协议下的绝对分数是多少？

结论先放前面：**四个基准里三个与 BF16 不可区分，LiveCodeBench 有一个约 1 分的可疑损失。**

## 为什么要做成配对实验

直接比两个绝对分数没有意义。换一个随机种子重跑整轮，同一配置的分数就会漂 1–2 分，
和量化造成的差异同量级。所以这套评测的判据不是「准确率差多少」，而是：

- **逐题配对 McNemar**：只看两侧不一致的题，检验方向是否偏。绝对值抖动被抵消。
- **PPL 哨兵**：NLL 是连续量，在准确率还看不出差异时就先报警。

两侧共用同一份 prompt（逐字节相同）、同一判分函数（全部用 evalscope 自带，不引入
LLM judge）、同一批数据、同一个引擎镜像，**唯一变量是 `--quantization=ascend`**。

## 结果

在 checkpoint 自带 `generation_config.json` 的协议下
（temperature=1.0 / top_p=0.95 / top_k=20，输出预算 = 262144 − prompt，16 并发，
每样本 3 次独立采样，majority@3）：

| 基准 | n | W8A8 | BF16 | 差值 | 仅 W8A8 对 | 仅 BF16 对 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPQA-Diamond | 198 | 90.15% | 90.91% | −0.76 | 6 | 9 | 0.45 – 1.00 |
| IFBench | 300 | 75.34% | 75.67% | −0.33 | 25 | 27 | 0.65 – 1.00 |
| MathVision | 399 | 82.46% | 83.71% | −1.25 | 35 | 45 | 0.43 – 0.64 |
| LiveCodeBench v6 | 1055 | 93.27% | 94.31% | −1.04 | **3** | **14** | **0.013** |

GPQA / IFBench / MathVision 各跑 2 轮（表中为两轮均值）；LiveCodeBench 跑 1 轮全量。

PPL 哨兵（teacher forcing，同一份语料同一分块）：

| 域 | BF16 | W8A8 | ΔNLL/token | 相对 | 逐 chunk 变差 |
|---|---:|---:|---:|---:|---:|
| 代码 | 1.6110 | 1.6769 | +0.040 | **+8.4%** | **10 / 10** |
| 散文 | 5.3721 | 5.4277 | +0.006 | +0.6% | 6 / 10 |

散文域那 +0.6% 就是噪声底；代码域 +8.4% 且 10/10 单调变差，说明量化对符号与结构化
token 的损伤是**真实存在**的，只是还没在三项指标上显现。

## 结论

1. **三个基准无可归因损失。** 逐题配对 p 全部 ≥ 0.43，输赢方向随机翻转，属逐题噪声。
   绝对值上 W8A8 略低 0.3–1.3 分，落在整轮噪声内。
2. **LiveCodeBench 是唯一例外。** 3:14，p = 0.013，方向与 PPL 哨兵一致——代码域 NLL
   相对上升 8.4%，损伤在代码任务上确实传导成了掉点。但只测了一轮，且 7 组检验做
   Bonferroni 校正（阈值 0.0071）后不显著，定性为**强烈提示、待复现确认**。
3. **与官方卡的分差不来自量化。** 同协议下 BF16 与 W8A8 表现几乎相同（官方卡数据取自 Qwen3.8-27B 官方 model card，HuggingFace，公开可查）：

   | 基准 | 本测试 | 官方卡 | 差异来源 |
   |---|---:|---:|---|
   | GPQA-Diamond | ~90.5% | 89.2 | 已达 |
   | LiveCodeBench v6 | ~93.8% | 90.3 | 高于官方 |
   | IFBench | ~75.5% | 79.5 | 采样协议细节 / 判分口径 |
   | MathVision | ~83.0% | 90.0 | 数据切片（本仓库用 399 题分层抽样） |

   后两项的差距在 BF16 上同样存在，不是量化引入的。
4. **引擎稳定性。** 同一镜像连续压测 70+ 小时、2 万余次生成，零崩溃零抢占零截断。
   过程中确认 `vllm-ascend v0.30.0rc2` 会每 40–50 分钟必挂（TP 集合通信死锁），
   与量化权重无关，故两侧统一退到 `v0.23.0` 同镜像。

## 实验环境

| 类别 | 配置 |
|---|---|
| NPU | 8 × Ascend 910B4，单卡 64 GiB HBM，测试用其中 4 卡（TP=4） |
| CPU / 内存 | 192 核 aarch64 / 1006 GiB |
| OS | openEuler 22.03 LTS，kernel 5.10.0-60.139.0.166.oe2203.aarch64 |
| 驱动 / 固件 | 26.0.rc1 / 9.0.0.0.205 |
| CANN | 8.3.RC2 |
| 推理引擎 | vLLM-Ascend **v0.23.0**（两侧同镜像 `quay.io/ascend/vllm-ascend:v0.23.0`） |
| 评测框架 | evalscope 1.14.0 / Python 3.12，纯 HTTP 客户端，不占 NPU |

两侧启动参数逐字相同，唯一差异是末行：

```bash
vllm serve <权重路径> \
  --host 0.0.0.0 --port <端口> --served-model-name <名称> \
  --max-model-len 262144 --max-num-batched-tokens 16384 --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 4 --data-parallel-size 1 \
  --trust-remote-code --enable-prefix-caching \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  [--quantization=ascend]        # 仅 W8A8 侧
```

## 复现

```bash
pip install -r requirements.txt

# 1) 采样参数冒烟：确认 top_k 真的下发到引擎（用首 token 分布反证）
#    top_k=1 → 20 次请求只有 1 种首 token；top_k=0 → 20 种全不同
#    确认细节见 docs/methodology.md

# 2) 全量跑（每侧每轮一个 jsonl，支持断点续跑）
python scripts/run_eval.py --base-url http://127.0.0.1:8000/v1/chat/completions \
    --model qwen3.8-27b-bf16 --tag bf16 --round 1 --datasets gpqa,ifbench,mv

python scripts/run_eval.py --base-url http://127.0.0.1:8001/v1/chat/completions \
    --model qwen3.8-27b-w8a8 --tag w8a8 --round 1 --datasets gpqa,ifbench,mv

# 3) 判分 + 配对检验
python scripts/score.py --tags w8a8,bf16 --rounds 1,2

# 4) PPL 哨兵（连续值，先于准确率报警）
bash scripts/make_corpora.sh        # 生成 corpus_code.txt / corpus_prose.txt
python scripts/ppl_sentinel.py --base-url http://127.0.0.1:8000/v1 \
    --model qwen3.8-27b-bf16 --label bf16 --text corpus_code.txt --chunks 10 --out ppl_bf16_code.json
python scripts/ppl_sentinel.py --base-url http://127.0.0.1:8001/v1 \
    --model qwen3.8-27b-w8a8 --label w8a8 --text corpus_code.txt --chunks 10 --out ppl_w8a8_code.json
python scripts/ppl_compare.py --ref ppl_bf16_code.json --cand ppl_w8a8_code.json
```

LiveCodeBench 走 evalscope（判分需要 Docker 沙箱），命令见 [docs/methodology.md](docs/methodology.md)。

## 目录

```
scripts/protocol.py       官方协议常量 + prompt 构造 + 输出预算（两侧共用同一份）
scripts/run_eval.py       全量跑，滚动窗口并发 + 断点续跑
scripts/score.py          判分 + Wilson CI + 逐题配对 McNemar
scripts/ppl_sentinel.py   单侧 PPL 测量（prompt_logprobs，不需加载权重）
scripts/ppl_compare.py    两侧逐 chunk 方向对比
scripts/make_corpora.sh   生成 PPL 哨兵语料（TinyStories + Python stdlib）
results/summary.json      12 次运行的聚合结果
results/ppl_chunks.csv    逐 chunk NLL 原始数据
results/mv_sample_ids.json  MathVision 399 题抽样 id（两侧共用）
docs/methodology.md       协议定版、判分口径、为什么用配对检验
docs/results.md           逐轮完整结果表
```

## 局限（不外推）

- 结论仅限 **TP=4 / 4 张 910B4 / 上述引擎版本 / 上述采样协议**。
- MathVision 只跑 **399 题分层抽样**（按数据集真实 level 分布），不是全量 3040。
- LiveCodeBench **只有一轮**，这也是它的 p 值只能算「提示」而非「结论」的原因。
- 不回答「W8A8 比 BF16 快多少」——那是另一个仓库的事：
  [vllm-ascend-performance-lab](https://github.com/ash934141435/vllm-ascend-performance-lab)。

## 免责声明

本实验为个人研究项目，不包含任何客户数据或公司内部资产；所用模型、数据集与基准均为公开资源。结论仅反映上述特定版本、镜像与硬件组合下的测量结果，不代表任何厂商的官方立场。
