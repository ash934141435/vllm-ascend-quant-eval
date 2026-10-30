#!/usr/bin/env python3
"""判分 + Wilson CI + 逐题配对 McNemar。

    python score.py --tags w8a8,bf16 --rounds 1,2 --out-dir results

判分全部复用 evalscope 自带函数，不引入 LLM judge：判分器本身不能成为变量。
两侧用同一判分器、同一份数据、同一 prompt，剩下的差异才可能归因到量化。
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset                                           # noqa: E402
from evalscope.benchmarks.ifbench.evaluation_lib import process_results  # noqa: E402
from evalscope.metrics.math.parser import extract_answer, math_equal     # noqa: E402
from evalscope.utils.multi_choices import parse_answers                # noqa: E402

from protocol import gpqa_target                                      # noqa: E402
from run_eval import find_arrow                                       # noqa: E402

TRUNC = ("max_tokens", "length")
BENCHES = ("gpqa", "ifbench", "mv")


class _State:
    """parse_answers 只用到 .choices / .output，给个最小壳子即可。"""

    def __init__(self, n):
        self.choices = [None] * n
        self.output = None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    """精确双侧 McNemar：只在不一致的对上做二项检验。

    这是判断"量化有没有真损失"的正确工具——绝对准确率差会被整轮噪声淹没，
    而不一致对的分布方向不受整轮抖动影响。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_rows(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out.setdefault(str(r["key"]), []).append(r)
    return out


def truncated(r):
    return r.get("stop_reason") in TRUNC or r.get("finish_reason") in TRUNC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="w8a8,bf16")
    ap.add_argument("--rounds", default="1,2")
    ap.add_argument("--full-dir", default="results/full")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data-dir", default="~/.cache/modelscope/hub/datasets")
    ap.add_argument("--mv-mode", default="rep399")
    ap.add_argument("--mv-ids", default="results/mv_sample_ids.json")
    a = ap.parse_args()

    tags = [x.strip() for x in a.tags.split(",") if x.strip()]
    rounds = [int(x) for x in a.rounds.split(",") if x.strip()]

    gpqa_ds = Dataset.from_file(find_arrow(a.data_dir, "AI-ModelScope___gpqa_diamond", "gpqa_diamond-train.arrow"))
    ifb_ds = Dataset.from_file(find_arrow(a.data_dir, "allenai___if_bench_test", "if_bench_test-train.arrow"))
    mv_ds = Dataset.from_file(find_arrow(a.data_dir, "evalscope___math_vision", "math_vision-test.arrow"))
    ifb_by = {str(ifb_ds[i]["key"]): ifb_ds[i] for i in range(len(ifb_ds))}
    mv_by = {str(mv_ds[i]["id"]): mv_ds[i] for i in range(len(mv_ds))}
    gpqa_tgt = {str(i): gpqa_target(gpqa_ds[i]) for i in range(len(gpqa_ds))}

    def score(bench, key, text):
        if bench == "gpqa":
            got = parse_answers(_State(4), multiple_correct=False, completion=text)
            return len(got) == 1 and next(iter(got)) == gpqa_tgt[key]
        if bench == "ifbench":
            d = ifb_by[key]
            doc = {"key": d["key"], "prompt": d["prompt"],
                   "instruction_id_list": d["instruction_id_list"], "kwargs": d["kwargs"]}
            try:
                return process_results(doc, [text])["prompt_level_strict"] >= 1.0
            except Exception:
                return False
        rec = mv_by[key]
        if len(rec["options"]) > 0:
            got = parse_answers(_State(len(rec["options"])), multiple_correct=False, completion=text)
            return "".join(sorted(got)) == str(rec["answer"])
        try:
            return math_equal(extract_answer(text), str(rec["answer"]))
        except Exception:
            return False

    results = {}
    for bench in BENCHES:
        for tag in tags:
            for rd in rounds:
                per = load_rows(os.path.join(a.full_dir, f"{tag}_{bench}_r{rd}.jsonl"))
                flags, trunc_n = {}, 0
                for key, rows in per.items():
                    if len(rows) < 3:          # 不足 3 次采样的题不计分，避免半截数据
                        continue
                    f = [score(bench, key, r.get("content") or "") for r in rows]
                    flags[key] = sum(f) * 2 > len(f)   # majority@3
                    if any(truncated(r) for r in rows):
                        trunc_n += 1
                n = len(flags)
                k = sum(1 for v in flags.values() if v)
                lo, hi = wilson(k, n)
                results[(bench, tag, rd)] = {"n": n, "ok": k, "acc": k / n if n else None,
                                             "ci": [lo, hi], "flags": flags, "trunc_items": trunc_n}

    lines = ["# 官方协议全量实测\n",
             "| 基准 | 侧 | 轮 | n | 正确 | 准确率 | 95% CI | 截断题 |",
             "|---|---|---|---|---|---|---|---|"]
    for bench in BENCHES:
        for tag in tags:
            for rd in rounds:
                r = results[(bench, tag, rd)]
                if not r["n"]:
                    lines.append(f"| {bench} | {tag} | {rd} | 0 | - | 未跑 | - | - |")
                    continue
                lines.append(f"| {bench} | {tag} | {rd} | {r['n']} | {r['ok']} | "
                             f"**{r['acc']*100:.2f}%** | [{r['ci'][0]*100:.1f}, {r['ci'][1]*100:.1f}] | "
                             f"{r['trunc_items']} |")

    lines += ["\n## 逐题配对（同轮，McNemar 精确检验）\n",
              "| 基准 | 轮 | 都对 | 都错 | 仅前者对 | 仅后者对 | Δ(前者−后者) | p |",
              "|---|---|---|---|---|---|---|---|"]
    t0, t1 = tags[0], tags[-1]
    for bench in BENCHES:
        for rd in rounds:
            if (bench, t0, rd) not in results or (bench, t1, rd) not in results:
                continue
            x = results[(bench, t0, rd)]["flags"]
            y = results[(bench, t1, rd)]["flags"]
            common = set(x) & set(y)
            if not common:
                continue
            both = sum(1 for k in common if x[k] and y[k])
            none = sum(1 for k in common if not x[k] and not y[k])
            xo = sum(1 for k in common if x[k] and not y[k])
            yo = sum(1 for k in common if y[k] and not x[k])
            n = len(common)
            delta = (sum(1 for k in common if x[k]) - sum(1 for k in common if y[k])) / n * 100
            lines.append(f"| {bench} | {rd} | {both} | {none} | {xo} | {yo} | "
                         f"{delta:+.2f} | {mcnemar(xo, yo):.3f} |")

    out = "\n".join(lines) + "\n"
    print(out)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write(out)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump({f"{b}_{t}_r{rd}": {"n": v["n"], "ok": v["ok"], "acc": v["acc"],
                                      "ci": v["ci"], "trunc_items": v["trunc_items"]}
                   for (b, t, rd), v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
