#!/usr/bin/env python3
"""对比两侧 PPL 哨兵结果，给出逐 chunk 方向一致性。

    python ppl_compare.py --ref ppl_bf16_code.json --cand ppl_w8a8_code.json

单看总 Δ 会被加性抖动骗：+0.6% 在散文域就是噪声，但在代码域配 10/10 全变差
就是真信号。所以这个脚本的重点是末端那两行——多少 chunk 变差、方向是否单调。
"""
import argparse
import json


def load(path):
    d = json.load(open(path))
    per = {}
    for c in d.get("per_chunk", []):
        per[c["i"]] = {"tokens": c["tokens"], "nll": c["nll"]}
    return d, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="对照侧 JSON（如 BF16）")
    ap.add_argument("--cand", required=True, help="被测侧 JSON（如 W8A8）")
    a = ap.parse_args()

    ref, rp = load(a.ref)
    cand, cp = load(a.cand)
    common = sorted(set(rp) & set(cp))
    if not common:
        raise SystemExit("两侧没有任何共同 chunk，检查 --text / --chunks / --chunk-chars 是否一致")

    print(f"| chunk | tokens | {ref['label']} NLL/tok | {cand['label']} NLL/tok | Δ | 变差 |")
    print("|---|---|---|---|---|---|")
    worse = 0
    for i in common:
        r = rp[i]["nll"] / rp[i]["tokens"]
        c = cp[i]["nll"] / cp[i]["tokens"]
        d = c - r
        worse += d > 0
        print(f"| {i} | {rp[i]['tokens']} | {r:.4f} | {c:.4f} | {d:+.4f} | {'是' if d > 0 else '否'} |")

    r_all = ref["nll_per_token"]
    c_all = cand["nll_per_token"]
    print()
    print(f"- {ref['label']}: NLL/token **{r_all:.4f}**  (PPL {ref['ppl']:.4f})")
    print(f"- {cand['label']}: NLL/token **{c_all:.4f}**  (PPL {cand['ppl']:.4f})")
    print(f"- Δ = {c_all - r_all:+.4f} / token，相对 **{(c_all / r_all - 1) * 100:+.1f}%**")
    print(f"- 变差 chunk：**{worse}/{len(common)}**")


if __name__ == "__main__":
    main()
