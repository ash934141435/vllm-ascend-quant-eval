#!/usr/bin/env python3
"""全量跑官方协议，支持断点续跑。

    python run_eval.py --base-url http://127.0.0.1:8000/v1/chat/completions \\
        --model qwen3.8-27b --tag bf16 --round 1 --datasets gpqa,ifbench,mv

每侧每轮产出一个 jsonl：results/full/<tag>_<bench>_r<round>.jsonl

用滚动窗口维持固定数量的在飞请求，而不是一次性 submit 全部 task——
后者在 1000+ 题规模下会把 future 队列堆死。
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import REPEATS, SEED_BASE, build_messages, call_api, log  # noqa: E402


def find_arrow(data_dir, dataset_dir, filename):
    """在 modelscope / HuggingFace 缓存里定位 arrow 文件（哈希子目录因机器而异）。"""
    root = os.path.expanduser(data_dir)
    hits = sorted(glob.glob(os.path.join(root, dataset_dir, "*", "*", "*", filename)))
    if not hits:
        raise SystemExit(f"找不到 {filename}，请用 --data-dir 指向数据集缓存根目录\n"
                         f"（首次可用 `evalscope eval --datasets ...` 触发下载）: {root}")
    return hits[-1]


def dataset_items(which, a):
    if which == "gpqa":
        ds = Dataset.from_file(find_arrow(a.data_dir, "AI-ModelScope___gpqa_diamond", "gpqa_diamond-train.arrow"))
        return "GPQA", [(str(i), ds[i]) for i in range(len(ds))]
    if which == "ifbench":
        ds = Dataset.from_file(find_arrow(a.data_dir, "allenai___if_bench_test", "if_bench_test-train.arrow"))
        return "IFBench", [(str(ds[i]["key"]), ds[i]) for i in range(len(ds))]
    if which == "mv":
        ds = Dataset.from_file(find_arrow(a.data_dir, "evalscope___math_vision", "math_vision-test.arrow"))
        by_id = {str(ds[i]["id"]): ds[i] for i in range(len(ds))}
        if a.mv_mode == "full":
            ids = list(by_id)
        else:
            # 分层抽样 id 列表由外部提供，两侧必须用同一份
            with open(a.mv_ids) as f:
                ids = json.load(f)["ids"]
        return "MV", [(sid, by_id[sid]) for sid in ids]
    raise SystemExit(f"未知数据集 {which}")


def run(bench, items, out_path, seed_base, a):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((str(d["key"]), d["variant"]))
                except Exception:
                    pass

    tasks = [(k, rec, v) for k, rec in items for v in range(REPEATS) if (str(k), v) not in done]
    total = len(items) * REPEATS
    log(f"  {bench}: {len(items)} items x {REPEATS} = {total}, 已完成 {len(done)}, "
        f"待跑 {len(tasks)}, 并发 {a.workers}")
    if not tasks:
        return

    finished = len(done)
    t0 = last_progress = time.time()
    it = iter(tasks)
    inflight = {}

    with open(out_path, "a") as fo, ThreadPoolExecutor(max_workers=a.workers) as pool:
        def submit_next():
            try:
                key, rec, v = next(it)
            except StopIteration:
                return False
            inflight[pool.submit(call_api, a.base_url, a.model, build_messages(bench, rec), seed_base + v)] = (key, v)
            return True

        for _ in range(a.workers):
            if not submit_next():
                break

        while inflight:
            done_set, _ = wait(list(inflight), timeout=120, return_when=FIRST_COMPLETED)
            if not done_set:
                log(f"  [{bench}] 警告：{time.time() - last_progress:.0f}s 无完成，在飞 {len(inflight)}")
                continue
            for fut in done_set:
                key, v = inflight.pop(fut)
                try:
                    result, raw_body = fut.result()
                except Exception as e:
                    log(f"  {bench} key={key} v={v} 异常 {e}")
                    submit_next()
                    continue
                if result is None:
                    log(f"  {bench} key={key} v={v} 失败")
                    submit_next()
                    continue
                ch = result["choices"][0]
                msg = ch["message"]
                rec_out = {
                    "key": str(key), "variant": v, "seed": seed_base + v,
                    "content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning") or "",
                    "stop_reason": ch.get("stop_reason"),
                    "finish_reason": ch.get("finish_reason"),
                    "usage": result.get("usage"),
                    "raw_body": raw_body,   # 逐行记录实际下发参数，事后可审计
                }
                fo.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                fo.flush()
                finished += 1
                last_progress = time.time()
                el = time.time() - t0
                rate = (finished - len(done)) / el * 60 if el > 0 else 0
                eta = (total - finished) / rate if rate > 0 else 0
                log(f"  {bench} [{finished}/{total}] key={key} v={v} "
                    f"stop={ch.get('stop_reason')} tok={result.get('usage', {}).get('completion_tokens')} "
                    f"| {rate:.1f}/min ETA {eta:.0f}min")
                submit_next()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="http://host:port/v1/chat/completions")
    ap.add_argument("--model", required=True, help="服务端 --served-model-name")
    ap.add_argument("--tag", required=True, help="本侧标签，如 w8a8 / bf16")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    ap.add_argument("--datasets", default="gpqa,ifbench,mv")
    ap.add_argument("--mv-mode", default="rep399", choices=["full", "rep399"])
    ap.add_argument("--mv-ids", default="results/mv_sample_ids.json")
    ap.add_argument("--data-dir", default="~/.cache/modelscope/hub/datasets")
    ap.add_argument("--out-dir", default="results/full")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    seed_base = SEED_BASE[a.round]
    log(f"=== tag={a.tag} round={a.round} seed_base={seed_base} base={a.base_url} model={a.model} ===")

    for which in [x.strip() for x in a.datasets.split(",") if x.strip()]:
        bench, items = dataset_items(which, a)
        out = os.path.join(a.out_dir, f"{a.tag}_{which}_r{a.round}.jsonl")
        log(f"\n=== {bench} ({which}) -> {out} ===")
        run(bench, items, out, seed_base, a)

    log("\n=== done ===")


if __name__ == "__main__":
    main()
