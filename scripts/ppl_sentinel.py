#!/usr/bin/env python3
"""PPL 哨兵：用 vLLM 的 prompt_logprobs 算 NLL，不需要本地加载权重。

    python ppl_sentinel.py --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b \\
        --label bf16 --text wikitext2_test.txt --chunks 10 --out ppl_bf16_code.json

两侧必须用同一份 --text、同样的 --chunk-chars / --chunks，分数才可比。
PPL 绝对值依赖语料和分块方式，A/B 的 Δ 才是信号。

为什么需要这个：基准准确率是离散的、分辨率被题数限制死（198 题时一格 = 0.5 分），
而量化造成的损伤往往小于这个格。NLL 是连续量，能在准确率还看不出差异时就先报警。
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request


def post(url, payload, api_key, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer %s" % (api_key or "EMPTY")},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chunk_text(text, chunk_chars, n_chunks):
    chunks = []
    step = max(1, int(chunk_chars))
    for i in range(0, len(text), step):
        piece = text[i:i + step].strip()
        if len(piece) >= step // 2:     # 跳过尾部过短的残块
            chunks.append(piece)
        if len(chunks) >= n_chunks:
            break
    return chunks


def score_chunk(base_url, model, chunk, api_key, timeout):
    payload = {
        "model": model, "prompt": chunk,
        "max_tokens": 1, "temperature": 0, "top_p": 1,
        "prompt_logprobs": 0, "ignore_eos": True,
    }
    data = post(base_url.rstrip("/") + "/completions", payload, api_key, timeout)
    entries = data["choices"][0].get("prompt_logprobs") or []
    nll, n_tok = 0.0, 0
    for e in entries:
        if not e:                       # 首个 token 没有条件概率
            continue
        vals = [v for v in e.values() if isinstance(v, dict) and "logprob" in v]
        if len(vals) == 1:              # prompt_logprobs=0 只回实际 token
            lp = vals[0]["logprob"]
        elif vals:                      # 罕见：回多个候选，取最可能的（轻微失真）
            lp = max(v["logprob"] for v in vals)
        else:
            continue
        nll += -float(lp)
        n_tok += 1
    return nll, n_tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--chunks", type=int, default=30)
    p.add_argument("--chunk-chars", type=int, default=4000)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    text = open(a.text, encoding="utf-8", errors="ignore").read()
    chunks = chunk_text(text, a.chunk_chars, a.chunks)
    if not chunks:
        sys.exit("语料太短：%s" % a.text)

    per_chunk, total_nll, total_tok = [], 0.0, 0
    for i, c in enumerate(chunks):
        try:
            nll, n = score_chunk(a.base_url, a.model, c, a.api_key, a.timeout)
        except urllib.error.HTTPError as e:
            body = e.read()[:400].decode("utf-8", "ignore")
            sys.exit("HTTP %s on chunk %d（端点不支持 prompt_logprobs？）: %s" % (e.code, i, body))
        if n == 0:
            sys.stderr.write("chunk %d 无有效 logprob，跳过\n" % i)
            continue
        per_chunk.append({"i": i, "tokens": n, "nll": nll, "ppl": math.exp(nll / n)})
        total_nll += nll
        total_tok += n
        sys.stderr.write("chunk %d/%d tokens=%d nll=%.4f ppl=%.4f\n"
                         % (i + 1, len(chunks), n, nll / n, math.exp(nll / n)))

    if not total_tok:
        sys.exit("没有任何 chunk 拿到 logprob")
    mean_nll = total_nll / total_tok
    result = {
        "label": a.label, "model": a.model,
        "chunks": len(per_chunk), "tokens": total_tok,
        "nll_per_token": mean_nll, "ppl": math.exp(mean_nll),
        "per_chunk": per_chunk,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_chunk"},
                     indent=2, ensure_ascii=False))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=2, ensure_ascii=False)
        sys.stderr.write("写入 %s\n" % a.out)


if __name__ == "__main__":
    main()
