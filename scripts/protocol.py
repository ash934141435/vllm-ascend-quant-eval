#!/usr/bin/env python3
"""官方采样协议 + prompt 构造 + 输出预算。

协议取自 checkpoint 自带的 generation_config.json：
    temperature=1.0 / top_p=0.95 / top_k=20

输出预算 = max_context − prompt，即不留固定 max_tokens。这样 262k 预算下
没有任何一题被截断，避免"截断判错"污染量化差异。

两侧（W8A8 / BF16）必须用逐字节相同的 prompt，否则 A/B 不成立，
所以所有 prompt 构造都只在这里实现一次。
"""
import base64
import hashlib
import io
import os
import random
import threading
import time

import requests

MAX_CONTEXT = int(os.environ.get("EVAL_MAX_CONTEXT", 262144))
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", 3600))
REPEATS = 3                      # 每样本独立采样次数，majority@3
SEED_BASE = {1: 42, 2: 142}      # 第 1 / 第 2 轮的 seed 起点

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def max_tokens_for(messages, img_tokens=0):
    """输出预算 = 全文上下文长度 − prompt 估算长度。

    prompt token 数按 chars/2 高估（真实 Qwen 分词器约 2–4 chars/token），
    因此预算永远落在服务端上限之内，对真实 prompt 仍接近 260k。
    """
    parts = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            parts.append(c)
        else:
            parts.extend(p.get("text", "") for p in c if p.get("type") == "text")
    n_prompt = len("".join(parts)) // 2 + 128   # 余量：template 与 special token
    return max(1024, MAX_CONTEXT - n_prompt - img_tokens)


def gpqa_prompt(rec):
    """GPQA 选项顺序按题目哈希固定打乱，两侧看到同一份选项排列。"""
    question_raw = rec["Question"]
    choices = [rec["Incorrect Answer 1"].strip(), rec["Incorrect Answer 2"].strip(),
               rec["Incorrect Answer 3"].strip(), rec["Correct Answer"].strip()]
    seed = int.from_bytes(hashlib.sha256(question_raw.strip().encode()).digest()[:8], "big")
    random.Random(seed).shuffle(choices)
    choices_text = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
    letters = ",".join(chr(65 + i) for i in range(len(choices)))
    return (f"Answer the following multiple choice question. The last line of your response "
            f"should be of the following format: 'ANSWER: [LETTER]' (without quotes) where "
            f"[LETTER] is one of {letters}. Think step by step before answering."
            f"\n\n{question_raw}\n\n{choices_text}")


def gpqa_target(rec):
    """与 gpqa_prompt 用同一哈希打乱，回推正确选项字母。"""
    q = rec["Question"]
    choices = [rec["Incorrect Answer 1"].strip(), rec["Incorrect Answer 2"].strip(),
               rec["Incorrect Answer 3"].strip(), rec["Correct Answer"].strip()]
    seed = int.from_bytes(hashlib.sha256(q.strip().encode()).digest()[:8], "big")
    random.Random(seed).shuffle(choices)
    return chr(65 + choices.index(rec["Correct Answer"].strip()))


def ifbench_prompt(rec):
    return rec["prompt"]


def _mv_text(rec):
    question = rec["question"]
    options = rec["options"]
    if len(options) > 0:
        letters = ",".join(options)
        choices_text = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(options))
        return (f"Answer the following multiple choice question. The last line of your response "
                f"should be of the following format: 'ANSWER: [LETTER]' (without quotes) where "
                f"[LETTER] is one of {letters}. Think step by step before answering."
                f"\n\n{question}\n\n{choices_text}")
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}} without units."


def mv_messages(rec):
    content = [{"type": "text", "text": _mv_text(rec)}]
    img = rec.get("decoded_image")
    img_bytes = None
    if isinstance(img, dict):
        img_bytes = img.get("bytes")
    elif img is not None and hasattr(img, "save"):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
    if img_bytes:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [{"role": "user", "content": content}]


def build_messages(bench, rec):
    if bench == "MV":
        return mv_messages(rec)
    if bench == "GPQA":
        return [{"role": "user", "content": gpqa_prompt(rec)}]
    return [{"role": "user", "content": ifbench_prompt(rec)}]


def call_api(base_url, model, messages, seed):
    """返回 (响应 JSON, 脱敏后的请求体)；重试 4 次仍失败则返回 (None, body)。"""
    has_img = any(isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"])
                  for m in messages)
    body = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens_for(messages, img_tokens=8192 if has_img else 0),
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed": seed,
    }
    for attempt in range(4):
        try:
            resp = requests.post(base_url, json=body, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json(), {k: v for k, v in body.items() if k != "messages"}
        except Exception:
            if attempt < 3:
                time.sleep(2 ** attempt)
    return None, {k: v for k, v in body.items() if k != "messages"}
