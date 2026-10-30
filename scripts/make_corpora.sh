#!/usr/bin/env bash
# 生成 PPL 哨兵用的两份语料。两侧必须用同一份文件，否则 A/B 不成立。
#
# 散文：TinyStories v2 GPT-4 valid，取前 2 000 000 字符
# 代码：Python 标准库源码，取前 2 000 000 字符
#
# 注意：标准库文件顺序依赖宿主机 Python 版本，跨机器复现时文件内容会不同。
# 但这对 A/B 没有影响——只要两侧看的是同一份 corpus_*.txt，Δ 就有意义；
# 跨实验比较 Δ 才需要固定语料，此时请把生成好的文件归档即可。
set -euo pipefail
cd "$(dirname "$0")/.."

curl -sL -o /tmp/tinystories.txt \
  https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
python3 -c "open('corpus_prose.txt','w').write(open('/tmp/tinystories.txt',errors='ignore').read(2000000))"

find /usr/lib/python3* -name '*.py' | sort | head -400 | xargs cat | head -c 2000000 > corpus_code.txt

wc -c corpus_prose.txt corpus_code.txt
