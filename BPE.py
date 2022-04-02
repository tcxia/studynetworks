# Byte-Pair Encoding

"""
    learning process:
    1. 准备足够大的训练语料
    2. 确定期望的subword词表大小
    3. 将单词拆分为字符序列并在末尾添加后缀"</w>", 统计单词频率。比如"low"的频率为5, 那么直接改写成"l o w </w>":5
    4. 统计每一个连续字节对的出现频率, 选择最高频合并成新的subword
    5. 重复第4步, 直到达到第2步设定的subword大小或下一个最高频的字节对出现频率为1
"""

import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbol = word.split()
        for i in range(len(symbol) - 1):
            pairs[symbol[i], symbol[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


vocab = {'l o w </w>':5, 'l o w e r </w>':2, 'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 1000
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

"""
编码以及解码过程
+ 将已经得到的subword词表按照子词长度由大到小排序。
+ 编码时, 对与每个单词, 遍历排序好的字词词表寻找是否有token是当前单词的子字符, 如果是, 则该token是表示单词tokens之一
+ 从最长的token迭代到最短的token, 尝试将每个单词中的子字符串替换为token. 最终, 将迭代所有tokens, 并将所有子字符串替换为tokens.
+ 解码就是将所有tokens拼接在一起
"""