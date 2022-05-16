from unittest import result
import jieba
import wordcloud

import imageio
mk = imageio.imread("chinamap.png")

w = wordcloud.WordCloud(width=1000, height=700, background_color="white", mask=mk, scale=15)

def generateChinese(filepath):
    with open(filepath, encoding="utf-8") as fr:
        txt = fr.readlines()
    result = []
    for t in txt:
        t = t.strip()
        txtLine = jieba.lcut(t)
        result.extend(txtLine)
    strRet = " ".join(result)
    w.generate(strRet)
    w.to_file("output.png")

def generateEnglish(filepath):
    with open(filepath, encoding="utf-8") as f:
        txt = f.read()
    string = txt.strip().split(" ")

    w.generate(string)
    w.to_file("output.png")


if __name__ == "__main__":
    generateChinese("test.txt")
