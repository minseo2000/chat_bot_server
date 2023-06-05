from konlpy.tag import Komoran
import numpy as np

s = "오늘 날씨는 구름이 많아요."

# 명사만 추출
komoran = Komoran()
noun = komoran.nouns(s)
print(noun)

dics = {}
for word in noun:
    if word not in dics.keys():
        dics[word] = len(dics)
print(dics)

nb_classes = len(dics)
targets = list(dics.values())
one_hot_targets = np.eye(nb_classes)
print(one_hot_targets)
one_hot_targets = np.eye(nb_classes)[targets]
print(one_hot_targets)

