#텍스트 유사도 n-gram에 관련된 코드이다.
from konlpy.tag import Komoran

#문장 속 거리n만큼의 토큰 추출
def word_ngram(token, n):
    tokens = tuple(token)
    ngrams = [tokens[x: x+n] for x in range(0, len(token))]
    return tuple(ngrams)

# 유사도 검색
def similarity(doc1, doc2):
    cnt = 0
    for token in doc1:
        if token in doc2:
            cnt+=1
    return cnt/len(doc1)

s1 = "6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다"
s2 = "6월에 뉴턴은 선생님의 제안으로 대학에 입학했다"
s3 = "나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다."

komoran = Komoran()
doc1 = komoran.nouns(s1)
doc2 = komoran.nouns(s2)
doc3 = komoran.nouns(s3)

doc1 = word_ngram(doc1,2)
doc2 = word_ngram(doc2, 2)
doc3 = word_ngram(doc3, 2)

print(similarity(doc1,doc2))
print(similarity(doc1, doc3))
