# word2vec 분산표현 임베딩 모델 만들기!
# nsmc 데이터 사용 -> ratings.txt
from gensim.models import Word2Vec
from konlpy.tag import Komoran
import time


#load nsmc data
def load_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

print('nsmc 데이터 불러오기')
document = load_data('./ratings.txt')
print(document[:1])
print('불러오기 완료')

print('형태소 분석하기 -> 명사만 추출하기')
komoran = Komoran()
docs = [komoran.nouns(sentences[1]) for sentences in document]
print(docs)
print('형태소 분석 완료')

print('분산 벡터 생성하는 중..')
model = Word2Vec(sentences=docs, vector_size=200, window=4, hs=1, min_count=2, sg=1)
print('분삭 벡터 생성 완료')

print('분산 벡터 모델 저장')
model.save('nsvc.model')
print('분산 벡터 모델 저장완료!')


