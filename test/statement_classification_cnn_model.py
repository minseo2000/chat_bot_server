import pandas as pd
import numpy as np
import tensorflow as tf
import keras.preprocessing as preprocessing
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

# 데이터 불러오기
train_file = './ChatbotData.csv'
train_data = pd.read_csv(train_file, delimiter=',')
features = train_data['Q'].tolist()
labels = train_data['label'].tolist()
# features 데이터의 변환 과정을 지켜보자
# 데이터를 불러온 후
# ['12시 땡!', '1지망 학교 떨어졌어', '3박4일 놀러가고 싶다', '3박4일 정도 놀러가고 싶다', 'PPL 심하네']

# 단어 인덱스 시퀀스 벡터
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
# 덱트스 전처리 후
# [['12시', '땡'], ['1지망', '학교', '떨어졌어'], ['3박4일', '놀러가고', '싶다'], ['3박4일', '정도', '놀러가고', '싶다'], ['ppl', '심하네']]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
# [[4646, 4647], [4648, 343, 448], [2580, 803, 11], [2580, 804, 803, 11], [4649, 2581]] 시퀀스로 변환 후
word_index = tokenizer.word_index
# 토크나이저로 인덱싱 한 후
#{'너무': 1, '좋아하는': 2, '거': 3, '싶어': 4, '같아': 5, '안': 6, '나': 7, '좀': 8, '사람': 9, '내가': 10, '싶다': 11, '어떻게': 12, '썸': 13... }

MAX_SEQ_LEN = 15 # 단어 시퀀스 벡터 크기 정의
padded_seqs = preprocessing.sequence.data_utils.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
# pad_sequences 처리 후
# [[4646 4647    0    0    0    0    0    0    0    0    0    0    0    0
#      0]
#  [4648  343  448    0    0    0    0    0    0    0    0    0    0    0
#      0]
#  [2580  803   11    0    0    0    0    0    0    0    0    0    0    0
#      0]
#  [2580  804  803   11    0    0    0    0    0    0    0    0    0    0
#      0]
#  [4649 2581    0    0    0    0    0    0    0    0    0    0    0    0
#      0]]

# 학습용, 검증용, 테스트용 데이터 셋 생성하기
# 학습셋 : 검증셋 : 테스트셋 = 7 : 2 : 1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size+val_size).take(test_size).batch(20)

# 하이퍼 파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1 # 전체 단어 수

#CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu,)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1,pool2,pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden)
predictions = Dense(3, activation=tf.nn.softmax)(logits)

#  모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

loss, accuracy = model.evaluate(test_ds, verbose=1)
print(accuracy)
print(loss)

model.save('cnn_model.h5')
