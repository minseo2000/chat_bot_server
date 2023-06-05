import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, LSTM, SimpleRNN

# time_step만큼 시퀀스 데이터 분리
def split_sequence(sequence, step):
    x, y = list(), list()

    for i in range(len(sequence)):
        end_idx = i + step
        if end_idx > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

#sin함수 학습 데이터
x = [i for i in np.arange(-10, 10, 0.1)]
train_y = [np.sin(x) for i in x]

#하이퍼 파라미터
n_timesteps = 15
n_features = 1

#시퀀스 나누기
train_x, train_y, = split_sequence(train_y, step=n_timesteps)

#RNN입력 벡터 사이즈 조정
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], n_features)

#RNN 모델 정의
model = Sequential()
model.add(SimpleRNN(units=10, return_sequences=False, input_shape=(n_timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#모델 학습
np.random.seed(0)
from tensorflow.python.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    mode='auto'
)
history = model.fit(train_x, train_y, epochs=1000, callbacks=[early_stopping])

