import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정규화 과정 0~1 사이 값 데이터 스케일링!
x_train, x_test = x_train/255.0, x_test/255.0

#tf.data를 사용하여 데이터셋을 섞고 배치 만들기 , 데이터의 전처리 과정을 나타낸다.
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000) # 이미지 중 10000개를 랜덤하게 추출ㄹ
train_size = int(len(x_train) * 0.7)
train_ds = ds.take(train_size).batch(20) # 70%만 가져와 그 중 20개 씩 랜덤하게 픽한다.
val_ds = ds.skip(train_size).batch(20) # 위와 동일

#모델 생성
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#모델 생성하기
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 모델 훈련하기
hist = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save('mnist_model.h5')

print('모델 저장 완료')

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
