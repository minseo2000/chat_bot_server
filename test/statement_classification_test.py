import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.models import Model, load_model
import keras.preprocessing as preprocessing

train_file = './ChatbotData.csv'
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
label = data['label'].tolist()

# 단어 인덱스 시퀀스 벡터 처리
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.data_utils.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, label))
ds = ds.shuffle(len(features))
test_ds = ds.take(2000).batch(20)

model = load_model('cnn_model.h5')
model.summary()
model.evaluate(test_ds, verbose=2)

print('단어시퀀스', corpus[10212])
print('단어 인덱스 시퀀스', padded_seqs[10212])
print('문장 분류 정답', label[10212])

predict_class = model.predict(padded_seqs[[10212]])
print(predict_class)
print(tf.math.argmax(predict_class, axis=1))