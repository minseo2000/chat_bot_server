from keras.datasets import mnist
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

model = load_model('mnist_model.h5')
model.summary()
model.evaluate(x_test, y_test, verbose=2)

plt.imshow(x_test[20])
plt.show()

predict = model.predict_classes(x_test[[20]])
print(predict)