from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train = X_train.reshape(60000, 28, 28, 1)/255.0
x_test = X_test.reshape(10000, 28, 28, 1)/255.0
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.1)

loss, accuracy, precision, recall = model.evaluate(x_test, y_test)
print('Test:')
print('Loss: %s\nAccuracy: %s' % (loss, accuracy))

print("The accucry of model : {0}".format(accuracy))
print("The loss of model : {0}".format(loss))
print("The precision of model : {0}".format(precision))
print("The Recall of model : {0}".format(recall))

first_right = 0
first_wrong = 0
first_pointer = 0
pred_y = model.predict(x_test)

# Find the first classified image and the first miss_classified image 
while(first_right == 0 or first_wrong == 0):
  if (pred_y[first_pointer].argmax() == y_test[first_pointer].argmax()):
    first_right = first_pointer
    first_pointer += 1
  elif(pred_y[first_pointer].argmax() != y_test[first_pointer].argmax()):
    first_wrong = first_pointer
    first_pointer += 1

# 顯示一張預測正確的資料
print("img_index: ", first_right)
print("pred: ", pred_y[first_right].argmax())
print("label", y_test[first_right].argmax())

right_image = x_test[first_right]
right_image = np.array(right_image, dtype='float')
pixels = right_image.reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.show()

# 顯示一張預測錯誤的資料
print("img_index: ", first_wrong)
print("pred: ", pred_y[first_wrong].argmax())
print("label", y_test[first_wrong].argmax())

wrong_image = x_test[first_wrong]
wrong_image = np.array(wrong_image, dtype='float')
pixels = wrong_image.reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.show()

#印出第一層的feature map
from keras.models import Model

feature_maps_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
feature_maps = feature_maps_model.predict(right_image)

f_num = feature_maps.shape[-1]
fig, axs = plt.subplots(8, 4, tight_layout=True, figsize=(10,10))

img_index = 0
for i in range(8):
  for j in range(4):
    img = feature_maps[:, :, :, img_index].reshape(28, 28)
    axs[i, j].imshow(img, cmap='gray')
    img_index += 1
plt.show()

#印出第二層的feature map
feature_maps_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
feature_maps = feature_maps_model.predict(right_image)

f_num = feature_maps.shape[-1]
fig, axs = plt.subplots(8, 8, tight_layout=True, figsize=(10,10))

img_index = 0
for i in range(8):
  for j in range(8):
    img = feature_maps[:, :, :, img_index].reshape(28, 28)
    axs[i, j].imshow(img, cmap='gray')
    img_index += 1
plt.show()