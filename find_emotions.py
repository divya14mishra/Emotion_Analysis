import sys
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


df = pd.read_csv("fer2013.csv")
# print(df.info())
# print(df["Usage"].value_counts())
x_train, y_train, x_test, y_test = [], [], [], []
c = 0
for index, row in df.iterrows():
    c += 1
    val = row['pixels'].split(" ")
    # print("index", index)
    # print("row", row)
    # print("val", val)
    # if c == 20:
    #     break
    try:
        if 'Training' in row['Usage']:
            x_train.append(np.array(val, 'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            x_test.append(np.array(val, 'float32'))
            y_test.append(row['emotion'])
    except Exception as e:
        print(f"error occured at index :{index} and row:{row}")

# print("x train :", x_train[0:4])
# print("x test :", x_test[0:4])
# print("y test :", y_test[0:4])
# print("y train :", y_train[0:4])

num_features = 64
num_labels = 7
batch_size = 64
epochs = 35
width, height = 48, 48

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

y_train = np_utils.to_categorical(y_train, num_classes=num_labels)
y_test = np_utils.to_categorical(y_test, num_classes=num_labels)

#cannot produce
#normalizing data between oand 1
x_train -= np.mean(x_train, axis=0)
x_test /= np.std(x_train, axis=0)

x_test -= np.mean(x_test, axis=0)
x_test /= np.std(x_test, axis=0)

x_train = x_train.reshape(x_train.shape[0], width, height, 1)

x_test = x_test.reshape(x_test.shape[0], width, height, 1)

# print(f"shape:{X_train.shape}")
##designing the cnn
#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1:])))
model.add(Conv2D(64, kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

#Training the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)


#Saving the  model to  use it later on
import pickle
with open('C:\\Users\\Hp\\PycharmProjects\\emotion_analysis\\emotion-analysis-model.pkl', 'wb') as f:
  pickle.dump(model, f)