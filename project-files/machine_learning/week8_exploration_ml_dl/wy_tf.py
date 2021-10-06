import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from read_images import read_prof_images, read_our_radar_data
from sklearn.model_selection import train_test_split

use_prof=True
if use_prof:
    X, y, class_labels = read_prof_images()
else:
    X, y, class_labels = read_our_radar_data()
    X = np.abs(X)
    X = np.expand_dims(X, axis=-1)

ttsData = train_test_split(X, y, test_size=0.33, random_state=42)
def dl(ttsData):
    X_train, X_test, y_train, y_test = ttsData
    y_train = tf.one_hot(y_train, y.shape[0])
    y_test = tf.one_hot(y_test, y.shape[0])

    model=k.Sequential()
    model.add(tf.keras.Input(shape=X.shape[1:]))
    model.add(k.layers.Conv2D(32,3,3,padding='valid',
        dilation_rate=(1, 1),
        activation="relu"))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(64,activation="relu"))
    model.add(k.layers.Dense(64,activation="relu"))
    model.add(k.layers.Dense(64,activation="relu"))
    model.add(k.layers.Dense(y.shape[0],activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history=model.fit(X_train,y_train,epochs=50,batch_size=8)
    y_preds = model.predict(X_test)
dl(ttsData)