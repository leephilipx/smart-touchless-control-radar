import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import os
import tensorflow as tf
import tensorflow.keras as k
import sklearn as sk
import pickle
from read_images import read_prof_images, read_our_radar_data
from sklearn.model_selection import train_test_split
from read_images import read_prof_images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn import metrics

def ml(classifier, ttsData, classifier_name):
    X_train, X_test, y_train, y_test = ttsData
    # Learn the digits on the train subset
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, f'model_files/{classifier_name}.pkl')    # Save the model as a pickle in a file
    model = joblib.load(f'model_files/{classifier_name}.pkl')        # Load the model from the file

    print(f"{classifier} Model")
    # Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Classification Accuracy \t:", model.score(X_train, y_train), end='\n\n')

    # Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Classification Accuracy \t:", model.score(X_test, y_test), end='\n\n')

    disp = metrics.plot_confusion_matrix(model, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    # plt.show()

    # Use the loaded model to make predictions
    print("Predcited values: ", model.predict(X_test))
    print("Actual values:    ", y_test)

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
    history = model.fit(X_train,y_train,epochs=50,batch_size=8)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    X, y, class_labels = read_prof_images()
    X = X.reshape(-1, 256*256*3)
    ttsData = train_test_split(X, y, test_size=.3, random_state=12) 
    # print(X.shape, y.shape, class_labels)
    # ml(KNeighborsClassifier(3), ttsData, "Nearest Neighbors") #0.93
    # ml(SVC(kernel="linear", C=0.025), ttsData, "Nearest Neighbors") #1.0
    # ml(OneVsOneClassifier(SVC(gamma=0.7, C=1), ttsData, "Nearest Neighbors") #0.
    # ml(GaussianProcessClassifier(1.0 * RBF(1.0)), ttsData, "Nearest Neighbors") #
    # ml(DecisionTreeClassifier(max_depth=5), ttsData, "Nearest Neighbors") #0.8777
    # ml(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), ttsData, "Nearest Neighbors") #0.9555
    ml(MLPClassifier(alpha=1, max_iter=1000), ttsData, "MLPClassifier") #0.
    # ml(AdaBoostClassifier(), ttsData, "Nearest Neighbors")
    # ml(GaussianNB(), ttsData, "Nearest Neighbors")
    # ml(QuadraticDiscriminantAnalysis(), ttsData, "Nearest Neighbors")
      
    # classifier_names = ["Nearest Neighbors"], "Linear SVM", "RBF SVM", #"Gaussian Process",
        #  "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #  "Naive Bayes", "QDA"]