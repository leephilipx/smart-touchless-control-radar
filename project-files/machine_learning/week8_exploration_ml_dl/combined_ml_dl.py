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
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
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
from sklearn.metrics import roc_auc_score


classifiers = [
KNeighborsClassifier(3),
SVC(kernel="linear", C=0.025, probability=True),
GaussianProcessClassifier(1.0*RBF(1.0)),
DecisionTreeClassifier(max_depth=5),
RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
MLPClassifier(alpha=1, max_iter=1000),
AdaBoostClassifier(),
GaussianNB(),
QuadraticDiscriminantAnalysis()]
# OneVsOneClassifier(SVC(gamma=0.7, C=1))
classifier_names = ["Nearest Neighbors", "Linear SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    "Naive Bayes", "QDA"]#"RBF SVM"

use_prof_samples = False
use_ml = False
loop_models = False

def machine_learning(classifier, ttsData, classifier_name):
    X_train, X_test, y_train, y_test = ttsData
    # Learn the digits on the train subset
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, f'model_files/{classifier_name}.pkl')    # Save the model as a pickle in a file
    model = joblib.load(f'model_files/{classifier_name}.pkl')        # Load the model from the file

    print(f"{classifier} Model")
    # Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    train_acc = model.score(X_train, y_train)
    print("Classification Accuracy \t:", train_acc, end='\n\n')

    # Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    test_acc = model.score(X_test, y_test)
    print("Classification Accuracy \t:", test_acc, end='\n\n')

    # disp = metrics.plot_confusion_matrix(model, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # Use the loaded model to make predictions
    # print("Predcited values: ", model.predict(X_test))
    # print("Actual values:    ", y_test)
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = model.decision_function(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    print("AUC Score: ", auc)
    return train_acc, test_acc, auc

def deep_learning(ttsData):
    X_train, X_test, y_train, y_test = ttsData
    y_train = tf.one_hot(y_train, len(np.unique(y)))
    y_test = tf.one_hot(y_test, len(np.unique(y)))
    ##### MODEL ######
    model=k.Sequential()
    model.add(tf.keras.Input(shape=X.shape[1:]))
    model.add(k.layers.Conv2D(32,3,3,padding='valid',
        dilation_rate=(1, 1),
        activation="relu"))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(64,activation="relu"))
    model.add(k.layers.Dense(64,activation="relu"))
    model.add(k.layers.Dense(64,activation="relu"))
    model.add(k.layers.Dense(len(np.unique(y)),activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', AUC()])
    ##### MODEL ######
    model.summary()
    history = model.fit(X_train,y_train,epochs=1,batch_size=8)
    model.save('models/deep_learning_model')
    # model = load_model('models/deep_learning_model')
    train_acc = model.evaluate(X_train, y_train, batch_size=128)
    test_acc = model.evaluate(X_test, y_test, batch_size=128)
    auc = np.max(history.history['auc'])
    print("AUC Score: ", auc)

    y_preds = model.predict(X_test)
    # y_labels = []
    # for i in range(len(y_preds)):
    #     y_labels.append(np.argmax(y_preds[i]))
    test_auc = roc_auc_score(y_test, y_preds, multi_class='ovr')
    print(test_auc)
    print(y_preds)
    # print(y_preds)
    return train_acc, test_acc, auc

def choose_dataset():
    """
    Selects either one of the datasets from read_prof_images() or read_our_radar_data(), depending on 
    the boolean variable use_prof_samples and reshape the dataset into a machine learning classifier,
    if applicable.

    Parameters
    ----------
    None
    
    Returns
    -------
    X: Feature parameters of shape (-1, feature_size) for ML or (#samples, width, height, colour) for DL
    y: Output labels of shape (#samples, )
    class_labels: Output class label names list 

    Examples
    --------
    >>> choose_dataset()
    """
    if use_prof_samples:
        print("Dataset chosen: Prof's images")
        X, y, class_labels = read_prof_images()
    else:
        print("Dataset chosen: Radar images")
        X, y, class_labels = read_our_radar_data()
        X = np.abs(X)
        X = np.expand_dims(X, axis=-1)
    if use_ml:  
        X = X.reshape(X.shape[0], -1)
    return X,y, class_labels

def select_pipeline(ttsData, classifiers, classifier_names):
    if use_ml:  
        if loop_models:
            train_acc_dict = {}
            test_acc_dict = {}
            auc_score_dict = {}
            fig, ax = plt.subplots()
            for cf, cf_name in zip(classifiers, classifier_names):
                train_acc, test_acc, auc = machine_learning(cf, ttsData, cf_name)
                train_acc_dict[cf_name] = train_acc
                test_acc_dict[cf_name] = test_acc
                auc_score_dict[cf_name] = auc
            ax.plot(auc_score_dict.keys(), auc_score_dict.values())
            print(auc_score_dict)
            ax.set_title("Comparison of AUC Score of Classifier Models")
            ax.set_xlabel("Classifier Models")
            ax.set_ylabel("AUC Score")
            plt.show()
        else:
            train_acc, test_acc, auc = machine_learning(KNeighborsClassifier(3), ttsData, "Nearest Neighbors") #0.93 
    else:
        train_acc, test_acc, auc = deep_learning(ttsData)
    return train_acc, test_acc, auc


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    X, y, class_labels = choose_dataset()
    ttsData = train_test_split(X, y, test_size=.3, random_state=12)
    train_acc, test_acc, auc = select_pipeline(ttsData, classifiers, classifier_names)
    # np.save('x_test.npy', ttsData[1])
    # np.savez_compressed('dataset.npz', X=X, y=y, class_labels=class_labels)
    # print(X.shape, y.shape, class_labels)   
    # select_pipeline(ttsData, classifiers, classifier_names)