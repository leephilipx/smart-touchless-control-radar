import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

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

classifiers_names = ["Nearest Neighbors"]#, "Linear SVM", "RBF SVM", #"Gaussian Process",
        #  "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #  "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3)]
    # SVC(kernel="linear", C=0.025),
    # OneVsOneClassifier(SVC(gamma=0.7, C=1)),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()]

X, y, class_labels = read_prof_images()
X = X.reshape(-1, 256*256*3)
print(X.shape, y.shape, class_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=12)

for cf, cf_name in zip(classifiers, classifiers_names):
    # Learn the digits on the train subset
    cf.fit(X_train, y_train)

    joblib.dump(cf, f'trained_model/{cf_name}.pkl')               # Save the model as a pickle in a file
    model = joblib.load(f'trained_model/{cf_name}.pkl')           # Load the model from the file

    print(f"{cf} Model")
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