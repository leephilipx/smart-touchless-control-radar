import radar, preprocess, ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

'''
KNN: 0.82, 
SVC: 

'''
classifiers = [
KNeighborsClassifier(6),
SVC(kernel="linear", C=0.025, probability=True),
GaussianProcessClassifier(1.0*RBF(1.0)),
DecisionTreeClassifier(max_depth=5),
RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
AdaBoostClassifier(),
GaussianNB(),
QuadraticDiscriminantAnalysis()]

classifier_names = ["Nearest Neighbors", "Linear SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes", "QDA"]

plot_type = ['Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC'][0]

if __name__ == "__main__":

    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_20_data_new_gestures')
    print(X.shape, Y.shape, class_labels)
    X, Y, class_labels = radar.cache('load')
    # X_mag = preprocess.get_magnitude(X)
    X_mag = preprocess.get_batch(X, mode='stft')
    X_input = preprocess.reshape_features(X_mag, 'ml')
    print(X_mag.shape, X_input.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_input, Y, test_size=0.3, random_state=12)
    train_acc_dict = {}
    test_acc_dict = {}
    train_auc_dict = {}
    test_auc_dict = {}
    for cf, cf_name in zip(classifiers, classifier_names):
        model = ml.MachineLearningModel(sklearn_model=cf)
        train_acc, train_auc = model.train_batch(X_train, y_train)
        test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test)
        train_acc_dict[cf_name] = train_acc
        test_acc_dict[cf_name] = test_acc
        train_auc_dict[cf_name] = train_auc
        test_auc_dict[cf_name] = test_auc
        print(train_acc)
    plt.plot()
    if plot_type == 'Train Accuracy':
        plt.bar(train_acc_dict.keys(), train_acc_dict.values())
    elif plot_type == 'Test Accuracy':
        plt.bar(test_acc_dict.keys(), test_acc_dict.values())
    elif plot_type == 'Train AUC':
        plt.bar(train_auc_dict.keys(), train_auc_dict.values())
    elif plot_type == 'Test AUC':
        plt.bar(test_auc_dict.keys(), test_auc_dict.values())
    plt.title(f"Comparison of {plot_type} Score of Classifier Models")
    plt.plotxlabel("Classifier Models")
    plt.ylabel(f"{plot_type} Score")
    plt.show()
    plt.savefig(f'{plot_type}')
