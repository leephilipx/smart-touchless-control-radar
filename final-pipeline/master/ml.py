import numpy as np
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import os
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

def ml():
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amax` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.15.0

    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

        .. versionadded:: 1.17.0

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `amax` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``amax(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    """
def ml(X_train, X_test, y_train, y_test, classifier, classifier_name):
    # Learn the digits on the train subset
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, f'trained_model/{classifier_name}.pkl')    # Save the model as a pickle in a file
    model = joblib.load(f'trained_model/{classifier_name}.pkl')        # Load the model from the file

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

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    X, y, class_labels = read_prof_images()
    X = X.reshape(-1, 256*256*3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=12) 
    # print(X.shape, y.shape, class_labels)
    ml(X_train, X_test, y_train, y_test, KNeighborsClassifier(3), "Nearest Neighbors")


    
    # SVC(kernel="linear", C=0.025),
    # OneVsOneClassifier(SVC(gamma=0.7, C=1)),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()])
    
    # classifier_names = ["Nearest Neighbors"], "Linear SVM", "RBF SVM", #"Gaussian Process",
        #  "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #  "Naive Bayes", "QDA"]