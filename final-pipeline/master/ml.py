import os, numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

class sklearn_clf():

    def __init__(self):
        self.model = None
        self.clf_name = 'knn'

    def load_model(self):
        self.model = load(f'models/{self.clf_name}.pkl')
        return self.model
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=12)
        self.model = KNeighborsClassifier(n_neighbors=len(np.unique(y)))
        self.model.fit(X_train, y_train)
        dump(self.model, f'models/{self.clf_name}.pkl')
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        y_prob = self.model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        return train_accuracy, test_accuracy, auc

    def evaluate(self, features):
        if self.model is None: return None
        y_preds = self.model.predict(features)
        return y_preds

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    dataset = np.load('models/dataset.npz')
    X, y, class_labels = dataset['X'], dataset['y'], dataset['class_labels']
    X = np.abs(X)
    # X = np.expand_dims(X, axis=-1) for dl
    X = X.reshape(X.shape[0], -1)
    model = sklearn_clf()

    #Train code
    train_accuracy, test_accuracy, auc = model.train(X,y)

    #Evaluate code
    # model.load_model()
    # x_test = np.load('models/x_test.npy').reshape(24, -1)
    # model.evaluate(x_test)
    # print(x_test.shape)
    # print(model.evaluate(x_test))