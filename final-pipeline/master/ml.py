import os, numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

class MachineLearningModel():

    def __init__(self, clf_name='knn'):
        self.model = None
        self.clf_name = clf_name
        self.root_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(self.root_dir, f'models/{self.clf_name}.pkl')

    def load_saved_model(self):
        self.model = load(self.model_dir)
        return self.model
    
    def train(self, X, y, save_model=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=12)
        self.model = KNeighborsClassifier(n_neighbors=len(np.unique(y)))
        self.model.fit(X_train, y_train)
        if save_model: dump(self.model, self.model_dir)
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        y_prob = self.model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        return train_acc, test_acc, auc

    def evaluate(self, features):
        try:
            y_preds = self.model.predict(features)
            return y_preds
        except Exception:
            print('Model is not defined!')


class DeepLearningModel():

    def __init__(self):
        pass




if __name__ == "__main__":

    # Get the data and preprocess it
    import radar, preprocess
    X, Y, class_labels = radar.getTrainData()
    print(X.shape, Y.shape, class_labels)
    X_mag = preprocess.get_magnitude(X)
    X_input = preprocess.reshape_features(X_mag, 'ml')
    print(X_mag.shape, X_input.shape)

    model = MachineLearningModel(clf_name='knn')

    # Train code
    train_acc, test_acc, auc = model.train(X_input, Y)
    print(train_acc, test_acc, auc)

    # Evaluate code
    # model.load_saved_model()
    # x_test = np.load('models/x_test.npy').reshape(24, -1)
    # model.evaluate(x_test)
    # print(x_test.shape)
    # print(model.evaluate(x_test))