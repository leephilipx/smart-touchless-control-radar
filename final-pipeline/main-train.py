from master import radar, preprocess, plotutils, ml
from sklearn.neighbors import KNeighborsClassifier
    
if __name__ == "__main__":

    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_11_data')
    print(X.shape, Y.shape, class_labels)
    
    X_mag = preprocess.get_magnitude(X)
    X_input = preprocess.reshape_features(X_mag, type='ml')
    print(X_mag.shape, X_input.shape)

    model = ml.MachineLearningModel(sklearn_model=KNeighborsClassifier(n_neighbors=4))
    X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=0)
    print('(train, test):', (len(y_train), len(y_test)))
    train_acc, train_auc = model.train_batch(X_train, y_train, model_path='knn.pkl')
    test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test)
    print('Train-Test Acc =', train_acc, test_acc)
    print('Train-Test AUC =', train_auc, test_auc)
