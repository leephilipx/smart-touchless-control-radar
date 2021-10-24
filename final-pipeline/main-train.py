from master import radar, preprocess, plotutils, ml
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
    
if __name__ == "__main__":

    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_20_data_new_gestures')#'2021_10_13_testing_data_new')
    print(X.shape, Y.shape, class_labels)
    # X, Y, class_labels = radar.cache('load')
    # X_features = preprocess.get_magnitude(X)
    X_features = preprocess.get_batch(X, mode='stft')

    ##### ML ######

    # X_input = preprocess.reshape_features(X_features, type='ml')
    # print(X_features.shape, X_input.shape)

    # model = ml.MachineLearningModel(sklearn_model=KNeighborsClassifier(n_neighbors=4))
    # X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=0)
    # print('(train, test):', (len(y_train), len(y_test)))
    # train_acc, train_auc = model.train_batch(X_train, y_train, model_path='knn.pkl')
    # test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test)
    # print('Train-Test Acc =', train_acc, test_acc)
    # print('Train-Test AUC =', train_auc, test_auc)
    
    ##### DL ######

    X_input = preprocess.reshape_features(X_features, type='dl')
    print(X_features.shape, X_input.shape)

    # np.savez_compressed('X_input.npz', X_input=X_input)
    # X_input = np.load('X_input.npz')['X_input']

    model = ml.DeepLearningModel()
    X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=0)
    y_train_one_hot, y_test_one_hot = preprocess.one_hot_dl([y_train, y_test])
    callbacks = model.instantiate_callbacks()
    model.initialise_model(X_train, y_train_one_hot)
    train_acc, train_auc = model.train_batch(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=20, batch_size=8, callbacks=callbacks)
    test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test_one_hot)
    print('Train-Test Acc =', round(train_acc,5), round(test_acc,5))
    print('Train-Test AUC =', round(train_auc,5), round(test_auc,5))
    print(y_preds.shape)