from master import radar, preprocess, ml
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
    
if __name__ == "__main__":

    deep = True
    # Get the data and preprocess it
    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_20_data_new_gestures')
    # radar.cache('save', X, Y, class_labels)
    # X, Y, class_labels = radar.cache('load')
    print(X.shape, Y.shape, class_labels)
    X_features = preprocess.get_batch(X, mode='stft')
    # X_features = preprocess.get_magnitude(X)

    if deep: # DL model: train code

        X_input = preprocess.reshape_features(X_features, 'dl')
        print(X_features.shape, X_input.shape)

        model = ml.DeepLearningModel()
        X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=12)
        y_train_one_hot, y_test_one_hot = preprocess.one_hot_dl([y_train, y_test])
        print(y_train_one_hot.shape, y_test_one_hot.shape)
        callbacks = model.instantiate_callbacks()
        model.initialise_model(X_train, y_train_one_hot)
        train_acc, train_auc = model.train_batch(X_train, y_train_one_hot, validation_data = (X_test, y_test_one_hot), epochs=20, batch_size=8, callbacks=callbacks)
        model.save_model('stft-final.h5')
        test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test_one_hot)
        print('Train-Test Acc =', round(train_acc, 5), round(test_acc, 5))
        print('Train-Test AUC =', round(train_auc, 5), round(test_auc, 5))
        print("Y_Preds", y_preds.shape)
        # model.fake_tensorboard()


    else: # ML model: train code

        X_input = preprocess.reshape_features(X_features, 'ml')
        print(X_features.shape, X_input.shape)
        # from sklearn.neighbors import KNeighborsClassifier
        # model = MachineLearningModel(sklearn_model=KNeighborsClassifier(n_neighbors=len(np.unique(Y))))
        from sklearn.linear_model import LogisticRegression
        model = ml.MachineLearningModel(sklearn_model=LogisticRegression())
        X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=12)
        train_acc, train_auc = model.train_batch(X_train, y_train, model_path='log-reg.pkl')
        test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test)
        print('Train-Test Acc =', round(train_acc,5), round(test_acc,5))
        print('Train-Test AUC =', round(train_auc,5), round(test_auc,5))
    
        # # ML model: sample load model + predict code
        # model = MachineLearningModel(model_path='knn.pkl')
        # y_preds = model.predict(X_input)
        # print(y_preds.shape)

        '''
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression

        [train-acc, test-acc, train-auc, test-auc]
        [0.97143, 0.96674, 0.99946, 0.99922] KNeighborsClassifier(n_neighbors=len(np.unique(Y)) 
        [0.99333, 0.96231, 0.99993, 0.99843] RandomForestClassifier(max_depth=5)
        [0,83714, 0.80710, 0.90494, 0.88612] GaussianNB()
        [1.00000, 1.00000, 1.00000, 1.00000] LogisticRegression()
        '''