import os, numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.saving import model_config
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.vgg16 import VGG16

class MachineLearningModel():

    def __init__(self, sklearn_model=None, **kwargs):
        '''
        Initiates an sklearn ML model, or loads one from model_path if provided.
        '''
        self.model = sklearn_model
        self.root_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(self.root_dir, 'models')
        if 'model_path' in kwargs:
            self.load_saved_model(kwargs['model_path'])
    
    def train_test_split(self, X, y, test_size, random_state):
        '''
        Performs train-test-split and returns a tuple of (X_train, X_test, y_train, y_test).
        '''
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_batch(self, X_train, y_train, model_path=None):
        self.model.fit(X_train, y_train)
        if model_path is not None:
            assert model_path.endswith('.pkl'), 'Unknown model_path'
            dump(self.model, os.path.join(self.model_dir, model_path))
            print(f'Model saved to {os.path.normpath(os.path.join(self.model_dir, model_path))}')
        return self.metrics(X_train, y_train)

    def metrics(self, X, y):
        y_probs = self.model.predict_proba(X)
        acc = self.model.score(X, y)
        auc = roc_auc_score(y, y_probs, multi_class='ovr')
        return acc, auc
    
    def evaluate_batch(self, X_test, y_test):
        y_preds = self.model.predict(X_test)
        test_acc, test_auc = self.metrics(X_test, y_test)
        return test_acc, test_auc, y_preds

    def load_saved_model(self, model_path):
        assert model_path.endswith('.pkl'), 'Unknown model_path'
        self.model = load(os.path.join(self.model_dir, model_path))
        return self.model

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)


class DeepLearningModel():

    def __init__(self, **kwargs):
        '''
        Initiates an TensorFlow ML model, or loads one from model_path if provided.
        '''
        self.model = None
        self.root_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(self.root_dir, 'models')
        if 'model_path' in kwargs:
            self.load_saved_model(kwargs['model_path'])
    
    def train_test_split(self, X, y, test_size, random_state):
        '''
        Performs train-test-split and returns a tuple of (X_train, X_test, y_train, y_test).
        '''
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def initialise_model(self, X, y):
        features = Input(shape=X.shape[1:], name='input')
        x = Conv2D(32, 3, 3, padding='valid', activation='relu', name='conv2d-1')(features)
        x = MaxPooling2D(2, 2, name='maxpool-1')(x)
        x = Conv2D(64, 3, 3, padding='valid', activation='relu', name='conv2d-2')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(64, activation='relu', name='dense-1')(x)
        x = Dense(32, activation='relu', name='dense-2')(x)
        preds = Dense(y.shape[1], activation='softmax', name='output')(x)
        self.model = Model(features, preds, name='deep_learning_classifier')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', AUC()])
        self.model.summary()

    def train_batch(self, X_train, y_train, **kwargs):
        self.history = self.model.fit(X_train, y_train, verbose=2, **kwargs)
        train_acc, train_auc = self.metrics(X_train, y_train, evaluate=False)
        return train_acc, train_auc

    def metrics(self, X, y, evaluate=True):
        if evaluate:
            _, acc, auc = self.model.evaluate(X, y, batch_size=8)
        else:
            acc = self.history.history['accuracy'][-1]
            auc = self.history.history['auc'][-1]
        return acc, auc

    def evaluate_batch(self, X_test, y_test):
        y_preds = self.model.predict(X_test)
        test_acc, test_auc = self.metrics(X_test, y_test, evaluate=True)
        return test_acc, test_auc, y_preds

    def load_saved_model(self, model_path):
        assert model_path.endswith('.h5'), 'Unknown model_path'
        self.model = load_model(os.path.join(self.model_dir, model_path))
        return self.model
    
    def predict(self, y_preds):
        return np.argmax(y_preds, axis=0)

    def predict_proba(self, features):
        return self.model.predict(features)
    
    def instantiate_callbacks(self, temp_path='temp_checkpoint.h5'):
        checkpoint = ModelCheckpoint(os.path.join(self.model_dir, temp_path), monitor='val_loss',
                                     verbose=0, save_best_only=True, mode='min', save_weights_only=False)
        earlystopping = EarlyStopping(monitor='val_loss', patience=4)
        return [checkpoint, earlystopping]

    def fake_tensorboard(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
if __name__ == "__main__":
    # Get the data and preprocess it
    import radar, preprocess
    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_11_data')
    print(X.shape, Y.shape, class_labels)
    X_mag = preprocess.get_batch(X, mode='stft')
    # X_mag = preprocess.get_batch(X, mode='mfcc')

    #### TAKE IMAGES ####
    # from read_images import read_stft_plots
    # X, Y, class_labels = read_stft_plots(folder='images_stft_old')
    # print(X.shape, Y.shape, class_labels)
    # X_mag = X

    X_input = preprocess.reshape_features(X_mag, 'dl')
    print(X_mag.shape, X_input.shape)

    # # ML model: sample train code
    # from sklearn.neighbors import KNeighborsClassifier
    # model = MachineLearningModel(sklearn_model=KNeighborsClassifier(n_neighbors=len(np.unique(Y))))
    # X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=12)
    # print('(train, test):', (len(y_train), len(y_test)))
    # train_acc, train_auc = model.train_batch(X_train, y_train, model_path='knn.pkl')
    # test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test)
    # print('Train-Test Acc =', train_acc, test_acc)
    # print('Train-Test AUC =', train_auc, test_auc)

    # # ML model: sample load model + predict code
    # model = MachineLearningModel(model_path='knn.pkl')
    # y_preds = model.predict(X_input)
    # print(y_preds.shape)

    model = DeepLearningModel()
    X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=12)
    y_train_one_hot, y_test_one_hot = preprocess.one_hot_dl([y_train, y_test])
    print(y_train_one_hot.shape, y_test_one_hot.shape)
    callbacks = model.instantiate_callbacks()
    model.initialise_model(X_train, y_train_one_hot)
    train_acc, train_auc = model.train_batch(X_train, y_train_one_hot, validation_data = (X_test, y_test_one_hot), epochs=20, batch_size=8, callbacks=callbacks)
    test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test_one_hot)
    print('Train-Test Acc =', train_acc, test_acc)
    print('Train-Test AUC =', train_auc, test_auc)
    print("Y_Preds", y_preds.shape)
    model.fake_tensorboard()