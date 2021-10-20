from sklearn.decomposition import PCA
import radar, preprocess, ml

X, Y, class_labels = radar.cache('load')
print(X.shape, Y.shape, class_labels)
X_mag = preprocess.get_batch(X, mode='stft')
# X_input = preprocess.reshape_features(X_mag, 'ml')
# print(X_input.shape)
# X_input = PCA().fit_transform(X_input)
print(X_input.shape)
X_input = preprocess.reshape_features(X_input, 'dl')
print(X_input.shape)

model = ml.DeepLearningModel()
X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=12)
y_train_one_hot, y_test_one_hot = preprocess.one_hot_dl([y_train, y_test])
print(y_train_one_hot.shape, y_test_one_hot.shape)
callbacks = model.instantiate_callbacks()
model.initialise_model(X_train, y_train_one_hot)
train_acc, train_auc = model.train_batch(X_train, y_train_one_hot, validation_data = (X_test, y_test_one_hot), epochs=20, batch_size=8, callbacks=callbacks)
test_acc, test_auc, y_preds = model.evaluate_batch(X_test, y_test_one_hot)
print('Train-Test Acc =', round(train_acc, 5), round(test_acc, 5))
print('Train-Test AUC =', round(train_auc, 5), round(test_auc, 5))
print("Y_Preds", y_preds.shape)