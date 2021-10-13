from master import radar, preprocess, ml
from time import sleep
import numpy as np

if __name__ == "__main__":

    model = ml.MachineLearningModel(model_path='knn.pkl')
    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_12_iot_data')

    X_input = preprocess.get_magnitude(X)
    X_input = preprocess.reshape_features(X_input, type='ml')
    y_preds = model.predict(X_input)
    
    print(y_preds, Y)
    print([class_labels[i] for i in y_preds])