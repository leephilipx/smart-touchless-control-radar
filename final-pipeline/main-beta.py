from master import radar, preprocess, plotutils, ml

if __name__ == "__main__":
    X, Y, class_labels = radar.getTrainData()
    print(X.shape, Y.shape, class_labels)
    X_mag = preprocess.get_magnitude(X)
    X_input = preprocess.reshape_features(X_mag, 'ml')
    print(X_mag.shape, X_input.shape)