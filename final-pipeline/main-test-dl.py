from master import radar, preprocess, ml
from time import sleep
import numpy as np

if __name__ == "__main__":

    model = ml.DeepLearningModel(model_path='temp_checkpoint.h5')
    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_11_data')
    class_labels.append('background')

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()
    radarSensor.connect_serial(port)
    radarSensor.start_session()
    
    X_frame = np.zeros((1, X.shape[1], X.shape[2]), dtype=X.dtype)
    frame_buffer = 0

    while True:

        try:
            new_frame = np.expand_dims(radarSensor.get_next(), axis=0)
            X_frame = np.concatenate([X_frame[:, 1:, :], new_frame], axis=1)
            frame_buffer += 1
            if frame_buffer == 64:
                frame_buffer = 0
                X_input = preprocess.get_magnitude(X_frame)
                X_input = preprocess.reshape_features(X_input, type='dl')

                 
                X_train, X_test, y_train, y_test = model.train_test_split(X_input, Y, test_size=0.3, random_state=12)
                y_train_one_hot, y_test_one_hot = preprocess.one_hot_dl([y_train, y_test])
                y_probs = model.predict_proba(X_input)
                y_preds = np.where(y_probs > 0.7, 1, 0)
                if np.sum(y_preds) == 0: y_preds = 4
                else: y_preds = np.argmax(y_preds)
                print(y_preds, class_labels[y_preds], np.squeeze(y_probs))

        except KeyboardInterrupt:
            print('>> KeyboardInterrupt caught! Exiting ...')
            break

        except Exception as e:
            try:
                print('\n>> Connection to sensor failed, trying again in 5 seconds ...')
                sleep(5)
                radarSensor.stop_session(verbose=False)
                radarSensor.disconnect_serial(verbose=False)
                port = radarSensor.autoconnect_serial_port()
                radarSensor.connect_serial(port)
                radarSensor.start_session()
            except KeyboardInterrupt:
                print('>> KeyboardInterrupt caught! Exiting ...')
                break