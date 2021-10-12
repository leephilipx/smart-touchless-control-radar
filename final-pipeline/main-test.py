from master import radar, preprocess, ml
from time import sleep
import numpy as np

if __name__ == "__main__":

    model = ml.MachineLearningModel(model_path='knn.pkl')
    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_11_data')

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()
    radarSensor.connect_serial(port)
    radarSensor.start_session()
    
    X_frame = np.zeros((1, X.shape[1], X.shape[2]), dtype=X.dtype)
    frame_buffer = 0

    while True:

        try:
            new_frame = radarSensor.get_next().reshape(1, 1, -1)
            X_frame = np.concatenate([X_frame[:, :-1, :], new_frame], axis=1)
            frame_buffer = 1
            if frame_buffer == 8:
                frame_buffer = 0
                X_input = preprocess.get_magnitude(X_frame)
                X_input = preprocess.reshape_features(X_input, type='ml')
                y_preds = model.predict(X_input)
                print(y_preds, class_labels[y_preds[0]])

        except KeyboardInterrupt:
            print('KeyboardInterrupt caught! Exiting ...')
            break

        except Exception as e:
            try:
                del radarSensor
                print('\n>> Connection to sensor failed, trying again in 5 seconds ...')
                sleep(5)
                radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
                port = radarSensor.autoconnect_serial_port()
                radarSensor.connect_serial(port)
                radarSensor.start_session()
            except KeyboardInterrupt:
                print('KeyboardInterrupt caught! Exiting ...')
                break