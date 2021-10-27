from master import radar, preprocess, ml
from time import sleep
import numpy as np


if __name__ == "__main__":

    model = ml.DeepLearningModel(model_path='stft-run2.h5')
    X_shape, Y_shape, class_labels = radar.getDatasetInfo(source_dir='2021_10_20_data_new_gestures')

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()
    # port = '/dev/ttyUSB0'
    radarSensor.connect_serial(port)
    radarSensor.start_session()
    
    X_frame = np.zeros((1, X_shape[1], X_shape[2]), dtype=np.complex)
    frame_buffer = 0
    y_preds_buffer = np.zeros((len(class_labels), ))
    consensus_buffer = 2
    confidence_threshold = 0.7
    np.set_printoptions(suppress=True, precision=3)

    # background = []
    # for i in range(64):
    #     background.append(radarSensor.get_next().ravel())
    # background = np.expand_dims(np.array(background), axis=0)

    while True:

        try:
            new_frame = np.expand_dims(radarSensor.get_next(), axis=0)
            X_frame = np.concatenate([X_frame[:, 1:, :], new_frame], axis=1)
            frame_buffer += 1
            
            if frame_buffer >= (64-consensus_buffer):
                # X_input = preprocess.get_magnitude(X_frame)
                X_input = preprocess.get_batch(X_frame, mode='stft')
                X_input = preprocess.reshape_features(X_input, type='dl')
                y_probs = model.predict_proba(X_input)
                # y_preds = np.where(y_probs > confidence_threshold, 1, 0)
                # if np.sum(y_preds) == 0: y_preds = 0
                # else: y_preds = np.argmax(y_preds)
                y_preds = np.argmax(y_probs)
                # print(y_preds, class_labels[y_preds], np.squeeze(y_probs))
                y_preds_buffer[y_preds] = y_preds_buffer[y_preds] + 1
            if frame_buffer == (64+consensus_buffer):
                frame_buffer = consensus_buffer
                y_consensus = np.argmax(y_preds_buffer)
                print(y_consensus, class_labels[y_consensus], np.squeeze(y_preds_buffer))
                y_preds_buffer = np.zeros((len(class_labels), ))

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