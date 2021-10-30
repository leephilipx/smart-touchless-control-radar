from master import radar, preprocess, ml
from time import sleep
import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DIP E047: Final Real-time Gesture Prediction')
    parser.add_argument('-rpi', '--rpi', action='store_true', help='Raspberry Pi mode to fix port')
    args = parser.parse_args()

    model = ml.DeepLearningModel(model_path='stft-run-3-new-data.h5')
    X_shape, Y_shape, class_labels = radar.getDatasetInfo(source_dir='2021_10_27_data_new_gestures')

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()
    radarSensor.connect_serial('/dev/ttyUSB0' if args.rpi else port)
    radarSensor.start_session()
    
    X_frame = np.zeros((1, X_shape[1], X_shape[2]), dtype=np.complex)
    frame_buffer = 0
    y_preds_buffer = np.zeros((len(class_labels), ))
    consensus_buffer = 2
    confidence_threshold = 0.7
    np.set_printoptions(suppress=True, precision=3)

    while True:

        try:
            X_frame[:, :-1, :] = X_frame[:, 1:, :]
            X_frame[:, -1, :] = np.expand_dims(radarSensor.get_next(), axis=0)
            frame_buffer += 1
            
            if frame_buffer >= (80-consensus_buffer):
                X_input = preprocess.get_magnitude(X_frame)
                # X_input = preprocess.get_batch(X_frame, mode='stft')
                X_input = preprocess.reshape_features(X_input, type='dl')
                y_probs = model.predict_proba(X_input)
                y_preds = np.argmax(y_probs)
                y_preds_buffer[y_preds] = y_preds_buffer[y_preds] + 1

            if frame_buffer == (80+consensus_buffer):
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
                radarSensor.connect_serial('/dev/ttyUSB0' if args.rpi else port)
                radarSensor.start_session()
            except KeyboardInterrupt:
                print('>> KeyboardInterrupt caught! Exiting ...')
                break