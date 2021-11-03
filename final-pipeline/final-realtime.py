from master import radar, preprocess, ml
from time import sleep
import numpy as np
import argparse
from pyautogui import press

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DIP E047: Final Real-time Gesture Prediction')
    parser.add_argument('-rpi', '--rpi', action='store_true', help='Raspberry Pi mode to fix port')
    parser.add_argument('-kb', '--kb', action='store_true', help='Keyboard presses')
    args = parser.parse_args()

    model = ml.DeepLearningModel(model_path='stft-final.h5')
    X_shape, Y_shape, class_labels = radar.getDatasetInfo(source_dir='2021_10_20_data_new_gestures')

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()
    radarSensor.connect_serial('/dev/ttyUSB0' if args.rpi else port)
    radarSensor.start_session()
    
    X_frame = np.zeros((1, 80, X_shape[2]), dtype=complex)
    preds_threshold = 0.6
    frame_buffer = 0
    consensus_buffer = 2
    y_probs_buffer = np.zeros((len(class_labels), ))
    
    np.set_printoptions(suppress=True, precision=3)

    while True:

        try:
            X_frame[:, :-1, :] = X_frame[:, 1:, :]
            X_frame[:, -1, :] = np.expand_dims(radarSensor.get_next(), axis=0)
            frame_buffer += 1
            
            if frame_buffer == 80:
                
                if args.kb: press('.')
                x_center = preprocess.get_frame_center(X_frame, consensus_buffer)

                for offset in range(-consensus_buffer, consensus_buffer+1):
                    X_input = X_frame[:, x_center+offset-32:x_center+offset+32, :]
                    # X_input = preprocess.get_magnitude(X_input)
                    X_input = preprocess.get_batch(X_input, mode='stft')
                    X_input = preprocess.reshape_features(X_input, type='dl')
                    y_probs = model.predict_proba(X_input)
                    y_probs_buffer = y_probs_buffer + y_probs

                frame_buffer = 0
                y_consensus = np.argmax(y_probs_buffer)
                if y_probs_buffer.max() > (preds_threshold * (2*consensus_buffer+1)):
                    print(f'center={x_center};  {y_consensus} {class_labels[y_consensus].ljust(12)}', np.squeeze(y_probs_buffer))
                    if args.kb: press(str(y_consensus))
                else:
                    print(f'center={x_center};  - {"n/a".ljust(12)}', np.squeeze(y_probs_buffer))
                    if args.kb: press('/')
                y_probs_buffer = np.zeros((len(class_labels), ))

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