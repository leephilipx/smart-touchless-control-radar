from master import radar, preprocess, ml
from time import sleep
import numpy as np
import argparse
# from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DIP E047: Final Real-time Gesture Prediction')
    parser.add_argument('-rpi', '--rpi', action='store_true', help='Raspberry Pi mode to fix port')
    args = parser.parse_args()

    model_dl = ml.DeepLearningModel(model_path='stft-final.tflite')
    model_ml = ml.MachineLearningModel(model_path='log-reg.pkl')
    X_shape, Y_shape, class_labels = radar.getDatasetInfo(source_dir='2021_10_20_data_new_gestures')

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()
    radarSensor.connect_serial('/dev/ttyUSB0' if args.rpi else port)
    radarSensor.start_session()
    
    X_frame = np.zeros((1, 80, X_shape[2]), dtype=np.complex128)
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

                x_center = preprocess.get_frame_center(X_frame, consensus_buffer)
                # now = datetime.now()

                for offset in range(-consensus_buffer, consensus_buffer+1):
                    X_input = X_frame[:, x_center+offset-32:x_center+offset+32, :]
                    # X_input = preprocess.get_magnitude(X_input)
                    X_input = preprocess.get_batch(X_input, mode='stft')
                    X_input_dl = preprocess.reshape_features(X_input, type='dl')
                    X_input_ml = preprocess.reshape_features(X_input, type='ml')
                    y_probs_dl = model_dl.predict_tflite(X_input_dl)
                    y_probs_ml = model_ml.predict_proba(X_input_ml)
                    y_probs_buffer = y_probs_buffer + y_probs_dl + y_probs_ml
                
                # print(datetime.now() - now)
                frame_buffer = 0
                y_consensus = np.argmax(y_probs_buffer)
                if y_probs_buffer.max() > (preds_threshold * (2*consensus_buffer+1) * 2):
                    print(f'center={x_center};  {y_consensus} {class_labels[y_consensus].ljust(12)}', np.squeeze(y_probs_buffer))
                else:
                    print(f'center={x_center};  - {"n/a".ljust(12)}', np.squeeze(y_probs_buffer))
                y_probs_buffer = np.zeros((len(class_labels), ))

        except KeyboardInterrupt:
            print('>> KeyboardInterrupt caught! Exiting ...')
            break

        except Exception as e:
            try:
                print(f'\n>> Exception caught! {e}')
                print('>> Connection to sensor failed, trying again in 5 seconds ...')
                sleep(5)
                radarSensor.stop_session(verbose=False)
                radarSensor.disconnect_serial(verbose=False)
                port = radarSensor.autoconnect_serial_port()
                radarSensor.connect_serial('/dev/ttyUSB0' if args.rpi else port)
                radarSensor.start_session()
            except KeyboardInterrupt:
                print('>> KeyboardInterrupt caught! Exiting ...')
                break