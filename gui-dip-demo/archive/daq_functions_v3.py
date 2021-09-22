import os, numpy as np
from json import loads
from acconeer.exptool import configs


class AcconeerSensorDataCollection:

    '''
    Creates a sensor object and its helper functions:
    • Load sensor configs from a json file
    • Save IQ data to a .npy file
    '''

    def __init__(self, config_path='sensor_configs.json'):

        self.get_config(config_path)


    def get_config(self, config_path):

        '''
        Loads configs from an external json file

        IQServiceConfig
        mode .............................. IQ
        sensor ............................ [1]
        range_interval .................... [0.2, 0.8]
        profile ........................... PROFILE_2
        update_rate ....................... 30.0
        sampling_mode ..................... A
        repetition_mode ................... HOST_DRIVEN
        downsampling_factor ............... 1
        hw_accelerated_average_samples .... 10
        gain .............................. 0.5
        maximize_signal_attenuation ....... False
        noise_level_normalization ......... True
        depth_lowpass_cutoff_ratio ........ None
        tx_disable ........................ False
        power_save_mode ................... ACTIVE
        asynchronous_measurement .......... True
        '''

        try:
            with open(config_path, 'r') as f:
                ext_dict = loads(f.read())
            config = configs.IQServiceConfig()
            config.hw_accelerated_average_samples = ext_dict['HWAAS']
            config.gain = ext_dict['gain']
            config.power_save_mode = ext_dict['power_save_mode']
            config.profile = ext_dict['profile']
            config.range_interval =[ext_dict['range_min'], ext_dict['range_max']]
            config.update_rate = ext_dict['update_rate']
            print(f'>> Successfully loaded {config_path}')

        except Exception:
            print('>> File not found / invalid json file! Using default configs ...')
            config = configs.IQServiceConfig()

        self.sensor_config = config

        return config


    def get_config_dict(self):

        return {
            'mode': self.sensor_config.mode,
            'sensor': self.sensor_config.sensor,
            'range_interval': self.sensor_config.range_interval,
            'profile': self.sensor_config.profile,
            'update_rate': self.sensor_config.update_rate,
            'sampling_mode': self.sensor_config.sampling_mode,
            'repetition_mode': self.sensor_config.repetition_mode,
            'downsampling_factor': self.sensor_config.downsampling_factor,
            'hw_accelerated_average_samples': self.sensor_config.hw_accelerated_average_samples,
            'gain': self.sensor_config.gain,
            'maximize_signal_attenuation': self.sensor_config.maximize_signal_attenuation,
            'noise_level_normalization': self.sensor_config.noise_level_normalization,
            'depth_lowpass_cutoff_ratio': self.sensor_config.depth_lowpass_cutoff_ratio,
            'tx_disable': self.sensor_config.tx_disable,
            'power_save_mode': self.sensor_config.power_save_mode,
            'asynchronous_measurement': self.sensor_config.asynchronous_measurement,
        }


    def __del__(self):

        pass


    def save_data(self, npy_filename):

        if not os.path.isdir('recordings'):
            os.makedirs('recordings')
        npy_filename = (os.path.join('recordings', npy_filename)).lower()
        np.save(npy_filename, self.data)
        print(f'>> Data saved to {npy_filename}')


if __name__ == '__main__':

    radar = AcconeerSensorDataCollection(config_path='sensor_configs.json')
    print(radar.get_config_dict())