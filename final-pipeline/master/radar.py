import os, numpy as np
from re import findall
from json import loads as json_loads
from pickle import dump as pickle_dump, load as pickle_load
from acconeer.exptool import utils, clients, configs


class AcconeerSensorLive:

    '''
    Creates a sensor object and its helper functions:
    (1) Load sensor configs from a json file
    (2) Connect to the Acconeer Radar sensor via serial port
    (3) Obtain data via IQ mode in real-time
    '''

    def __init__(self, config_path='sensor_configs_final.json', port=None):
        '''
        Initialize the AcconeerSensorLive object.
        '''
        self.connection_state = False
        self.session_state = False
        self.__sensor_config = self.get_config(config_path)
        print(self.__sensor_config)
        self.port = port
        if port is not None:
            self.autoconnect_serial_port()

    def get_config(self, config_path):
        '''
        Loads configs from an external json file.
        '''
        '''
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
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
            with open(config_path, 'r') as f:
                ext_dict = json_loads(f.read())
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
        return config

    def get_config_dict(self):
        '''
        Returns config dict object as a Python dictionary.
        '''
        return {
            'mode': self.__sensor_config.mode,
            'sensor': self.__sensor_config.sensor,
            'range_interval': self.__sensor_config.range_interval,
            'profile': self.__sensor_config.profile,
            'update_rate': self.__sensor_config.update_rate,
            'sampling_mode': self.__sensor_config.sampling_mode,
            'repetition_mode': self.__sensor_config.repetition_mode,
            'downsampling_factor': self.__sensor_config.downsampling_factor,
            'hw_accelerated_average_samples': self.__sensor_config.hw_accelerated_average_samples,
            'gain': self.__sensor_config.gain,
            'maximize_signal_attenuation': self.__sensor_config.maximize_signal_attenuation,
            'noise_level_normalization': self.__sensor_config.noise_level_normalization,
            'depth_lowpass_cutoff_ratio': self.__sensor_config.depth_lowpass_cutoff_ratio,
            'tx_disable': self.__sensor_config.tx_disable,
            'power_save_mode': self.__sensor_config.power_save_mode,
            'asynchronous_measurement': self.__sensor_config.asynchronous_measurement,
        }

    def list_serial_ports(self):
        '''
        Lists the available serial ports for different platforms.
        '''
        try:
            opsys = os.uname()
            in_wsl = 'microsoft' in opsys.release.lower() and 'linux' in opsys.sysname.lower()
        except Exception:
            in_wsl = False
        port_tag_tuples = utils.get_tagged_serial_ports()
        if not in_wsl and os.name == 'posix':
            ports = []
            for i, (port, tag) in enumerate(port_tag_tuples):
                tag_string = ''
                if tag:
                    select = i
                    tag_string = ' ({})'.format(tag)
                ports.append(port + tag_string)
        else:
            ports = [port for port, *_ in port_tag_tuples]
        try:
            if in_wsl:
                print('>> WSL detected. Limiting serial ports')
                ports_reduced = []
                for p in ports:
                    if int(findall(r'\d+', p)[0]) < 20:
                        ports_reduced.append(p)
                ports = ports_reduced
        except Exception:
            pass
        return ports

    def autoconnect_serial_port(self):
        '''
        Connects to the Acconeer sensor given the serial port.
        '''
        if self.port is not None:
            if self.connect_serial(self.port):
                return self.port
        else:
            ports = self.list_serial_ports()
            for port in ports[::-1]:
                try:
                    if self.connect_serial(port):
                        return port
                except Exception:
                    pass
        print(f'>> No sensor found on any serial port!')
    
    def connect_serial(self, port):
        '''
        Connects to the Acconeer sensor given the serial port.
        '''
        if not self.connection_state:
            self.client = clients.UARTClient(port)
            self.client.squeeze = False
            try:
                info = self.client.connect()
                self.rss_version = info.get('version_str', None)
                print(f'>> Connection success on {port}! RSS v{self.rss_version}')
                self.connection_state = True
                return True   
            except Exception:
                print(f'>> Could not connect to sensor on {port}, please check the physical connection / free up the port.')
            
    def disconnect_serial(self, verbose=True):
        '''
        Disconnects and releases the serial port connected to the Acconeer sensor.
        '''
        if self.connection_state:
            try:
                if not self.session_state:
                    self.stop_session(verbose=False)
                self.client.disconnect()
                print('>> Client disconnected!')
                self.connection_state = False
            except Exception:
                pass
        elif verbose:
            print('>> Disconnect failed, no active connection found!')

    def check_connection_state(self):
        '''
        Returns the connection state, while printing an error message if appropriate.
        '''
        if self.connection_state:
            return True
        else:
            print('>> Connection not established with client. Please run connect_serial()')
            return False

    def start_session(self):
        '''
        Starts an active session with the Acconeer sensor.
        '''
        if not self.check_connection_state(): return
        if not self.session_state:
            try:
                self.session_info = self.client.setup_session(self.__sensor_config)
                self.client.start_session()
                self.session_state = True
                print('>> Session started!')
                self.depths = utils.get_range_depths(self.__sensor_config, self.session_info)
                self.Ndepths = self.depths.size
                return True
            except Exception:
                print('>> Session failed to start!')
                return

    def stop_session(self, verbose=True):
        '''
        Stops any active sessions with the Acconeer sensor.
        '''
        if self.session_state:
            try:
                self.session_state = False
                self.client.stop_session()
                print('>> Active session stopped!')
            except Exception:
                pass
        elif verbose:
            print('>> Stop session failed, no active session found!')

    def get_next(self):
        '''
        Returns a frame of IQ data obtained from the Acconeer sensor.
        '''      
        return self.client.get_next()[1]

    def __del__(self):
        '''
        Clean up session and serial port.
        '''
        self.stop_session(verbose=False)
        self.disconnect_serial(verbose=False)
        print('>> AcconeerSensorLive object cleaned up!')


def getTrainData(source_dir):
    '''
    Scans all files in the directory provided under project-files\\radar_data\\ and saves dataset info.
    Return numpy arrays of (X, Y, class_labels), with the first dimension of X and Y being the sample_index.
    '''
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'project-files', 'radar_data', source_dir)
    dirs = [dir for dir in os.listdir(root_dir) if (dir.startswith('gesture_') and os.path.isdir(os.path.join(root_dir, dir)))]
    assert len(dirs) > 0, "No matching folders with prefix 'gesture_'!"
    X = []
    Y = []
    class_labels = [label[8:] for label in dirs]
    for ind, dir in enumerate(dirs):
        radarData = [np.load(os.path.join(root_dir, dir, data))['sample'] for data in os.listdir(os.path.join(root_dir, dir)) if data.endswith('.npz')]
        X = X + radarData
        Y = Y + [ind] * len(radarData)
    with open(os.path.join(root_dir, 'dataset_info.pickle'), 'wb') as f:
        pickle_dump((np.array(X).shape, np.array(Y).shape, class_labels), f)
    return np.array(X), np.array(Y), class_labels


def getDatasetInfo(source_dir):
    '''
    Return dataset info (X_shape, Y_shape, class_labels) from previous training instance.
    '''
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'project-files', 'radar_data', source_dir)
    with open(os.path.join(root_dir, 'dataset_info.pickle'), 'rb') as f:
        result = pickle_load(f)
    return result


def cache(mode, X=None, Y=None, class_labels=None):
    if mode == 'save':
        np.savez_compressed('dataset_cache.npz', X=X, Y=Y, class_labels=class_labels)
    elif mode == 'load':
        npfile = np.load('dataset_cache.npz')
        return npfile['X'], npfile['Y'], npfile['class_labels']


if __name__ == '__main__':
    # radarSensor = AcconeerSensorLive(config_path='sensor_configs_final.json', port=None)
    # port = radarSensor.autoconnect_serial_port()
    # radarSensor.connect_serial(port)
    # radarSensor.start_session()
    # radarSensor.get_next()
    # X, Y, class_labels = getTrainData(source_dir='2021_10_13_data')
    print(getDatasetInfo(source_dir='2021_10_13_data'))
