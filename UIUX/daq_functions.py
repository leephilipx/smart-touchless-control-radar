import os, numpy as np
from re import findall
from acconeer.exptool import utils, clients, configs


class AcconeerSensorDataCollection:

    '''
    Creates a sensor object and its helper functions:
    • Connect to the Acconeer Radar sensor via serial/socket,
    • Load sensor configs from a json file,
    • Obtain data via IQ mode and save it to a .npy file.
    '''

    def __init__(self, method, Nframes, config_path='sensor_configs.json'):

        self.connection_state = False
        self.session_state = False
        self.method = method
        self.Nframes = Nframes
        self.config_path = config_path


    def autodetect_serial_port(self):
        return utils.autodetect_serial_port()


    def list_serial_ports(self):

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


    def connect_sensor(self, port):

        if not self.connection_state:

            self.client = clients.UARTClient(port)
            self.client.squeeze = False

            try:
                info = self.client.connect()
                self.rss_version = info.get('version_str', None)
                print(f'>> Connection success! RSS v{self.rss_version}')
                self.connection_state = True
                
            except Exception:
                print('>> Could not connect to sensor, please check the physical connection / free up the port.')
            
    
    def disconnect_sensor(self, verbose=True):

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

        if self.connection_state:
            return True
        else:
            print('>> Connection not established with client. Please run AcconeerSensorDIP.connect_sensor()')
            return False
        

    def get_data(self):

        if not self.check_connection_state(): return
        return self.client.get_next()


    def start_session(self):
        
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

        if not self.check_connection_state(): return
        self.sensor_config = configs.IQServiceConfig()
        self.sensor_config.range_interval = [0.3, 0.6]
        self.sensor_config.update_rate = 30

        if not self.session_state:
            
            try:
                self.session_info = self.client.setup_session(self.sensor_config)
                self.client.start_session()
                self.session_state = True
                print('>> Session started!')
            except Exception:
                print('>> Session failed to start!')


    def stop_session(self, verbose=True):
        
        if self.session_state:
            try:
                self.client.stop_session()
                self.session_state = False
                print('>> Active session stopped!')
            except Exception:
                pass

        elif verbose:
            print('>> Stop session failed, no active session found!')


    def __del__(self):

        self.stop_session(verbose=False)
        self.disconnect_sensor(verbose=False)



if __name__ == '__main__':

    radar = AcconeerSensorDataCollection(method='serial', Nframes=128, config_path='')
    port = radar.autodetect_serial_port()
    radar.connect_sensor(port)
    radar.start_session()