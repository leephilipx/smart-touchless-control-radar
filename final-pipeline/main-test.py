from master import radar, preprocess, plotutils, ml

if __name__ == "__main__":

    radarSensor = radar.AcconeerSensorLive(config_path='sensor_configs_final.json')
    port = radarSensor.autoconnect_serial_port()