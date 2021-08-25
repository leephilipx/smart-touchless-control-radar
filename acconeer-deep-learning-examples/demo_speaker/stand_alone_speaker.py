import argparse
import os
import sys
import threading
import time
import traceback
from queue import Queue

import numpy as np
from squeezebox_controller import SqueezeBoxController

from acconeer.exptool import imock, utils
from acconeer.exptool.clients import SocketClient


imock.add_mock_packages(imock.GRAPHICS_LIBS)

# Path to acconeer-python-exploration folder relative to this file
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../../acconeer-python-exploration")
)
# Path to ml processing files relative to this file
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../../acconeer-python-exploration/gui/ml")
)

import examples.processing.presence_detection_sparse as presence_detection  # isort:skip
import feature_processing as feature_proc  # isort:skip
import keras_processing as kp  # isort:skip

if kp.tf_version != "1":
    print("To run this demo you need to install Tensorflow 1.x!")
    sys.exit(1)

answer = input('This model does not work properly with RSS version 2.0. Continue anyway [y,N]?\n')
if not (answer.lower() == 'yes' or answer.lower() == 'y'):
    sys.exit(1)

# Demo config:
SENSOR_SPEAKER = [2, 4]
USE_LEDS = True
PLAYLIST = "Oldies"
USE_PRESENCE = False
SENSOR_PRESENCE = [2]
UPDATE_RATE_PRESENCE = 80
PRESENCE_RANGE = [0.18, 2]
PRESENCE_ZONES = [.6, 1, 1.4, 1.8]  # Switch to speaker mode at closest zone

# Demo behavior
VOL_COOL_DOWN = 5
SWIPE_COOL_DOWN = 200
VOL_SKIPS = 10
SWIPEMODE = [0, 1, 6, 7]
SWIPE_COLOR = [100, 0, 100]
VOL_STEP = 100/16
PLAY_PAUSE_COOL_DOWN = 50
IDLE_COUNT = 30 * 60  # 30 seconds to switch from speaker to presence mode
PRINT_PREDICTIONS = True

if USE_LEDS:
    from acc_bstick_lib import MPBlinkstickWrapper  # isort:skip
else:
    import unittest  # isort:skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--load-train-set", dest="server",
                        help="IP of streaming server", default="127.0.0.1")
    args = parser.parse_args()

    filename = "model_data_speaker_sparse.npy"

    keras_proc = kp.MachineLearning()
    model_data, message = keras_proc.load_model(filename)

    print(message, "\n")

    if not model_data["loaded"]:
        print("Failed to load model!")
        sys.exit(1)

    try:
        client = SocketClient(args.server)
    except Exception:
        print("Failed to connect to server at {}!\n".format(args.server))
        traceback.print_exc()
        sys.exit(1)

    conf_speaker = model_data["sensor_config"]
    conf_speaker.sensor = SENSOR_SPEAKER
    feature_list = model_data["feature_list"]
    feature_list[0]["sensors"] = [SENSOR_SPEAKER[0]]
    feature_list[1]["sensors"] = [SENSOR_SPEAKER[1]]
    frame_settings = model_data["frame_settings"]

    frame_settings["auto_threshold"] = 1.5
    frame_settings["dead_time"] = 30
    frame_settings["auto_offset"] = 15

    frame_settings["collection_mode"] = "auto_feature_based"

    feature_process = feature_proc.FeatureProcessing(conf_speaker)
    feature_process.set_feature_list(feature_list)
    feature_process.set_frame_settings(frame_settings)

    handles = init_demo()
    handles["feature_process"] = feature_process
    handles["keras_proc"] = keras_proc

    # get session config for speaker mode
    info_speaker = client.setup_session(conf_speaker)
    handles["dist_processors"], handles["dist_tags"] = setup_distance_detectors(
        conf_speaker,
        info_speaker,
        SENSOR_SPEAKER
    )

    try:
        client.start_session()
        client.stop_session()
    except Exception:
        print("Failed to start session!")
        traceback.print_exc()
        sys.exit(1)

    demo_mode = "speaker"
    if USE_PRESENCE:
        # get session config for presence mode
        demo_mode = "presence"
        conf_presence = presence_detection.get_sensor_config()
        conf_presence.sensor = SENSOR_PRESENCE
        conf_presence.range_interval = PRESENCE_RANGE
        info_presence = client.setup_session(conf_presence)
        handles["presence_processor"] = setup_presence_detector(conf_presence, info_presence)

    if USE_PRESENCE:
        start_mode = "presence"
    else:
        start_mode = "speaker"
    print("Starting demo in {}-mode!".format(start_mode))

    interrupt_handler = utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end demo")

    client.start_session()

    while not interrupt_handler.got_signal:
        try:
            info, sweep = client.get_next()

            if demo_mode == "presence":
                new_mode = do_presence_mode(info, sweep, handles)
            else:
                data = {
                    "sweep_data": sweep,
                    "sensor_config": conf_speaker,
                    "session_info": info_speaker,
                }
                new_mode = do_speaker_mode(info, data, handles)

            # switch between presence and speaker mode
            if new_mode != demo_mode:
                demo_mode = new_mode
                handles["led_handle"].double_flash("#000000")
                time.sleep(1)
                client.stop_session()

                if demo_mode == "presence":
                    print("Switching to presence mode!\n")
                    handles["led_handle"].double_flash("#000000")
                    time.sleep(1)
                    handles["idle_counts"] = 0
                    info_presence = client.setup_session(conf_presence)
                    handles["presence_processor"] = setup_presence_detector(
                        conf_presence,
                        info_presence
                    )
                    if handles["play_mode"] == "play":
                        color = "#00ff00"
                    else:
                        color = "#ff0000"
                    handles["led_handle"].set_color(color, pos=[3, 4], brightness=0.1)
                else:
                    print("Switching to speaker mode!\n")
                    info_speaker = client.setup_session(conf_speaker)
                    handles["led_handle"].set_color("#00ff00", pos=[3, 4], brightness=0.1)

                client.start_session()
        except Exception:
            traceback.print_exc()
            break

    print("Disconnecting...")
    if handles["play_mode"] == "play":
        handles["lms_handle"].queue.put("PAUSE")
    handles["lms_handle"].stop()
    handles["led_handle"].off()
    handles["led_handle"].exit()
    client.disconnect()


def do_speaker_mode(info, data, handles):
    ml_frame_data = handles["feature_process"].feature_extraction(data)
    feature_map = ml_frame_data["current_frame"]["feature_map"]
    complete = ml_frame_data["current_frame"]["frame_complete"]

    if handles["vol_mode"] and handles["vol_skips"]:
        handles["vol_skips"] -= 1
        complete = None
        feature_map = None

    if handles["play_pause_cool_down"]:
        handles["play_pause_cool_down"] -= 1
        complete = None
        feature_map = None

    if handles["swipe_mode"]:
        if handles["swipe_cool_down"]:
            handles["swipe_cool_down"] -= 1
        else:
            handles["swipe_mode"] = False
            handles["swipe_cool_down"] = SWIPE_COOL_DOWN
            handles["leds_handle"].set_color("#00ff00", pos=[3, 4], brightness=0.1)

    if USE_PRESENCE:
        handles["idle_counts"] += 1

    # check if hand is over sensor
    hand_postion = get_hand_position(data["sweep_data"], handles)

    if complete and feature_map is not None:
        predict = handles["keras_proc"].predict(feature_map)[0]
        prediction_label = predict["prediction"]
        handles["idle_counts"] = 0

        # clean up prediction label
        if "left" in prediction_label:
            prediction_label = "left"
        elif "right" in prediction_label:
            prediction_label = "right"
        if "up_down" in prediction_label:
            prediction_label = "play_pause"

        if prediction_label in ["volume", "down"]:
            if hand_postion is None:
                prediction_label = "empty"
            else:
                prediction_label = hand_postion

        # Debug print prediction
        print_prediction = False
        if handles["vol_mode"]:
            if prediction_label in ["up", "down", "volume"]:
                print_prediction = True
        else:
            print_prediction = True

        if PRINT_PREDICTIONS and print_prediction:
            print("Predicted '{}' @ {:.2f}%".format(
                  prediction_label,
                  predict["confidence"] * 100
                  ))

        do_speaker_action(prediction_label, handles)

    # Check for mode switch
    if handles["idle_counts"] == IDLE_COUNT:
        return "presence"
    else:
        return "speaker"


def get_hand_position(sweep_data, handles):
    p = {}
    distances = []
    presences = []
    for s, tag in enumerate(handles["dist_tags"]):
        p[tag] = handles["dist_processors"][tag].process(sweep_data[s])
        distances.append(p[tag]["presence_distance"])
        presences.append(np.max(p[tag]["depthwise_presence"]))

    distances = np.array(distances)
    presences = np.array(presences)

    if np.max(presences) <= 5:
        return None
    if (distances > 0.20).all():
        return "up"
    elif (distances < 0.20).all():
        return "down"
    else:
        return None


def do_speaker_action(predciction, handles):
    h = handles
    leds = h["led_handle"]
    lms = h["lms_handle"]

    # Check if swipe is performed
    if h["swipe_mode"]:
        if predciction not in ["left", "right"]:
            h["swipe_mode"] = False
            h["swipe_cool_down"] = SWIPE_COOL_DOWN
            leds.double_flash("#000000")
            leds.set_color("#00ff00", pos=[3, 4], brightness=0.1)

    if predciction == "left" and not h["vol_mode"]:
        h["swipe_cool_down"] = SWIPE_COOL_DOWN
        if h["swipe_mode"] is False:
            h["swipe_mode"] = True
            leds.set_color("#6a0dad", pos=[0, 1, 3, 4, 6, 7], brightness=0.1)
        else:
            lms.queue.put("SKIP")
            leds.swipe_left("#6a0dad")
            leds.set_color("#6a0dad", pos=[0, 1, 3, 4, 6, 7], brightness=0.1)
            h["play_mode"] = "play"

    elif predciction == "right" and not h["vol_mode"]:
        h["swipe_cool_down"] = SWIPE_COOL_DOWN
        if h["swipe_mode"] is False:
            h["swipe_mode"] = True
            leds.set_color("#6a0dad", pos=[0, 1, 3, 4, 6, 7], brightness=0.1)
        else:
            lms.queue.put("PREVIOUS")
            leds.swipe_right("#6a0dad")
            h["play_mode"] = "play"
            leds.set_color("#6a0dad", pos=[0, 1, 3, 4, 6, 7], brightness=0.1)

    # Check if play/pause is performed
    elif predciction == "play_pause" and not h["vol_mode"] and not h["swipe_mode"]:
        if h["play_mode"] == "pause":
            color = "#00ff00"
            h["play_mode"] = "play"
        else:
            color = "#ff0000"
            h["play_mode"] = "pause"
        leds.double_flash(color)
        leds.set_color(color, pos=[3, 4], brightness=0.1)
        lms.queue.put("PAUSE")
        h["play_pause_cool_down"] = PLAY_PAUSE_COOL_DOWN

    # Check if volume change is performed
    elif predciction == "up" and not h["swipe_mode"]:
        h["vol"] += VOL_STEP
        h["vol"] = min(h["vol"], 100)

    elif predciction == "down" and not h["swipe_mode"]:
        h["vol"] -= VOL_STEP
        h["vol"] = max(h["vol"], 0)

    if predciction in ["up", "down", "volume"] and not h["swipe_mode"]:
        level = int(h["vol"] / (100 / 8))
        led_on = list(range(level))

        if not h["vol_mode"]:
            h["vol_mode"] = True
            print("Switching to continuous mode")
            frame_settings = {
                "collection_mode": "continuous",
                "rolling": "rolling"
            }
            handles["feature_process"].set_frame_settings(frame_settings)
            leds.set_color("#6a0dad", pos=led_on, brightness=0.1)
        else:
            if level != h["last_led"]:
                if level > h["last_led"]:
                    color = "#0000ff"
                else:
                    color = "#ff0000"
                leds.set_color(color, pos=led_on, brightness=0.1)
        h["last_led"] = level

        if "volume" not in predciction:
            if int(h["vol"]) != h["last_vol"]:
                print("Setting volume to {}".format(int(h["vol"])))
                lms.queue.put({"cmd": "volume", "percent": h["vol"]})
            h["last_vol"] = int(h["vol"])

    # Volume mode cool down
    if h["vol_mode"] and predciction not in ["up", "down", "volume"]:
        if h["vol_cool_down"] > 0:
            h["vol_cool_down"] -= 1
        else:
            h["vol_mode"] = False
            h["vol_cool_down"] = VOL_COOL_DOWN
            leds.double_flash("#000000")
            if h["play_mode"] == "play":
                color = "#00ff00"
            else:
                color = "#ff0000"
            leds.set_color(color, pos=[3, 4], brightness=0.1)
            print("Switching to auto mode")
            h["feature_process"].set_frame_settings({"collection_mode": "auto_feature_based"})

    if h["vol_skips"] == 0:
        h["vol_skips"] = VOL_SKIPS


def do_presence_mode(info, sweep, handles):
    leds = handles["led_handle"]

    p = handles["presence_processor"].process(sweep)
    distance = p["presence_distance"]
    presence_level = 0
    if p["presence_detected"]:
        if distance < PRESENCE_ZONES[0]:
            presence_level = 4
        elif distance < PRESENCE_ZONES[1]:
            presence_level = 3
        elif distance < PRESENCE_ZONES[2]:
            presence_level = 2
        elif distance < PRESENCE_ZONES[3]:
            presence_level = 1

    if presence_level != handles["presence_level_last"]:
        if presence_level == 0:
            leds.off()
        else:
            leds.set_color("#ffffff", pos=range(2, presence_level + 2), brightness=0.2)
    handles["presence_level_last"] = presence_level

    # Swtich to speaker mode
    if presence_level == 4:
        return "speaker"
    else:
        return "presence"


def init_demo():
    q1 = Queue()
    lms_handle = ThreadedLMSContoller(q1)
    lms_handle.start()
    time.sleep(0.1)

    if USE_LEDS:
        led_handle = MPBlinkstickWrapper()
        time.sleep(0.1)
        led_handle.knightrider("#ff0000", reversed=True, dt=0.2)
        time.sleep(1)
        led_handle.knightrider("#ff0000", dt=0.2)
        time.sleep(1)
    else:
        led_handle = unittest.mock.MagicMock()

    vol = 50
    try:
        lms_handle.queue.put({"cmd": "search_and_play", "playlist": PLAYLIST})
        lms_handle.queue.put({"cmd": "volume", "percent": vol})
        lms_handle.queue.put("SHUFFLE SONGS")
    except Exception:
        traceback.print_exc()
        print("Could not start playlist!")

    if not USE_PRESENCE:
        led_handle.set_color("#00ff00", pos=[3, 4], brightness=0.1)

    data_container = {
        "lms_handle": lms_handle,
        "led_handle": led_handle,
        "vol_mode": False,
        "vol_cool_down": VOL_COOL_DOWN,
        "vol_skips": VOL_SKIPS,
        "vol": vol,
        "last_vol": vol,
        "swipe_cool_down": SWIPE_COOL_DOWN,
        "swipe_mode": False,
        "play_pause_cool_down": 0,
        "play_mode": "play",
        "idle_counts": 0,
        "presence_level": 0,
        "presence_level_last": 0,
    }
    return data_container


def setup_distance_detectors(config, session_info, sensors):
    processing_config = presence_detection.get_processing_config()
    processing_config.inter_frame_fast_cutoff = 100
    processing_config.inter_frame_slow_cutoff = 1
    processing_config.inter_frame_deviation_time_const = 0.01
    processing_config.intra_frame_time_const = 0.01
    processing_config.intra_frame_weight = 0.7
    processing_config.detection_threshold = 0
    processors = {}
    tags = []
    sensor_id = 0
    config.sensor = sensors[sensor_id]
    for s in sensors:
        tag = "sensor_{}".format(s)
        tags.append(tag)
        processors[tag] = presence_detection.Processor(
            config,
            processing_config,
            session_info
        )
        processors[tag].depth_filter_length = 1
        sensor_id += 1
        if sensor_id < len(sensors):
            config.sensor = sensors[sensor_id]
    config.sensor = sensors

    return processors, tags


def setup_presence_detector(config, session_info):
    processing_config = presence_detection.get_processing_config()
    processing_config.inter_frame_fast_cutoff = 100
    processing_config.inter_frame_slow_cutoff = 1
    processing_config.detection_threshold = 1.5
    processing_config.inter_frame_deviation_time_const = 0.4
    processing_config.intra_frame_time_const = 0.1
    processing_config.output_time_const = 0.4
    processor = presence_detection.Processor(
        config,
        processing_config,
        session_info
    )

    return processor


class ThreadedLMSContoller(threading.Thread):
    def __init__(self, queue, ip="127.0.0.1", port=9000, player_name="DemoPlayer"):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue

        try:
            self.controller = SqueezeBoxController(ip, port)
        except Exception as e:
            print("LMS contorller not ready!\n", e)
            self.controller = None
        else:
            self.lms_param = {
                "player": player_name,
                "command": None,
                "query": None,
            }
            self.player = player_name

    def run(self):
        while True:
            cmd = self.queue.get()
            if cmd == "kill":
                break
            else:
                self.send_cmd(cmd)
                time.sleep(0.1)

    def stop(self):
        self.queue.put("kill")

    def send_cmd(self, cmd):
        if self.controller is None:
            return
        try:
            if isinstance(cmd, dict):
                if cmd["cmd"] == "volume":
                    self.set_volume(cmd["percent"])
                elif cmd["cmd"] == "search_and_play":
                    self.search_and_play(cmd["playlist"])
            else:
                self.lms_param["command"] = cmd
                self.controller.simple_command(self.lms_param)
        except Exception as e:
            print("Failed to communicate with LMS player!")
            print("Tried to send:\n", cmd)
            print(e)

    def set_volume(self, volume):
        try:
            self.controller._make_request(self.player, ["mixer", "volume", str(int(volume))])
        except Exception as e:
            print("Failed to set Volume")
            print(e)

    def search_and_play(self, search_term, mode="PLAYLIST"):
        if self.controller is None:
            return
        try:
            self.lms_param["type"] = mode
            self.lms_param["term"] = search_term
            self.controller.search_and_play(self.lms_param)
        except Exception as e:
            print("Failed to communicate with LMS player!")
            print("Tried to send:\n", self.lms_param)
            print(e)

    def send_query(self, cmd):
        if self.controller is None:
            return
        try:
            self.lms_param["query"] = cmd
            query = self.controller.simple_query(self.lms_param)
            if cmd == "VOLUME":
                return int(query.split("at")[1].split("p")[0])
            else:
                return query
        except Exception as e:
            print("Failed to communicate with LMS player!")
            print("Tried to send:\n", cmd)
            print(e)


if __name__ == "__main__":
    main()
