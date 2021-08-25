import multiprocessing as mp
import queue
import traceback
from functools import partial
from time import sleep

import numpy as np


def dim(ct, f):
    return tuple(int(round(c * f)) for c in ct)


class BlinkstickWrapper:
    def __init__(self):
        from blinkstick import blinkstick

        self.bs = blinkstick.BlinkStick()

        self.stick = blinkstick.BlinkStickPro(
            r_led_count=8,
            g_led_count=8,
            b_led_count=8,
        )

        self.stick.connect()

    def sleep(self, t):
        sleep(t)

    def off(self):
        self.stick.off()

    def flash(self, color, steps=8):
        c = self.bs._hex_to_rgb(color)

        fs = np.sqrt(np.linspace(0, 1, steps))
        fs = np.concatenate([fs, np.flip(fs)])

        self.stick.clear()

        for f in fs:
            for i in range(8):
                self.stick.set_color(0, i, *dim(c, f))

            self.stick.send_data_all()

        self.stick.off()

    def double_flash(self, color, steps=8, mid_dt=0.1):
        self.flash(color, steps)
        sleep(mid_dt)
        self.flash(color, steps)

    def swipe_left(self, color, dt=0.02):
        self.swipe(color, "l", dt)

    def swipe_right(self, color, dt=0.02):
        self.swipe(color, "r", dt)

    def set_color(self, color, pos=[range(8)], brightness=0.5):
        c = self.bs._hex_to_rgb(color)

        self.stick.clear()

        for i in pos:
            self.stick.set_color(0, i, *dim(c, brightness))

        self.stick.send_data_all()

    def swipe(self, color, direction, dt):
        c = self.bs._hex_to_rgb(color)

        idxs = np.arange(8)
        if direction == "l":
            idxs = np.flip(idxs)

        for i in idxs:
            self.stick.clear()
            for j in [-1, 1]:
                k = i + j
                if 0 <= k < 8:
                    self.stick.set_color(0, k, *dim(c, 0.1))

            self.stick.set_color(0, i, *c)
            self.stick.send_data_all()
            sleep(dt)

        self.stick.off()

    def knightrider(self, color, reversed=False, dt=0.05):
        c = self.bs._hex_to_rgb(color)

        idxs = np.arange(4)
        if reversed:
            idxs = np.flip(idxs)

        for i in idxs:
            self.stick.clear()
            self.stick.set_color(0, i, *c)
            self.stick.set_color(0, 7 - i, *c)
            self.stick.send_data_all()
            sleep(dt)

        self.stick.off()


def MPWrap(objtype):
    class ParentProcess(object):
        def __init__(self):
            self.q = mp.Queue()
            self.p = ChildProcess(self.q)
            self.p.start()

        def __getattribute__(self, attr):
            if hasattr(objtype, attr) or attr in ["exit"]:
                return partial(self.__wrap_fun, attr)
            else:
                return super().__getattribute__(attr)

        def __wrap_fun(self, name, *args, **kwargs):
            self.q.put((name, args, kwargs))

    class ChildProcess(mp.Process):
        def __init__(self, queue):
            super().__init__(daemon=True)
            self.q = queue

            self.obj = objtype()

        def run(self):
            try:
                self._run()
            except Exception:
                traceback.print_exc()
                print("\n\n")

            while True:
                try:
                    self.q.get(timeout=0.01)
                except queue.Empty:
                    break

        def _run(self):
            while True:
                (fun_name, fun_args, fun_kwargs) = self.q.get()

                if fun_name == "exit":
                    break

                fun = getattr(self.obj, fun_name)
                fun(*fun_args, **fun_kwargs)

    return ParentProcess


MPBlinkstickWrapper = MPWrap(BlinkstickWrapper)
