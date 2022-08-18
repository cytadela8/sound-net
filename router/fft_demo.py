import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sounddevice as sd
import time

CHUNK = 1024
CHANNELS = 1
RATE = 44100

sd.default.samplerate = RATE
sd.default.device = 'USB Audio Device'

frames = []


def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)


def record():
    stream = sd.InputStream(channels=CHANNELS, callback=callback,
                            samplerate=RATE, blocksize=CHUNK, dtype=np.float32)

    print("Start recording")

    fig, ax = plt.subplots()

    xs = range(CHUNK // 2 + 1)

    plt.ylim(-10, 10)

    trans, = plt.plot(xs, [0] * (CHUNK // 2 + 1))

    def draw_callback(i):
        if not frames or len(frames[-1]) != CHUNK:
            print("KEK")
            return trans,
        trans.set_data(xs, np.abs(np.fft.rfft(frames[-1])))
        return trans,

    stream.start_stream()

    animation.FuncAnimation(fig, draw_callback, frames=range(100), \
                            interval=10, blit=True, )

    plt.show()

    time.sleep(3)

    stream.stop_stream()
    stream.close()


if __name__ == '__main__':
    record()
