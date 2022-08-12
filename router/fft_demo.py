import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pyaudio
import time

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

frames = []


def callback(in_data, frame_count, time_info, status):
    frames.append(np.fromstring(in_data, dtype=np.float32))
    return (None, pyaudio.paContinue)


def record():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    stream_callback=callback,
                    frames_per_buffer=CHUNK)

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

    myAnimation = animation.FuncAnimation(fig, draw_callback, frames=range(100), \
                                          interval=10, blit=True, )

    plt.show()

    time.sleep(3)

    stream.stop_stream()
    stream.close()
    p.terminate()

    sample_width = p.get_sample_size(FORMAT)
    return sample_width


if __name__ == '__main__':
    record()
