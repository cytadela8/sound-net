#!/bin/env python3
import pyaudio
import wave
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

frames = []

def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
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

    stream.start_stream()

    time.sleep(2.5)

    stream.stop_stream()
    stream.close()
    p.terminate()

    sample_width = p.get_sample_size(FORMAT)
    return sample_width

if __name__ == '__main__':
    pass
