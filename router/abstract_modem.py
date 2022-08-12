import abc
from queue import Queue
import threading
import time
import io

import numpy as np
import matplotlib.pyplot as plt

import pyaudio


class FloatLoop:
    SIZE = 44100

    def __init__(self):
        self.buffer = np.zeros(self.SIZE, dtype=np.float32)
        self.roffset = 0
        self.woffset = 0
        self.available = 0
        self.lock = threading.Lock()
        self.notify = threading.Condition()

    def read(self, n=-1):
        with self.lock:
            toread = min(self.available, n)
            if n == -1:
                toread = self.available
            chunk = self.buffer[
                    self.roffset: min(self.SIZE, self.roffset + toread)]
            self.roffset += toread
            if self.roffset >= self.available:
                self.roffset -= self.SIZE
                chunk = np.append(chunk, self.buffer[0:self.roffset])
            self.available -= toread

        with self.notify:
            self.notify.notify()
        return chunk

    def write(self, s):
        while self.SIZE == self.available:
            print("Waiting ", len(s), self.available)
            with self.notify:
                self.notify.wait()
        with self.lock:
            towrite = min(self.SIZE - self.woffset, self.SIZE - self.available, len(s))
            self.buffer[self.woffset:self.woffset + towrite] = s[:towrite]
            self.woffset += towrite
            self.available += towrite
            if self.woffset == self.SIZE:
                self.woffset = 0
        if towrite < len(s):
            self.write(s[towrite:])


class Audio:
    p = pyaudio.PyAudio()
    _buffer = FloatLoop()

    CHANNELS = 1
    RATE = 44100

    def __init__(self):
        def callback(in_data, frame_count, time_info, flag):
            # using Numpy to convert to array for processing
            # audio_data = np.fromstring(in_data, dtype=np.float32)
            out_data = self._buffer.read(frame_count * self.RATE)
            if len(out_data) != 0:
                print("PLAYING", out_data)
            out_data = np.append(out_data, np.zeros(frame_count * self.RATE - len(out_data),
                                                    dtype=np.float32))
            return out_data, pyaudio.paContinue

        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  output=True,
                                  input=True,
                                  stream_callback=callback)

        self.stream.start_stream()

    def close(self):
        # stop stream (6)
        self.stream.stop_stream()
        self.stream.close()

        # close PyAudio (7)
        self.p.terminate()

    def play(self, signal: np.ndarray):
        """
        Asynchronously plays given signal
        """
        self._buffer.write(signal)


class AbstractModemSender(abc.ABC):
    """Implements convertion from data to signal"""

    @abc.abstractmethod
    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        """Converts an array of bytes into the sound wave"""

    def __init__(self, buf_size=1024):
        self._buf_size = buf_size
        self._buffer = Queue(buf_size)
        self._player = Audio()

    def write_bytes(self, data: bytes):
        """
        Writes data to be send. If buffer is full will block until
        all data is in buffer.
        """
        for b in data:
            self._buffer.put(b)

    def _consume_buffer(self, wait_nonempty=False) -> bytes:
        """
        Returns the content of buffer as bytes. Consumes returned content.
        """
        data = list()
        if wait_nonempty:
            data.append(self._buffer.get())
        for _ in range(
                self._buf_size if not wait_nonempty else self._buf_size - 1):
            if self._buffer.empty():
                break
            data.append(self._buffer.get())
        return bytes(data)

    def flush(self, wait_nonempty=False):
        """
        Empties the buffer. Should not be used from two threads at once!
        """
        data = self._consume_buffer(wait_nonempty=wait_nonempty)
        print(data)
        signal = self.bytes_to_signal(data)
        print(signal)
        self._player.play(signal)

    def start(self):
        while True:
            # Without wait_nonempty this would be a busyloop
            self.flush(wait_nonempty=True)


class TrivialModemSender(AbstractModemSender):

    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        return np.array(data)


class NaiveModemSender(AbstractModemSender):
    CHUNK_SIZE = 4410
    MIN = 30.0
    SIN_RANGE = np.linspace(0, 2 * np.pi, CHUNK_SIZE)
    MUL = (2.0 ** (1 / 6))

    def wave_by_hz(self, hz: int):
        return np.sin(self.SIN_RANGE * hz)

    def wave_by_id(self, id: int):
        base_hz = int(self.MIN * (self.MUL ** id))
        tau = 0.7
        return np.sum([self.wave_by_hz(base_hz * i) * (tau ** i) for i in
                       range(1, 11, 1)], axis=0)

    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        # Use float32 because default is float64 and you will get garbage :P
        result = np.zeros(self.CHUNK_SIZE * len(data), dtype=np.float32)
        for i, el in enumerate(data):
            wave = np.sum([self.wave_by_id(b + 2) for b in range(8) if
                           (el & (1 << b)) != 0] + [self.wave_by_id(i % 2)],
                          axis=0)
            result[
            i * self.CHUNK_SIZE: (i + 1) * self.CHUNK_SIZE] = wave / np.max(
                np.abs(wave))
        return result


if __name__ == '__main__':
    modem = NaiveModemSender(10)


    def _writer():
        for i in range(256):
            modem.write_bytes(bytes([i for _ in range(1)]))


    t1 = threading.Thread(target=_writer)

    t1.start()

    t2 = threading.Thread(target=modem.start)

    t2.start()

    t1.join()
    t2.join()
