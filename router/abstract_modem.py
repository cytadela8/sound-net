import abc
from queue import Queue
import threading
import hamming_codec

import numpy as np
import sounddevice as sd

sd.default.samplerate = 44100


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
            if self.roffset >= self.SIZE:
                self.roffset -= self.SIZE
                chunk = np.append(chunk, self.buffer[0:self.roffset])
            self.available -= toread

        with self.notify:
            self.notify.notify()
        return chunk

    def write(self, s):
        while len(s) > 0:
            while self.SIZE == self.available:
                # print("Waiting ", len(s), self.available)
                with self.notify:
                    self.notify.wait()
            with self.lock:
                towrite = min(self.SIZE - self.woffset, self.SIZE - self.available,
                              len(s))
                self.buffer[self.woffset:self.woffset + towrite] = s[:towrite]
                self.woffset += towrite
                self.available += towrite
                if self.woffset == self.SIZE:
                    self.woffset = 0
            s = s[towrite:]


class Audio:
    _buffer = FloatLoop()

    CHANNELS = 1
    RATE = 44100

    def __init__(self, chunk_size):
        self.handlers = []
        self.buffer = np.array([], dtype=np.float32)
        self.prev_chunk = None

        def callback(in_data: np.ndarray, out_data: np.ndarray, frames: int,
                     time, status):
            # using Numpy to convert to array for processing
            # audio_data = np.fromstring(in_data, dtype=np.float32)
            out_data.fill(0)
            channel = self._buffer.read(frames)
            channel = np.append(channel, np.zeros(frames - len(channel),
                                                  dtype=np.float32))
            out_data[:, 0] = channel
            self.buffer = np.append(self.buffer, in_data[:, 0])
            while len(self.buffer) > chunk_size:
                chunk = self.buffer[:chunk_size]
                self.buffer = self.buffer[chunk_size:]
                for handler in self.handlers:
                    handler(chunk, self.prev_chunk)
                self.prev_chunk = chunk

        self.stream = sd.Stream(channels=1, callback=callback,
                                samplerate=self.RATE, dtype='float32')
        self.stream.start()

    def close(self):
        # stop stream (6)
        self.stream.stop()
        self.stream.close()

    def play(self, signal: np.ndarray):
        """
        Asynchronously plays given signal
        """
        self._buffer.write(signal)

    def register_handler(self, handler):
        self.handlers.append(handler)


class AbstractModem(abc.ABC):
    """Implements conversion from data to signal"""

    @abc.abstractmethod
    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        """Converts an array of bytes into the sound wave"""

    def __init__(self, buf_size=1024, chunk_size=4410):
        self._buf_size = buf_size
        self.chunk_size = chunk_size
        self._buffer = Queue(buf_size)
        self._player = Audio(chunk_size)

    def write_bytes(self, data: bytes|list[int]):
        """
        Writes data to send. If buffer is full will block until
        all data is in buffer.
        """
        self._buffer.put([data])

    def flush(self):
        """
        Empties the buffer. Should not be used from two threads at once!
        """
        [data] = self._buffer.get()
        print("SENDING", data)
        signal = self.bytes_to_signal(data)
        self._player.play(signal)

    def start(self):
        while True:
            # Without wait_nonempty this would be a busyloop
            self.flush()


class Encoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, data):
        """gets a list of some data, returns encoded list"""

    @abc.abstractmethod
    def decode(self, data):
        """gets a list of encoded data and returns pair (decoded,
        corrected_error_count) """


class StackedModem(AbstractModem):
    out = []
    packet_buffer = []
    IN_PACKET = "in"
    NO_PACKET = "out"

    def __init__(self, encoders: [Encoder], buf_size=1024, chunk_size=4410):
        super().__init__(buf_size, chunk_size)

        self._player.register_handler(self.handle_frame)
        self.encoders = encoders
        self.handlers = []

    @abc.abstractmethod
    def data_to_signal(self, data) -> np.ndarray:
        """Converts encoded data into the sound wave"""

    @abc.abstractmethod
    def in_frame_decoder(self, frame: np.ndarray, prev_frame: np.ndarray):
        """"""

    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        for encoder in self.encoders:
            data = encoder.encode(data)
        return self.data_to_signal(data)

    def handle_frame(self, frame: np.ndarray, prev_frame: np.ndarray):
        if prev_frame is None:
            return []
        decoded, status = self.in_frame_decoder(frame, prev_frame)
        if status == self.NO_PACKET:
            packet = self.packet_buffer
            self.packet_buffer = []
            if len(packet) != 0:
                print("RECEIVED", packet)

            for i, encoder in enumerate(reversed(self.encoders)):
                if len(packet) == 0:
                    return packet
                packet, errors = encoder.decode(packet)
                print(f"Decoded stage={i} (error={errors}), data {packet}")
            for handler in self.handlers:
                handler(packet)
        else:
            if len(self.packet_buffer) == 0:
                print("Receiving", flush=True, end="")
            else:
                print(".", flush=True, end="")
            self.packet_buffer += decoded

    def register_handler(self, handler):
        self.handlers.append(handler)


class TrivialModemSender(AbstractModem):

    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        return np.array(data)
