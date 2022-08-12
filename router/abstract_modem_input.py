import abc
from queue import Queue
import threading
import time

import numpy as np
import matplotlib.pyplot as plt

import pyaudio


class Recorder:

  p = pyaudio.PyAudio()

  CHANNELS = 1
  RATE = 44100

  stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True)
  

  def record(self, sample_count: int) -> np.ndarray:
    """
    Synchronously records signal from the input
    """
    return np.frombuffer(self.stream.read(sample_count), dtype=np.float32)
  

class AbstractModemReceiver(abc.ABC):

  @abc.abstractmethod
  def wait_for_data(self):
    """
    Blocks until data signal is detected.
    It may do for example initial synchronization.
    """
    pass

  @abc.abstractmethod
  def read_byte(self) -> int | None:
    """
    Reads one byte from the channel.
    Should return None if no more bytes can be read.
    """
    pass

  def read(self) -> bytes:
    """
      Read bytes.
    """
    self.wait_for_data()
    data = []
    while next_byte:=self.read_byte():
      data.append(next_byte)
    return Packet(bytes(data))
  

class SimpleModemReceiver(AbstractModemReceiver):

  CHUNK = 4410
  NOISE_LEVEL = 0.1

  def __init__(self) -> None:
    self.recorder = Recorder()

  def read_byte(self) -> int | None:
    data = self.recorder.record(self.CHUNK)
    transformed = np.fft.rfft(data)
    return np.argmax(np.abs(transformed)[10:])
  
  def wait_for_data(self):
    pass