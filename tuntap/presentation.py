from tuntun import *
from base64 import b64encode, b64decode
import sys
import threading

dev = TunDevice("xyz")
dev.open()

def pchaj():
    while True:
        packet = dev.read()
        sys.stdout.buffer.write(b64encode(packet) + b'\n')


def ciąg():
    while True:
        encoded = sys.stdin.readline()
        packet = b64decode(encoded)
        dev.write(packet)

if __name__ == '__main__':
	  t1 = threading.Thread(target=pchaj); t1.start(); 
	  t2 = threading.Thread(target=ciąg); t2.start(); 
