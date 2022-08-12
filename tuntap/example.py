from ethernet import EthernetPacket
from tuntun import TapDevice

def read_some(dev):
  for i in range(10):
    print(dev.read())

def write_some(dev):
  frame = EthernetPacket(
    b'\xdc\xfb\x48\x70\x19\x71',
    b'\xf2\x3b\x38\xed\xa3\xef',
    b'This is just an ethernet frame kek'
  )
  dev.write(frame)
  
if __name__ == '__main__':
  # Assume device abc exists and is on
  dev = TapDevice("abc")
  dev.open()
  # read_some(dev)
  write_some(dev)
