from dataclasses import dataclass
from typing import Optional

from mac import MAC


ETHERTYPES = {
  None: "Pure",
  2048: "IPv4",
  34525: "IPv6"
}


@dataclass
class EthernetPacket:
    dst: MAC
    src: MAC
    payload: bytes
    ethertype: Optional[int] = None

    MAX_LENGTH = 1526

    def is_pure(self) -> bool:
        return self.ethertype is None

    @classmethod
    def from_bytes(cls, data: bytes) -> 'EthernetPacket':
        dst = MAC(data[4:10])
        src = MAC(data[10:16])
        ethertype = int.from_bytes(data[16:18], 'big')
        ethertype = ethertype if ethertype > cls.MAX_LENGTH else None
        payload = data[18:]
        return EthernetPacket(dst, src, payload, ethertype)
    
    def ethertype_string(self):
      ethertype = ETHERTYPES.get(self.ethertype, 'unknown')
      return f'({ethertype})'
    
    def __str__(self) -> str:
        return "\n".join((
            f"Destination: {self.dst}",
            f"Source: {self.src}",
            f"Ethertype: {self.ethertype} {self.ethertype_string()}",
            f"Payload: {self.payload.hex()}"
        ))

    def __bytes__(self) -> bytes:
        return b''.join(bytes(x) for x in (self.dst, self.src, len(self.payload).to_bytes(2, 'big'), self.payload))