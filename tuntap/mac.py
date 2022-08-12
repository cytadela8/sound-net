

class MAC:
    def __init__(self, address: bytes):
        if len(address) != 6:
            raise ValueError("MAC address should have exactly 6 bytes")
        self._address = address
    
    def __str__(self) -> str:
        return ":".join(self._address[i:i+1].hex() for i in range(6))
    
    def __eq__(self, other) -> bool:
      return self._address == other._address
      
    def __repr__(self) -> str:
        return str(self)

    def __bytes__(self) -> bytes:
        return self._address
