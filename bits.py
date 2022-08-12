from itertools import cycle
def from_bits(bitstring: str) -> bytes:
    if len(bitstring) % 8 != 0:
        raise ValueError("This is not a valid bitstring")
    if not set(bitstring).issubset({'0', '1'}):
        raise ValueError("Should be only 0s and 1s")
    return bytes(int(bitstring[8*i:8*(i+1)], 2) for i in range(len(bitstring) // 8))

def byte_to_bin(x: int) -> str:
    return bin(x)[2:].zfill(8)

def to_bits(arr: bytes) -> str:
    return "".join(byte_to_bin(x) for x in arr)
