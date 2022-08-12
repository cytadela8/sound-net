#!/usr/bin/env python
"""Utility for creating and using tun interfaces in python"""
import enum
import fcntl
import grp
from functools import wraps
import pwd
import struct
from argparse import ArgumentParser

from constants import IFF_TUN, IFF_TAP, TUNSETPERSIST, IFNAMSIZ, TUNSETIFF, TUNSETOWNER, TUNSETGROUP

from ethernet import EthernetPacket


class TunError(Exception):
    pass


def ensure_opened(f):
    @wraps(f)
    def inner(self, *args, **kwargs):
        if not self._device:
            raise TunError("Device not opened.")
        return f(self, *args, **kwargs)
    return inner


class DevFlags(enum.IntFlag):
    IFF_TUN = IFF_TUN
    IFF_TAP = IFF_TAP


class TunTapDevice:
    CLONE_DEVICE = "/dev/net/tun"
    IFR_STRUCT = f"{IFNAMSIZ}sh"

    def __init__(self, device_name: str | bytes | None, flags: DevFlags):
        """This may fail in case of insufficient permissions."""
        self.device_name = device_name.encode() if isinstance(device_name, str) else device_name or b""
        self._device = None
        self.flags = flags

    def open(self):
        if self._device:
            raise TunError("Device is already opened")
        clone_device = open(self.CLONE_DEVICE, "rb+", buffering=0)
        try:
            ifr = struct.pack(self.IFR_STRUCT, self.device_name, self.flags.value)
            result = fcntl.ioctl(clone_device, TUNSETIFF, ifr)
            self.device_name = struct.unpack(self.IFR_STRUCT, result)[0]
            self._device = clone_device
        except OSError as e:
            clone_device.close()
            raise TunError(f"Failed to open tun device {self.device_name}") from e

    @ensure_opened
    def set_ownership(self, user: str | int | None, group: str | int | None):
        userid = user if isinstance(user, int) else pwd.getpwnam(user).pw_uid if user else None
        groupid = group if isinstance(group, int) else grp.getgrnam(group).gr_gid if group else None
        try:
            if userid:
                fcntl.ioctl(self._device, TUNSETOWNER, userid)
            if groupid:
                fcntl.ioctl(self._device, TUNSETGROUP, groupid)
        except OSError as e:
            raise TunError(f"Could not set permissions for device {self.device_name}") from e

    @ensure_opened
    def set_persistent(self, status: bool = True):
        try:
            fcntl.ioctl(self._device, TUNSETPERSIST, 1 if status else 0)
        except OSError as e:
            raise TunError(f"Could not set required persistence {status} for device {self.device_name}") from e

    @ensure_opened
    def close(self):
        self._device.close()
        self._device = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def setup_device(cls, name: bytes | None, user: int | str | None, group: int | str | None):
        """
        Run this as root to get the persistent device set up.
        """
        with cls(name) as dev:
            dev.set_ownership(user, group)
            dev.set_persistent()
            return dev.device_name


class TapDevice(TunTapDevice):

    def __init__(self, device_name: str | bytes | None):
        super().__init__(device_name, DevFlags.IFF_TAP)
    
    @ensure_opened
    def read(self) -> EthernetPacket:
        """Read ethernet packet"""
        return EthernetPacket.from_bytes(self._device.read(EthernetPacket.MAX_LENGTH))

    @ensure_opened
    def write(self, packet: EthernetPacket):
        """Writes packet into device"""
        return self._device.write(bytes(packet))


class TunDevice(TunTapDevice):

    def __init__(self, device_name: str | bytes | None):
        super().__init__(device_name, DevFlags.IFF_TUN)
    
    @ensure_opened
    def read(self) -> bytes:
        return self._device.read(66000)

    @ensure_opened
    def write(self, packet: bytes):
        return self._device.write(packet)


# if __name__ == "__main__":
#     print(TapDevice.setup_device(b"abc", "janczarknurek", "janczarknurek"))

if __name__ == '__main__':
    parser = ArgumentParser("tuntap", description="Create tun/tap device on your system")
    parser.add_argument("--name", required=True, help="Device name, should be at most 8 bytes")
    parser.add_argument("--type", required=True, help="Type of device", choices=["tun", "tap"])
    args = parser.parse_args()
    DevCls = TunDevice if args.type == "tun" else TapDevice
    print(DevCls.setup_device(args.name.encode("ASCII"), 1000, 1000))

