#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Won't work if not run as a root, use sudo or similar."
  exit 1
fi

if [ -z "$1" ]
    then echo "Please specify last byte of your desired address."
    exit 1
fi

sysctl net.ipv6.conf.all.disable_ipv6=1

ADDRESS="6.9.6.$1"
NETWORK="6.9.6.0"

./tuntun.py --name xyz --type tun
ip link set xyz up
ip addr add "$ADDRESS" dev xyz
ip route add "$NETWORK/24" via "$ADDRESS" dev xyz
