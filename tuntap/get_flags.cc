#include <sys/ioctl.h>
#include <linux/if_tun.h>
#include <linux/if.h>
#include <iostream>

#define _P(constant) std::cout << #constant "=" << constant << std::endl

/* Pull constants from linux library */

int main() {
        std::cout << "# This file is generated." << std::endl;
        _P(IFF_TUN);
        _P(IFF_TAP);
        _P(IFF_PERSIST);
        _P(TUNSETPERSIST);
        _P(TUNSETOWNER);
        _P(TUNSETGROUP);
        _P(TUNSETIFF);
        _P(IFNAMSIZ);
}
