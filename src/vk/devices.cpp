#include "support/context.h"
#include "support/device.h"

#include <cstring>
#include <iostream>

int main(int argc, char **argv) {
    support::ContextOptions options;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            options.device = argv[++i];
        } else if (std::strcmp(argv[i], "--validation") == 0) {
            options.validation = true;
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--device <index|name>] [--validation]\n";
            return 2;
        }
    }

    try {
        if (options.device.empty()) {
            bool any_suitable = false;
            for (const support::DeviceReport &report :
                 support::probe_devices()) {
                std::cout << report;
                any_suitable |= report.suitable();
            }
            return any_suitable ? 0 : 1;
        }

        support::Context ctx(options);
        std::cout << ctx.report;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
