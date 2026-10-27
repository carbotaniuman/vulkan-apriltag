#include "quad_decoding.h"
#include "vk/apriltag.h"

#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

const char *const STAGES[] = {"threshold", "segmentation", "boundaries",
                              "cluster_grouping", "fit_quad"};

struct Stats {
    double median;
    double min;
};

Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front()};
}

void usage(const char *argv0) {
    std::cerr << "usage: " << argv0
              << " <image.png> [--device <index|name>] [--validation] "
                 "[--staging] "
                 "[--iterations <n>] [--warmup <n>] [--dispatches] "
                 "[--decode-threads <n>]\n";
}

int main(int argc, char **argv) {
    support::ContextOptions options;
    std::string input_path;
    int iterations = 1;
    int warmup = 0;
    bool dispatches = false;
    unsigned decode_threads = 4;
    for (int i = 1; i < argc; i++) {
        const bool has_value = i + 1 < argc;
        if (std::strcmp(argv[i], "--device") == 0 && has_value) {
            options.device = argv[++i];
        } else if (std::strcmp(argv[i], "--iterations") == 0 && has_value) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--warmup") == 0 && has_value) {
            warmup = std::max(0, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--decode-threads") == 0 && has_value) {
            decode_threads = unsigned(std::max(1, std::atoi(argv[++i])));
        } else if (std::strcmp(argv[i], "--dispatches") == 0) {
            dispatches = true;
        } else if (std::strcmp(argv[i], "--validation") == 0) {
            options.validation = true;
        } else if (std::strcmp(argv[i], "--staging") == 0) {
            options.force_staging = true;
        } else if (argv[i][0] != '-' && input_path.empty()) {
            input_path = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (input_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    try {
        int w, h, comp;
        stbi_uc *data = stbi_load(input_path.c_str(), &w, &h, &comp, STBI_grey);
        if (!data) {
            throw std::runtime_error("cannot load " + input_path + ": " +
                                     stbi_failure_reason());
        }
        const std::vector<uint8_t> image(data, data + size_t(w) * h);
        stbi_image_free(data);
        const uint32_t width = w;
        const uint32_t height = h;

        support::Context ctx(options);
        std::cout << "device: " << ctx.report.name << "\nimage: " << width
                  << "x" << height << "\ndecode threads: " << decode_threads
                  << "\n";
        avk::AprilTagPipeline pipeline(ctx, width, height);
        QuadDecoder decoder(decode_threads);

        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        support::GpuTimer timer(ctx);
        std::vector<double> upload_ms, submit_ms, frame_ms, decode_ms, total_ms;
        std::map<std::string, std::vector<double>> stage_ms;
        std::vector<std::pair<std::string, std::vector<double>>> mark_ms;
        std::vector<Detection> detections;
        size_t quad_count = 0;

        size_t dropping_frames = 0;
        uint32_t most_dropped = 0;
        for (int it = 0; it < warmup + iterations; it++) {
            const auto t0 = Clock::now();
            pipeline.upload(image.data());
            const auto t1 = Clock::now();
            ctx.submit_and_wait(
                [&](vk::CommandBuffer cmd) { pipeline.record(cmd, &timer); });
            const auto t2 = Clock::now();
            const auto corners = pipeline.corners();
            detections = decoder.decode(corners.data(), corners.size(), width,
                                        height, image.data());
            quad_count = corners.size();
            const uint32_t dropped = pipeline.dropped_points();
            if (dropped > 0) {
                dropping_frames++;
                most_dropped = std::max(most_dropped, dropped);
            }
            const auto t3 = Clock::now();
            if (it < warmup) {
                continue;
            }

            upload_ms.push_back(ms(t0, t1));
            submit_ms.push_back(ms(t1, t2));
            frame_ms.push_back(ms(t0, t2));
            decode_ms.push_back(ms(t2, t3));
            total_ms.push_back(ms(t0, t3));
            std::map<std::string, double> per_stage;
            const auto marks = timer.read();
            for (size_t m = 0; m < marks.size(); m++) {
                if (mark_ms.size() <= m) {
                    mark_ms.emplace_back(marks[m].first, std::vector<double>());
                }
                mark_ms[m].second.push_back(marks[m].second);
            }
            for (const auto &[label, t] : marks) {
                for (const char *stage : STAGES) {
                    if (label.rfind(stage, 0) == 0) {
                        per_stage[stage] += t;
                    }
                }
            }
            double gpu_total = 0;
            for (const auto &[stage, t] : per_stage) {
                stage_ms[stage].push_back(t);
                gpu_total += t;
            }
            stage_ms["gpu total"].push_back(gpu_total);
        }

        if (dropping_frames > 0) {
            std::cout << "warning: the cluster hash map was full in "
                      << dropping_frames << " of " << warmup + iterations
                      << " frames, which lost up to " << most_dropped
                      << " boundary points\n";
        }
        std::cout << quad_count << " quads passed the geometric checks, "
                  << detections.size() << " decoded:\n";
        for (const Detection &d : detections) {
            std::cout << "  id " << d.id << " (hamming " << d.hamming
                      << ", margin " << std::fixed << std::setprecision(1)
                      << d.decision_margin << ") centre ("
                      << std::setprecision(2) << d.center[0] << ", "
                      << d.center[1] << ") corners";
            for (int k = 0; k < 4; k++) {
                std::cout << " (" << d.corners[k][0] << ", " << d.corners[k][1]
                          << ")";
            }
            std::cout << "\n";
        }

        if (iterations > 1 || warmup > 0) {
            std::cout << "timing over " << iterations << " frames after "
                      << warmup << " warm-up frames (median / min, ms):\n";
            auto row = [](const std::string &name,
                          const std::vector<double> &v) {
                const Stats s = stats(v);
                std::cout << "  " << std::left << std::setw(36) << name
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(9) << s.median << " / " << std::setw(9)
                          << s.min << "\n";
            };
            for (const char *stage : STAGES) {
                if (stage_ms.count(stage)) {
                    row(std::string("gpu ") + stage, stage_ms[stage]);
                }
                for (const auto &[label, v] : mark_ms) {
                    if (dispatches && label.rfind(stage, 0) == 0) {
                        row("    " + label, v);
                    }
                }
            }
            if (stage_ms.count("gpu total")) {
                row("gpu total (timestamps)", stage_ms["gpu total"]);
            }
            row("host upload copy", upload_ms);
            row("record + submit + wait", submit_ms);
            row("frame: upload + gpu + wait", frame_ms);
            row("cpu decode", decode_ms);
            row("frame + decode", total_ms);
        }

        if (ctx.validation_errors) {
            std::cout << "validation: " << ctx.validation_errors << " errors\n";
            return 1;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
