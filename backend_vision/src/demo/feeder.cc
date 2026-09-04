#include "demo/feeder.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

namespace {
std::string getYouTube(const std::string& youtube_url);

const char* feederTypeToString(FeederType type) {
    switch (type) {
    case FeederType::CAMERA:
        return "CAMERA";
    case FeederType::VIDEO:
        return "VIDEO";
    case FeederType::IPCAMERA:
        return "IPCAMERA";
    case FeederType::YOUTUBE:
        return "YOUTUBE";
    default:
        return "UNKNOWN";
    }
}

bool openCaptureBySetting(const FeederSetting& setting, cv::VideoCapture& cap, bool& delay_on) {
    switch (setting.feeder_type) {
    case FeederType::CAMERA: {
        cap.open(stoi(setting.src_path), cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
        cap.set(cv::CAP_PROP_FPS, 30);
        delay_on = false;
        break;
    }
    case FeederType::VIDEO: {
#ifdef _WIN32
        cap.open(setting.src_path, cv::CAP_FFMPEG);
#else
        cap.open(setting.src_path);
#endif
        delay_on = true;
        break;
    }
    case FeederType::IPCAMERA: {
        cap.open(setting.src_path);
        delay_on = false;
        break;
    }
    case FeederType::YOUTUBE: {
        cap.open(getYouTube(setting.src_path));
        delay_on = true;
        break;
    }
    }

    return cap.isOpened();
}

bool rewindCapture(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        return false;
    }

    if (!cap.set(cv::CAP_PROP_POS_FRAMES, 0)) {
        return false;
    }

    // Some backends report the first frame as 0 or 1 after seek.
    double pos = cap.get(cv::CAP_PROP_POS_FRAMES);
    return pos <= 1.0;
}

void logFeederOpenStatus(const FeederSetting& setting, const cv::VideoCapture& cap) {
    if (cap.isOpened()) {
        return;
    }

    std::cerr << "[FEEDER] type=" << feederTypeToString(setting.feeder_type)
              << " src=" << setting.src_path
              << " opened=false";
    std::cerr << std::endl;
}

std::string getYouTube(const std::string& youtube_url) {
#ifdef _MSC_VER
    // (kibum): Need to implement.
    std::cerr << "Youtube input is not implemented for MSVC.\n";
    return "";
#else
    char buf[128];
    std::string URL;
    std::string cmd = "yt-dlp -f \"best[height<=720][width<=1280]\" -g " + youtube_url;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return URL;
    }

    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        URL += buf;
    }
    pclose(pipe);

    if (!URL.empty()) {
        URL.erase(URL.find('\n'));
    }
    return URL;
#endif
}
}  // namespace

Feeder::Feeder(const FeederSetting& feeder_setting) : mFeederSetting(feeder_setting) {
    openCaptureBySetting(mFeederSetting, mCap, mDelayOn);

    if (mFeederSetting.feeder_type == FeederType::VIDEO && mCap.isOpened()) {
        double fps = mCap.get(cv::CAP_PROP_FPS);
        if (fps >= 1.0 && fps <= 240.0) {
            mVideoFps = fps;
        }
    }

    if (!mCap.isOpened() && !mLoggedOpenFailure) {
        logFeederOpenStatus(mFeederSetting, mCap);
        mLoggedOpenFailure = true;
    }
}

bool Feeder::consumeFrame(cv::Mat& frame, int64_t& frame_index) {
    int64_t latest_index = frame_index;
    auto sc = mFeederBuffer.getLatest(frame, latest_index);
    if (sc != MatBuffer::OK) {
        return false;
    }
    if (latest_index == frame_index) {
        return false;
    }
    frame_index = latest_index;
    return !frame.empty();
}

void Feeder::produceFrames() {
    mFeederBuffer.open();
    while (mIsFeederRunning.load(std::memory_order_relaxed)) {
        if (mCap.isOpened()) {
            const bool is_loop_source =
                (mFeederSetting.feeder_type == FeederType::VIDEO ||
                 mFeederSetting.feeder_type == FeederType::YOUTUBE);
            int delay_ms = 0;
            if (mDelayOn) {
                if (mFeederSetting.feeder_type == FeederType::VIDEO && mVideoFps >= 24.0 &&
                    mVideoFps <= 240.0) {
                    delay_ms = std::max(1, (int)std::lround(1000.0 / mVideoFps));
                } else {
                    delay_ms = 33;
                }
            }
            produceFramesInternal(mCap, delay_ms);
            if (!mIsFeederRunning.load(std::memory_order_relaxed)) {
                break;
            }

            if (is_loop_source) {
                bool rewind_ok = rewindCapture(mCap);
                if (!rewind_ok || !mCap.isOpened()) {
                    mCap.release();
                    openCaptureBySetting(mFeederSetting, mCap, mDelayOn);
                    if (mFeederSetting.feeder_type == FeederType::VIDEO && mCap.isOpened()) {
                        double fps = mCap.get(cv::CAP_PROP_FPS);
                        if (fps >= 1.0 && fps <= 240.0) {
                            mVideoFps = fps;
                        }
                    }
                }
            } else {
                rewindCapture(mCap);
            }
        } else {
            const bool reopened = openCaptureBySetting(mFeederSetting, mCap, mDelayOn);
            if (reopened) {
                if (mFeederSetting.feeder_type == FeederType::VIDEO) {
                    double fps = mCap.get(cv::CAP_PROP_FPS);
                    if (fps >= 1.0 && fps <= 240.0) {
                        mVideoFps = fps;
                    }
                }
                continue;
            }

            produceFramesInternalDummy();
        }
    }
    mFeederBuffer.close();
}

void Feeder::produceFramesInternal(cv::VideoCapture& cap, int delay_ms) {
    Benchmarker benchmarker;
    int perf_count = 0;
    bool has_produced_frame = false;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        cap >> frame;
        if (!mIsFeederRunning.load(std::memory_order_relaxed)) {
            break;
        }
        if (frame.empty()) {
            if (!has_produced_frame && !mLoggedInitialReadFailure) {
                std::cerr << "[FEEDER] empty frame"
                          << " type=" << feederTypeToString(mFeederSetting.feeder_type)
                          << " src=" << mFeederSetting.src_path
                          << " pos_frames=" << cap.get(cv::CAP_PROP_POS_FRAMES)
                          << " frame_count=" << cap.get(cv::CAP_PROP_FRAME_COUNT)
                          << std::endl;
                mLoggedInitialReadFailure = true;
            }
            break;
        }

        mFeederBuffer.put(frame);
        has_produced_frame = true;

        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        benchmarker.end();
        if ((perf_count++ % 60) == 0) {
            // printf("[FEED] idx=%d interval=%.3fms fps=%.2f\n", index,
            //        benchmarker.getSec() * 1000, benchmarker.getFPS());
            // fflush(stdout);
        }
    }
}

void Feeder::produceFramesInternalDummy() {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(frame, "Dummy Feeder", cv::Point(140, 190), cv::FONT_HERSHEY_DUPLEX,
                    1.5, cv::Scalar(0, 255, 0), 2);
        if (frame.empty() || !mIsFeederRunning.load(std::memory_order_relaxed)) {
            break;
        }

        mFeederBuffer.put(frame);

        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        benchmarker.end();
    }
}
