#ifndef DEMO_INCLUDE_FEEDER_H_
#define DEMO_INCLUDE_FEEDER_H_

#include <atomic>
#include <chrono>
#include <string>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

class Feeder {
public:
    Feeder() = delete;
    Feeder(const FeederSetting& feeder_setting);
    ~Feeder() = default;

    bool consumeFrame(cv::Mat& frame, int64_t& frame_index);
    void produceFrames();
    MatBuffer& getMatBuffer() { return mFeederBuffer; }
    void start() { mIsFeederRunning.store(true, std::memory_order_relaxed); }
    void stop() { mIsFeederRunning.store(false, std::memory_order_relaxed); }

private:
    void produceFramesInternal(cv::VideoCapture& cap, int delay_ms);
    void produceFramesInternalDummy();

    FeederSetting mFeederSetting;
    MatBuffer mFeederBuffer;
    std::atomic<bool> mIsFeederRunning{true};
    cv::VideoCapture mCap;
    bool mDelayOn;
    double mVideoFps = 30.0;
    std::chrono::steady_clock::time_point mVideoClock{};
    bool mLoggedOpenFailure = false;
    bool mLoggedInitialReadFailure = false;
};
#endif
