#ifndef DEMO_INCLUDE_MJPEG_SERVER_H_
#define DEMO_INCLUDE_MJPEG_SERVER_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "opencv2/opencv.hpp"

class MjpegServer {
public:
    using FrameProvider = std::function<cv::Mat()>;
    using JsonProvider = std::function<std::string()>;

    MjpegServer(int port, FrameProvider frame_provider, JsonProvider json_provider = {},
                JsonProvider layout_provider = {});
    ~MjpegServer();

    bool start();
    void stop();

private:
    void acceptLoop();
    void handleClient(std::intptr_t client_fd);

    int mPort;
    FrameProvider mFrameProvider;
    JsonProvider mJsonProvider;
    JsonProvider mLayoutProvider;
    std::intptr_t mServerFd = -1;
    std::atomic<bool> mRunning{false};
    std::thread mAcceptThread;
};

#endif
