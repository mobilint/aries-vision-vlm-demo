#ifndef DEMO_INCLUDE_DEMO_H_
#define DEMO_INCLUDE_DEMO_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <thread>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"
#include "qbruntime/model.h"
#include "qbruntime/qbruntime.h"

class Model;
class Feeder;
class MjpegServer;

struct HttpStreamConfig {
    bool enabled = true;
    int port = 8081;
};

class Demo {
public:
    Demo(bool headless = false, HttpStreamConfig http_stream_config = {});
    void run();
    cv::Mat getDisplayFrameCopy() const;
    std::string getDetectionSnapshotJson() const;
    std::string getLayoutMetadataJson() const;

private:
    void startWorker(int index);
    void stopWorker(int index);
    void startFeeder(int index);
    void stopFeeder(int index);

    void startWorkerAll();
    void stopWorkerAll();
    void startFeederAll();
    void stopFeederAll();
    void ensureWorkerEnabledStorage(size_t n);

    void startProcessing();
    void stopProcessing();

    int getWorkerIndex(int x, int y);
    static void onMouseEvent(int event, int x, int y, int flags, void* userdata);

    void modelInferLoop(size_t model_index);
    bool tryProcessWorker(int worker_index);
    float smoothDisplayFPS(int worker_index, float instant_fps);

    void initWindow();
    void initLayout(std::string path);
    void initModels(std::string path);
    void initFeeders(std::string path);
    void checkLayoutValidation();
    void display();
    cv::Size getCanvasSize() const;
    void resizeDisplayCanvas(const cv::Size& size);
    void updateWorkerDetectionSnapshot(int worker_index, int feeder_index, int model_index,
                                       const WorkerDetectionSnapshot& snapshot);

    void toggleDisplayFPSMode();
    void toggleDisplayTimeMode();
    void toggleScreenSize();
    bool keyHandler(int key);

    void setMode(int mode_index);
    void setMode1();
    void setMode2();
    void setMode3();
    void applyMode(int mode_index, const std::string& layout_path,
                   const std::string& model_path, bool reload_model, int splash_index);
    void showSplash(int splash_index);

    std::vector<FeederSetting> loadFeederSettingYAML(const std::string& path,
                                                     bool generate_default = false);
    std::vector<ModelSetting> loadModelSettingYAML(const std::string& path,
                                                   bool generate_default = false);
    LayoutSetting loadLayoutSettingYAML(const std::string& path,
                                        bool generate_default = false);

    const std::string WINDOW_NAME = "Mobilint Inference Demo";

    mutable std::mutex mDisplayMutex;
    mutable std::mutex mDetectionSnapshotMutex;
    cv::Mat mDisplay;      // front buffer
    cv::Mat mDisplayBase;  // static layout/background
    Benchmarker mBenchmarker;

    bool mDisplayFPSMode;
    bool mDisplayTimeMode;

    std::vector<cv::Mat> mSplashes;
    int mModeIndex;

    std::vector<FeederSetting> mFeederSetting;  // FeederSetting.yaml에서 읽은 정보 저장
    std::vector<ModelSetting> mModelSetting;    // ModelSetting.yaml에서 읽은 정보 저장
    LayoutSetting mLayoutSetting;               // LayoutSetting.yaml에서 읽은 정보 저장
    std::vector<uint8_t> mWorkerLayoutValid;
    std::vector<std::vector<int>> mWorkersByModel;

    std::map<int, std::unique_ptr<mobilint::Accelerator>> mAccs;

    std::vector<std::unique_ptr<Model>> mModels;
    std::vector<std::unique_ptr<Feeder>> mFeeders;

    std::atomic<bool> mProcessingOn{false};

    std::vector<std::thread> mInferThreads;
    std::vector<std::thread> mFeederThreads;
    ItemQueue mRenderQueue;

    std::unique_ptr<std::atomic<uint32_t>[]> mModelWorkerCursor;
    size_t mModelWorkerCursorSize = 0;

    std::unique_ptr<std::atomic<uint8_t>[]> mWorkerEnabled;
    size_t mWorkerEnabledSize = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> mWorkerInFlight;
    size_t mWorkerInFlightSize = 0;
    std::vector<int64_t> mWorkerLastFrameIndex;
    std::vector<Benchmarker> mWorkerInferBench;
    std::vector<Benchmarker> mWorkerDisplayFPSBench;
    std::vector<std::deque<float>> mWorkerDisplayFPSHistory;
    std::vector<float> mWorkerDisplayFPSSum;
    std::vector<WorkerDetectionSnapshot> mWorkerDetectionSnapshots;
    static constexpr size_t DISPLAY_FPS_AVG_WINDOW = 10;

    bool mHeadless = false;
    HttpStreamConfig mHttpStreamConfig;
    std::unique_ptr<MjpegServer> mMjpegServer;
};

#endif
