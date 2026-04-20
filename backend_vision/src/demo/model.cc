#include "demo/model.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "demo/post_yolo11_det.h"
#include "opencv2/opencv.hpp"

Model::Model(const ModelSetting& model_setting, mobilint::Accelerator& acc) {
    mobilint::StatusCode sc;
    mobilint::ModelConfig mc;

    if (model_setting.is_num_core) {
        mc.setSingleCoreMode(model_setting.num_core);
    } else {
        mc.setSingleCoreMode(model_setting.core_id);
    }

    mModel = mobilint::Model::create(model_setting.mxq_path, mc, sc);
    mModel->launch(acc);

    // clang-format off
    switch (model_setting.model_type) {
    case ModelType::YOLO11      : initYolo11(); break;
    case ModelType::YOLO26      : initYolo26(); break;
    }
    // clang-format on
}

Model::~Model() { mModel->dispose(); }

void Model::initYolo11() {
    float conf_thres = 0.05f;
    float iou_thres = 0.45f;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int nc = 2;

    mPost = std::make_unique<YOLO11DetPostProcessor>(nc, h, w, conf_thres, iou_thres);
    mInference = &Model::inferenceYolo11;
}

void Model::initYolo26() {
    mInference = &Model::inferenceYolo26;
}

Model::Yolo11WorkerScratch& Model::getYolo11Scratch(int stream_id) {
    std::lock_guard<std::mutex> lk(mYolo11ScratchMutex);
    return mYolo11ScratchByWorker[stream_id];
}

Model::Yolo26WorkerScratch& Model::getYolo26Scratch(int stream_id) {
    std::lock_guard<std::mutex> lk(mYolo26ScratchMutex);
    return mYolo26ScratchByWorker[stream_id];
}

void Model::work(Model* model, int worker_index, SizeState* size_state,
                 ItemQueue* item_queue, MatBuffer* feeder_buffer) {
    Benchmarker benchmarker;

    cv::Mat frame, result;
    cv::Size result_size;

    int64_t frame_index = 0;
    while (true) {
        // workerReceive 함수에서 Mat()를 받으면 worker가 죽은 것으로 간주하고 화면을
        // clear한다.
        auto ssc = size_state->checkUpdate(result_size);
        if (ssc != SizeState::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        auto msc = feeder_buffer->get(frame, frame_index);
        if (msc != MatBuffer::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        benchmarker.start();
#ifdef USE_SLEEP_DRIVER
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cv::resize(frame, result, result_size);
#else
        result = model->inference(frame, result_size, worker_index);
#endif
        benchmarker.end();

        item_queue->push({worker_index, result, benchmarker.getFPS(),
                          benchmarker.getTimeSinceCreated(), benchmarker.getCount()});
    }
}

cv::Mat Model::inference(cv::Mat frame, cv::Size size, int stream_id) {
    return (this->*mInference)(frame, size, stream_id);
}

WorkerDetectionSnapshot Model::getLatestDetectionSnapshot(int stream_id) const {
    std::lock_guard<std::mutex> lk(mLatestDetectionMutex);
    auto it = mLatestDetectionByWorker.find(stream_id);
    if (it == mLatestDetectionByWorker.end()) {
        WorkerDetectionSnapshot empty;
        empty.worker_index = stream_id;
        return empty;
    }
    return it->second;
}

void Model::updateLatestDetectionSnapshot(int stream_id, const cv::Mat& image,
                                          const std::vector<DetectionBox>& detections) {
    WorkerDetectionSnapshot snapshot;
    snapshot.worker_index = stream_id;
    snapshot.has_detection = !detections.empty();
    snapshot.frame_size = image.size();
    snapshot.detections = detections;
    if (!image.empty()) {
        snapshot.image = image.clone();
    }

    std::lock_guard<std::mutex> lk(mLatestDetectionMutex);
    mLatestDetectionByWorker[stream_id] = std::move(snapshot);
}

std::string Model::getLabelName(int label) {
    switch (label) {
    case 0:
        return "gun";
    case 1:
        return "knife";
    default:
        return "unknown";
    }
}
