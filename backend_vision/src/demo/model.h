#ifndef DEMO_INCLUDE_MODEL_H_
#define DEMO_INCLUDE_MODEL_H_

#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"
#include "post.h"
#include "qbruntime/model.h"

class Model {
public:
    Model() = delete;
    Model(const ModelSetting& model_setting, mobilint::Accelerator& acc);
    ~Model();

    static void work(Model* model, int worker_index, SizeState* size_state,
                     ItemQueue* item_queue, MatBuffer* feeder_buffer);

    cv::Mat inference(cv::Mat frame, cv::Size size, int stream_id);
    WorkerDetectionSnapshot getLatestDetectionSnapshot(int stream_id) const;

private:
    cv::Mat (Model::*mInference)(cv::Mat, cv::Size, int);

    std::unique_ptr<mobilint::Model> mModel;
    std::unique_ptr<PostProcessor> mPost;

    struct ScoreEmaTrack {
        std::array<float, 4> box;
        int label = -1;
        int missed = 0;
        float score_ema = 0.0f;
    };

    std::mutex mScoreTracksMutex;
    std::unordered_map<int, std::vector<ScoreEmaTrack>> mScoreTracksByWorker;
    mutable std::mutex mLatestDetectionMutex;
    std::unordered_map<int, WorkerDetectionSnapshot> mLatestDetectionByWorker;

    struct Yolo11WorkerScratch {
        int w = 0;
        int h = 0;
        int c = 0;
        size_t input_size = 0;
        mobilint::NDArray<float> input_img;
        cv::Mat resized_frame;
        cv::Mat resized_for_letterbox;
        cv::Mat rgb;
        cv::Mat result_frame;
    };

    struct Yolo26WorkerScratch {
        int w = 0;
        int h = 0;
        int c = 0;
        size_t input_size = 0;
        mobilint::NDArray<unsigned char> input_img;
        cv::Mat resized_frame;
        cv::Mat resized_for_letterbox;
        cv::Mat result_frame;
    };

    std::mutex mYolo11ScratchMutex;
    std::unordered_map<int, Yolo11WorkerScratch> mYolo11ScratchByWorker;
    std::mutex mYolo26ScratchMutex;
    std::unordered_map<int, Yolo26WorkerScratch> mYolo26ScratchByWorker;

    float mScoreEmaAlpha = 0.7f;
    float mScoreEmaMatchIou = 0.3f;
    float mScoreEmaDisplayThres = 0.25f;
    int mScoreEmaMaxMissed = 8;

    void initYolo11();
    void initYolo26();
    Yolo11WorkerScratch& getYolo11Scratch(int stream_id);
    Yolo26WorkerScratch& getYolo26Scratch(int stream_id);
    void updateLatestDetectionSnapshot(int stream_id, const cv::Mat& image,
                                       const std::vector<DetectionBox>& detections);
    static std::string getLabelName(int label);

    cv::Mat inferenceYolo11(cv::Mat frame, cv::Size size, int stream_id);
    cv::Mat inferenceYolo26(cv::Mat frame, cv::Size size, int stream_id);
};
#endif
