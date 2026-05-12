#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>

#include "demo/model.h"
#include "opencv2/opencv.hpp"
#include "qbruntime/qbruntime.h"

namespace {
struct LetterboxParams {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;
};

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float iou_xyxy(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    float denom = area_a + area_b - inter;
    if (denom <= 0.0f) return 0.0f;
    return inter / denom;
}

void letterbox_bgr(const cv::Mat& src, cv::Mat& dst, cv::Mat& resized, int dst_w,
                   int dst_h, LetterboxParams& p) {
    p.src_w = src.cols;
    p.src_h = src.rows;
    p.dst_w = dst_w;
    p.dst_h = dst_h;

    if (src.empty() || src.cols <= 0 || src.rows <= 0 || dst_w <= 0 || dst_h <= 0) {
        dst = cv::Mat::zeros(std::max(1, dst_h), std::max(1, dst_w), CV_8UC3);
        p.scale = 1.0f;
        p.pad_x = 0;
        p.pad_y = 0;
        return;
    }

    float sx = (float)dst_w / (float)src.cols;
    float sy = (float)dst_h / (float)src.rows;
    p.scale = std::min(sx, sy);

    int new_w = std::max(1, (int)std::round((float)src.cols * p.scale));
    int new_h = std::max(1, (int)std::round((float)src.rows * p.scale));
    p.pad_x = (dst_w - new_w) / 2;
    p.pad_y = (dst_h - new_h) / 2;

    dst.create(dst_h, dst_w, CV_8UC3);
    dst.setTo(cv::Scalar(114, 114, 114));

    resized.create(new_h, new_w, CV_8UC3);
    cv::resize(src, resized, cv::Size(new_w, new_h));
    resized.copyTo(dst(cv::Rect(p.pad_x, p.pad_y, new_w, new_h)));
}

inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

std::array<float, 4> undo_letterbox_xyxy(const std::array<float, 4>& b,
                                         const LetterboxParams& p) {
    if (p.scale <= 0.0f || p.src_w <= 0 || p.src_h <= 0) return b;

    float x1 = (b[0] - (float)p.pad_x) / p.scale;
    float y1 = (b[1] - (float)p.pad_y) / p.scale;
    float x2 = (b[2] - (float)p.pad_x) / p.scale;
    float y2 = (b[3] - (float)p.pad_y) / p.scale;

    x1 = clampf(x1, 0.0f, (float)p.src_w);
    y1 = clampf(y1, 0.0f, (float)p.src_h);
    x2 = clampf(x2, 0.0f, (float)p.src_w);
    y2 = clampf(y2, 0.0f, (float)p.src_h);
    return {x1, y1, x2, y2};
}

}  // namespace

cv::Mat Model::inferenceYolo11(cv::Mat frame, cv::Size size, int stream_id) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    static thread_local Benchmarker bm_prep;
    static thread_local Benchmarker bm_infer;
    static thread_local Benchmarker bm_post;
    static thread_local Benchmarker bm_draw;
    static thread_local int perf_count = 0;
    const bool perf_print = ((perf_count++ % 60) == 0);

#ifdef DEMO_DEBUG_LOG
    static int dbg_count = 0;
    const bool dbg_print = ((dbg_count++ % 60) == 0);
    if (dbg_print) {
        std::cout << "[YOLO11] frame=" << frame.cols << "x" << frame.rows
                  << " display=" << size.width << "x" << size.height << " input=" << w
                  << "x" << h << "x" << c << std::endl;
        auto outs = mModel->getOutputBufferInfo();
        std::cout << "[YOLO11] output_count=" << outs.size() << std::endl;
        for (size_t i = 0; i < outs.size(); i++) {
            std::cout << "  - out[" << i << "] " << outs[i].original_width << "x"
                      << outs[i].original_height << "x" << outs[i].original_channel
                      << std::endl;
        }
    }
#endif

    bm_prep.start();
    auto& scratch = getYolo11Scratch(stream_id);
    LetterboxParams lb;

    const size_t input_size = static_cast<size_t>(w) * h * c;
    if (scratch.w != w || scratch.h != h || scratch.c != c ||
        scratch.input_size != input_size) {
        scratch.input_img = mobilint::NDArray<float>({1, h, w, c}, sc);
        scratch.w = w;
        scratch.h = h;
        scratch.c = c;
        scratch.input_size = input_size;
    }

    letterbox_bgr(frame, scratch.resized_frame, scratch.resized_for_letterbox, w, h, lb);

    scratch.rgb.create(h, w, CV_8UC3);
    cv::cvtColor(scratch.resized_frame, scratch.rgb, cv::COLOR_BGR2RGB);

    cv::Mat input_mat(h, w, CV_32FC3, scratch.input_img.data());
    scratch.rgb.convertTo(input_mat, CV_32FC3, 1.0f / 255.0f);
    bm_prep.end();

    bm_infer.start();
    auto result = mModel->infer({scratch.input_img}, sc);
    bm_infer.end();

    if (!sc) {
#ifdef DEMO_DEBUG_LOG
        std::cout << "[YOLO11] infer failed" << std::endl;
#endif
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO11] infer ok, npu_outs=" << result.size();
        if (!result.empty()) std::cout << " out0_size=" << result[0].size();
        std::cout << std::endl;
        if (!result.empty()) {
            size_t nshow = std::min<size_t>(12, result[0].size());
            std::cout << "[YOLO11] out0_head:";
            for (size_t i = 0; i < nshow; i++) std::cout << " " << result[0][i];
            std::cout << std::endl;
        }
    }
#endif

    bm_post.start();
    static thread_local std::vector<std::array<float, 4>> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> labels;
    static thread_local std::vector<std::vector<float>> extras;

    boxes.clear();
    scores.clear();
    labels.clear();
    extras.clear();

    uint64_t ticket =
        mPost->enqueue(scratch.resized_frame, result, boxes, scores, labels, extras);
    mPost->receive(ticket);

    for (size_t i = 0; i < boxes.size(); i++) {
        boxes[i] = undo_letterbox_xyxy(boxes[i], lb);
    }

    static thread_local std::vector<float> ema_scores;
    ema_scores.assign(scores.size(), 0.0f);
    {
        std::lock_guard<std::mutex> lk(mScoreTracksMutex);
        auto& score_tracks = mScoreTracksByWorker[stream_id];
        static thread_local std::vector<int> track_used;
        track_used.assign(score_tracks.size(), 0);

        for (size_t i = 0; i < scores.size(); i++) {
            int best_track = -1;
            float best_iou = 0.0f;

            for (size_t j = 0; j < score_tracks.size(); j++) {
                if (track_used[j]) continue;
                if (score_tracks[j].label != labels[i]) continue;

                float iou = iou_xyxy(score_tracks[j].box, boxes[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_track = (int)j;
                }
            }

            if (best_track >= 0 && best_iou >= mScoreEmaMatchIou) {
                auto& track = score_tracks[best_track];
                track.score_ema = mScoreEmaAlpha * scores[i] +
                                  (1.0f - mScoreEmaAlpha) * track.score_ema;

                track.box = boxes[i];
                track.label = labels[i];
                track.missed = 0;

                track_used[best_track] = 1;
                ema_scores[i] = track.score_ema;
            } else {
                ScoreEmaTrack new_track;

                new_track.box = boxes[i];
                new_track.label = labels[i];
                new_track.score_ema = scores[i];
                new_track.missed = 0;

                score_tracks.push_back(new_track);
                track_used.push_back(1);
                ema_scores[i] = new_track.score_ema;
            }
        }

        for (size_t j = 0; j < score_tracks.size(); j++) {
            if (!track_used[j]) score_tracks[j].missed++;
        }

        score_tracks.erase(std::remove_if(score_tracks.begin(), score_tracks.end(),
                                          [this](const ScoreEmaTrack& t) {
                                              return t.missed > mScoreEmaMaxMissed;
                                          }),
                           score_tracks.end());
    }
    bm_post.end();

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO11] det_count=" << boxes.size() << std::endl;
        for (size_t i = 0; i < std::min<size_t>(boxes.size(), 5); i++) {
            std::cout << "  det[" << i << "] label=" << labels[i]
                      << " score=" << scores[i] << " ema=" << ema_scores[i] << " box=("
                      << boxes[i][0] << "," << boxes[i][1] << "," << boxes[i][2] << ","
                      << boxes[i][3] << ")" << std::endl;
        }
    }
#endif

    bm_draw.start();
    scratch.result_frame.create(size.height, size.width, frame.type());
    cv::resize(frame, scratch.result_frame, size);
    const cv::Mat clean_detection_frame = scratch.result_frame.clone();

    float sx = (float)size.width / (float)frame.cols;
    float sy = (float)size.height / (float)frame.rows;
    bool detected = false;
    std::vector<DetectionBox> latest_detections;
    for (int i = 0; i < (int)boxes.size(); i++) {
        int x1 = (int)(boxes[i][0] * sx);
        int y1 = (int)(boxes[i][1] * sy);
        int x2 = (int)(boxes[i][2] * sx);
        int y2 = (int)(boxes[i][3] * sy);

        x1 = std::max(0, std::min(x1, size.width - 1));
        y1 = std::max(0, std::min(y1, size.height - 1));
        x2 = std::max(0, std::min(x2, size.width - 1));
        y2 = std::max(0, std::min(y2, size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        if (ema_scores[i] < mScoreEmaDisplayThres) continue;

        detected = true;
        cv::Scalar clr =
            (labels[i] == 0) ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 255);
        cv::rectangle(scratch.result_frame, cv::Point(x1, y1), cv::Point(x2, y2), clr,
                      2);
        latest_detections.push_back(
            {cv::Rect(x1, y1, x2 - x1, y2 - y1), ema_scores[i], labels[i],
             getLabelName(labels[i])});
        char conf_text[32];
        std::snprintf(conf_text, sizeof(conf_text), "%.2f", ema_scores[i]);
        int text_y = std::max(14, y1 - 6);
        cv::putText(scratch.result_frame, conf_text, cv::Point(x1, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv::LINE_AA);
    }

    if (detected) {
        cv::rectangle(scratch.result_frame, cv::Point(0, 0),
                      cv::Point(size.width - 1, size.height - 1), cv::Scalar(0, 0, 255),
                      3);
    }
    updateLatestDetectionSnapshot(stream_id, clean_detection_frame, latest_detections);
    bm_draw.end();

    if (perf_print) {
        // printf("[YOLO11-PERF] prep=%.3fms infer=%.3fms post=%.3fms draw=%.3fms\n",
        //        bm_prep.getSec() * 1000.0f, bm_infer.getSec() * 1000.0f,
        //        bm_post.getSec() * 1000.0f, bm_draw.getSec() * 1000.0f);
        // fflush(stdout);
    }

    return scratch.result_frame;
}

cv::Mat Model::inferenceYolo26(cv::Mat frame, cv::Size size, int stream_id) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    static thread_local Benchmarker bm_prep;
    static thread_local Benchmarker bm_infer;
    static thread_local Benchmarker bm_post;
    static thread_local Benchmarker bm_draw;
    static thread_local int perf_count = 0;
    const bool perf_print = ((perf_count++ % 60) == 0);

#ifdef DEMO_DEBUG_LOG
    static int dbg_count = 0;
    const bool dbg_print = ((dbg_count++ % 60) == 0);
    if (dbg_print) {
        std::cout << "[YOLO26] frame=" << frame.cols << "x" << frame.rows
                  << " display=" << size.width << "x" << size.height << " input=" << w
                  << "x" << h << "x" << c << std::endl;
        auto outs = mModel->getOutputBufferInfo();
        std::cout << "[YOLO26] output_count=" << outs.size() << std::endl;
        for (size_t i = 0; i < outs.size(); i++) {
            std::cout << "  - out[" << i << "] " << outs[i].original_width << "x"
                      << outs[i].original_height << "x" << outs[i].original_channel
                      << std::endl;
        }
    }
#endif

    bm_prep.start();
    auto& scratch = getYolo26Scratch(stream_id);
    LetterboxParams lb;

    const size_t input_size = static_cast<size_t>(w) * h * c;
    if (scratch.w != w || scratch.h != h || scratch.c != c ||
        scratch.input_size != input_size) {
        scratch.input_img = mobilint::NDArray<unsigned char>({1, h, w, c}, sc);
        scratch.w = w;
        scratch.h = h;
        scratch.c = c;
        scratch.input_size = input_size;
    }

    letterbox_bgr(frame, scratch.resized_frame, scratch.resized_for_letterbox, w, h, lb);

    cv::Mat input_mat(h, w, CV_8UC3, scratch.input_img.data());
    cv::cvtColor(scratch.resized_frame, input_mat, cv::COLOR_BGR2RGB);
    bm_prep.end();

    bm_infer.start();
    auto result = mModel->infer({scratch.input_img}, sc);
    bm_infer.end();

    if (!sc) {
#ifdef DEMO_DEBUG_LOG
        std::cout << "[YOLO26] infer failed" << std::endl;
#endif
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO26] infer ok, npu_outs=" << result.size();
        if (!result.empty()) std::cout << " out0_size=" << result[0].size();
        std::cout << std::endl;
        if (!result.empty()) {
            size_t nshow = std::min<size_t>(12, result[0].size());
            std::cout << "[YOLO26] out0_head:";
            for (size_t i = 0; i < nshow; i++) std::cout << " " << result[0][i];
            std::cout << std::endl;
        }
    }
#endif

    bm_post.start();
    static thread_local std::vector<std::array<float, 4>> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> labels;

    boxes.clear();
    scores.clear();
    labels.clear();

    const int nc = 2;
    const float conf_thres = 0.25f;
    const float inv_conf_thres = std::log(conf_thres / (1.0f - conf_thres));
    const size_t max_det = 300;

    struct OutputView {
        int idx;
        int grid_h;
        int grid_w;
        int ch;
        size_t elem_count;
    };

    std::vector<OutputView> box_outs;
    std::vector<OutputView> cls_outs;
    auto out_infos = mModel->getOutputBufferInfo();
    size_t nout = std::min(result.size(), out_infos.size());
    box_outs.reserve(nout);
    cls_outs.reserve(nout);

    for (size_t i = 0; i < nout; i++) {
        int ow = out_infos[i].original_width;
        int oh = out_infos[i].original_height;
        int oc = out_infos[i].original_channel;
        if (ow <= 0 || oh <= 0 || oc <= 0) continue;

        size_t expect_size = (size_t)ow * (size_t)oh * (size_t)oc;
        if (expect_size != result[i].size()) continue;

        OutputView out = {(int)i, oh, ow, oc, expect_size};
        if (oc == 4) {
            box_outs.push_back(out);
        } else if (oc == nc) {
            cls_outs.push_back(out);
        }
    }

    std::sort(box_outs.begin(), box_outs.end(),
              [](const OutputView& a, const OutputView& b) {
                  return a.elem_count > b.elem_count;
              });
    std::sort(cls_outs.begin(), cls_outs.end(),
              [](const OutputView& a, const OutputView& b) {
                  return a.elem_count > b.elem_count;
              });

    std::vector<std::array<float, 4>> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;
    cand_boxes.reserve(8400);
    cand_scores.reserve(8400);
    cand_labels.reserve(8400);

    const int total_anchors =
        (h / 8) * (w / 8) + (h / 16) * (w / 16) + (h / 32) * (w / 32);
    static thread_local int flat_cache_w = -1;
    static thread_local int flat_cache_h = -1;
    static thread_local std::vector<float> flat_anchor_x;
    static thread_local std::vector<float> flat_anchor_y;
    static thread_local std::vector<float> flat_stride_x;
    static thread_local std::vector<float> flat_stride_y;
    if (flat_cache_w != w || flat_cache_h != h) {
        flat_anchor_x.clear();
        flat_anchor_y.clear();
        flat_stride_x.clear();
        flat_stride_y.clear();
        flat_anchor_x.reserve(total_anchors);
        flat_anchor_y.reserve(total_anchors);
        flat_stride_x.reserve(total_anchors);
        flat_stride_y.reserve(total_anchors);
        for (int stride : {8, 16, 32}) {
            int gh = h / stride;
            int gw = w / stride;
            for (int gy = 0; gy < gh; gy++) {
                for (int gx = 0; gx < gw; gx++) {
                    flat_anchor_x.push_back((float)gx + 0.5f);
                    flat_anchor_y.push_back((float)gy + 0.5f);
                    flat_stride_x.push_back((float)stride);
                    flat_stride_y.push_back((float)stride);
                }
            }
        }
        flat_cache_w = w;
        flat_cache_h = h;
    }

    size_t npairs = std::min(box_outs.size(), cls_outs.size());
    for (size_t p = 0; p < npairs; p++) {
        const auto& box_view = box_outs[p];
        const auto& cls_view = cls_outs[p];
        if (box_view.grid_h != cls_view.grid_h || box_view.grid_w != cls_view.grid_w) {
            continue;
        }

        const auto& box = result[box_view.idx];
        const auto& cls = result[cls_view.idx];
        int grid_h = box_view.grid_h;
        int grid_w = box_view.grid_w;
        int ncell = grid_h * grid_w;
        bool is_flat = ((grid_h == 1 || grid_w == 1) && ncell == total_anchors &&
                        flat_anchor_x.size() == (size_t)total_anchors);
        float stride_x = is_flat ? 0.0f : (float)w / (float)grid_w;
        float stride_y = is_flat ? 0.0f : (float)h / (float)grid_h;

        for (int cell = 0; cell < ncell; cell++) {
            int cls_base = cell * nc;
            int label = 0;
            float best_logit = cls[cls_base];
            for (int cidx = 1; cidx < nc; cidx++) {
                float logit = cls[cls_base + cidx];
                if (logit > best_logit) {
                    best_logit = logit;
                    label = cidx;
                }
            }
            if (best_logit <= inv_conf_thres) continue;

            float conf = sigmoid(best_logit);
            if (conf < conf_thres) continue;

            size_t box_base = (size_t)cell * 4;
            float l = box[box_base + 0];
            float t = box[box_base + 1];
            float r = box[box_base + 2];
            float b = box[box_base + 3];

            float ax = 0.0f;
            float ay = 0.0f;
            float sx = 0.0f;
            float sy = 0.0f;
            if (is_flat) {
                ax = flat_anchor_x[cell];
                ay = flat_anchor_y[cell];
                sx = flat_stride_x[cell];
                sy = flat_stride_y[cell];
            } else {
                int gx = cell % grid_w;
                int gy = cell / grid_w;
                ax = (float)gx + 0.5f;
                ay = (float)gy + 0.5f;
                sx = stride_x;
                sy = stride_y;
            }

            float x1 = (ax - l) * sx;
            float y1 = (ay - t) * sy;
            float x2 = (ax + r) * sx;
            float y2 = (ay + b) * sy;

            x1 = std::max(0.0f, std::min(x1, (float)w));
            y1 = std::max(0.0f, std::min(y1, (float)h));
            x2 = std::max(0.0f, std::min(x2, (float)w));
            y2 = std::max(0.0f, std::min(y2, (float)h));
            if (x2 <= x1 || y2 <= y1) continue;

            cand_boxes.push_back({x1, y1, x2, y2});
            cand_scores.push_back(conf);
            cand_labels.push_back(label);
        }
    }

    std::vector<int> order(cand_scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return cand_scores[a] > cand_scores[b]; });

    size_t keep_n = std::min(max_det, order.size());
    boxes.reserve(keep_n);
    scores.reserve(keep_n);
    labels.reserve(keep_n);
    for (size_t k = 0; k < keep_n; k++) {
        int idx = order[k];
        boxes.push_back(cand_boxes[idx]);
        scores.push_back(cand_scores[idx]);
        labels.push_back(cand_labels[idx]);
    }

    for (size_t i = 0; i < boxes.size(); i++) {
        boxes[i] = undo_letterbox_xyxy(boxes[i], lb);
    }
    bm_post.end();

    static thread_local std::vector<float> ema_scores;
    ema_scores.assign(scores.size(), 0.0f);
    {
        std::lock_guard<std::mutex> lk(mScoreTracksMutex);
        auto& score_tracks = mScoreTracksByWorker[stream_id];
        static thread_local std::vector<int> track_used;
        track_used.assign(score_tracks.size(), 0);

        for (size_t i = 0; i < scores.size(); i++) {
            int best_track = -1;
            float best_iou = 0.0f;

            for (size_t j = 0; j < score_tracks.size(); j++) {
                if (track_used[j]) continue;
                if (score_tracks[j].label != labels[i]) continue;

                float iou = iou_xyxy(score_tracks[j].box, boxes[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_track = (int)j;
                }
            }

            if (best_track >= 0 && best_iou >= mScoreEmaMatchIou) {
                auto& track = score_tracks[best_track];
                track.score_ema = mScoreEmaAlpha * scores[i] +
                                  (1.0f - mScoreEmaAlpha) * track.score_ema;

                track.box = boxes[i];
                track.label = labels[i];
                track.missed = 0;

                track_used[best_track] = 1;
                ema_scores[i] = track.score_ema;
            } else {
                ScoreEmaTrack new_track;

                new_track.box = boxes[i];
                new_track.label = labels[i];
                new_track.score_ema = scores[i];
                new_track.missed = 0;

                score_tracks.push_back(new_track);
                track_used.push_back(1);
                ema_scores[i] = new_track.score_ema;
            }
        }

        for (size_t j = 0; j < score_tracks.size(); j++) {
            if (!track_used[j]) score_tracks[j].missed++;
        }

        score_tracks.erase(std::remove_if(score_tracks.begin(), score_tracks.end(),
                                          [this](const ScoreEmaTrack& t) {
                                              return t.missed > mScoreEmaMaxMissed;
                                          }),
                           score_tracks.end());
    }

#ifdef DEMO_DEBUG_LOG
    if (dbg_print) {
        std::cout << "[YOLO26] det_count=" << boxes.size() << std::endl;
        for (size_t i = 0; i < std::min<size_t>(boxes.size(), 5); i++) {
            std::cout << "  det[" << i << "] label=" << labels[i]
                      << " score=" << scores[i] << " ema=" << ema_scores[i] << " box=("
                      << boxes[i][0] << "," << boxes[i][1] << "," << boxes[i][2] << ","
                      << boxes[i][3] << ")" << std::endl;
        }
    }
#endif

    bm_draw.start();
    scratch.result_frame.create(size.height, size.width, frame.type());
    cv::resize(frame, scratch.result_frame, size);
    const cv::Mat clean_detection_frame = scratch.result_frame.clone();

    float sx = (float)size.width / (float)frame.cols;
    float sy = (float)size.height / (float)frame.rows;
    bool detected = false;
    std::vector<DetectionBox> latest_detections;
    for (int i = 0; i < (int)boxes.size(); i++) {
        int x1 = (int)(boxes[i][0] * sx);
        int y1 = (int)(boxes[i][1] * sy);
        int x2 = (int)(boxes[i][2] * sx);
        int y2 = (int)(boxes[i][3] * sy);

        x1 = std::max(0, std::min(x1, size.width - 1));
        y1 = std::max(0, std::min(y1, size.height - 1));
        x2 = std::max(0, std::min(x2, size.width - 1));
        y2 = std::max(0, std::min(y2, size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        if (ema_scores[i] < mScoreEmaDisplayThres) continue;

        detected = true;
        cv::Scalar clr =
            (labels[i] == 0) ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 255);
        cv::rectangle(scratch.result_frame, cv::Point(x1, y1), cv::Point(x2, y2), clr,
                      2);
        latest_detections.push_back(
            {cv::Rect(x1, y1, x2 - x1, y2 - y1), ema_scores[i], labels[i],
             getLabelName(labels[i])});
        char conf_text[32];
        std::snprintf(conf_text, sizeof(conf_text), "%.2f", ema_scores[i]);
        int text_y = std::max(14, y1 - 6);
        cv::putText(scratch.result_frame, conf_text, cv::Point(x1, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv::LINE_AA);
    }

    if (detected) {
        cv::rectangle(scratch.result_frame, cv::Point(0, 0),
                      cv::Point(size.width - 1, size.height - 1), cv::Scalar(0, 0, 255),
                      3);
    }
    updateLatestDetectionSnapshot(stream_id, clean_detection_frame, latest_detections);
    bm_draw.end();

    if (perf_print) {
        // printf("[YOLO26-PERF] prep=%.3fms infer=%.3fms post=%.3fms draw=%.3fms\n",
        //        bm_prep.getSec() * 1000.0f, bm_infer.getSec() * 1000.0f,
        //        bm_post.getSec() * 1000.0f, bm_draw.getSec() * 1000.0f);
        // fflush(stdout);
    }

    return scratch.result_frame;
}
