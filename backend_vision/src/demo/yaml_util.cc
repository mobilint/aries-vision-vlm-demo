#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "demo/define.h"
#include "demo/demo.h"
#include "demo/yaml_util.h"
#include "yaml-cpp/yaml.h"

namespace {
bool FileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}
}  // namespace

std::string ResolveCoreAllocationPath(const std::string& ctor_override) {
    std::vector<std::string> tried;
    if (!ctor_override.empty()) {
        if (FileExists(ctor_override)) {
            return ctor_override;
        }
        tried.push_back("constructor override: " + ctor_override);
    }
    if (const char* env = std::getenv("ARIES_CORE_ALLOCATION_PATH")) {
        if (env[0] != '\0') {
            if (FileExists(env)) {
                return env;
            }
            tried.push_back(std::string("ARIES_CORE_ALLOCATION_PATH: ") + env);
        }
    }
    const std::string default_path = "/etc/aries/core_allocation.yaml";
    if (FileExists(default_path)) {
        return default_path;
    }
    tried.push_back("default: " + default_path);
    // Repo-relative fallback so ./demo run out of backend_vision/build/ works locally.
    // From backend_vision/build/, the repo root is two levels up.
    const std::string fallback = "../../core_allocation.yaml";
    if (FileExists(fallback)) {
        return fallback;
    }
    tried.push_back("repo-relative fallback: " + fallback);

    std::string msg = "core_allocation.yaml not found. Tried: [";
    for (size_t i = 0; i < tried.size(); ++i) {
        if (i > 0) msg += ", ";
        msg += tried[i];
    }
    msg +=
        "]. Set ARIES_CORE_ALLOCATION_PATH or bind-mount to "
        "/etc/aries/core_allocation.yaml.";
    throw std::runtime_error(msg);
}

std::map<std::string, std::vector<CoreId>> LoadVisionCoreAllocation(
    const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    YAML::Node vision_node = root["vision"];
    if (!vision_node || !vision_node.IsMap()) {
        throw std::runtime_error(
            "core_allocation.yaml at " + path + " is missing 'vision' section");
    }

    std::map<std::string, std::vector<CoreId>> result;
    for (auto it = vision_node.begin(); it != vision_node.end(); ++it) {
        std::string category = it->first.as<std::string>();
        YAML::Node cores_node = it->second;
        std::vector<CoreId> cores;
        for (int i = 0; i < static_cast<int>(cores_node.size()); ++i) {
            CoreId c;
            c.cluster = cores_node[i]["cluster"].as<int>();
            c.core = cores_node[i]["core"].as<int>();
            cores.push_back(c);
        }
        result.emplace(std::move(category), std::move(cores));
    }
    return result;
}

namespace {
float clampFloat(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

void applyPipelineConfigYAML(const YAML::Node& node, PipelineConfig& config) {
    if (!node || !node.IsMap()) {
        return;
    }

    YAML::Node labels_node = node["labels"];
    if (labels_node && labels_node.IsSequence()) {
        config.labels.clear();
        for (int i = 0; i < labels_node.size(); i++) {
            config.labels.push_back(labels_node[i].as<std::string>());
        }
    }
    if (node["conf_threshold"]) {
        config.conf_threshold = node["conf_threshold"].as<float>();
    }
    if (node["iou_threshold"]) {
        config.iou_threshold = node["iou_threshold"].as<float>();
    }
    if (node["display_confidence_threshold"]) {
        config.display_confidence_threshold =
            node["display_confidence_threshold"].as<float>();
    }
    if (node["bbox_thickness"]) {
        config.bbox_thickness = node["bbox_thickness"].as<int>();
    }
    if (node["draw_label_text"]) {
        config.draw_label_text = node["draw_label_text"].as<bool>();
    }
    if (node["draw_score_text"]) {
        config.draw_score_text = node["draw_score_text"].as<bool>();
    }
    if (node["draw_confidence_text"]) {
        config.draw_score_text = node["draw_confidence_text"].as<bool>();
    }
    if (node["draw_detection_border"]) {
        config.draw_detection_border = node["draw_detection_border"].as<bool>();
    }
    YAML::Node allowed_node = node["allowed_label_names"];
    if (allowed_node && allowed_node.IsSequence()) {
        config.allowed_label_names.clear();
        for (int i = 0; i < allowed_node.size(); i++) {
            config.allowed_label_names.push_back(allowed_node[i].as<std::string>());
        }
    }
}

void normalizePipelineConfig(PipelineConfig& config) {
    config.conf_threshold = clampFloat(config.conf_threshold, 0.001f, 0.999f);
    config.iou_threshold = clampFloat(config.iou_threshold, 0.001f, 1.0f);
    config.display_confidence_threshold =
        clampFloat(config.display_confidence_threshold, 0.0f, 1.0f);
    config.bbox_thickness = std::max(1, config.bbox_thickness);
}

void generateDefaultFeederSettingYAML(const std::string& path) {
    YAML::Node feeder_setting_node[4];
    feeder_setting_node[0]["feeder_type"] = "CAMERA";
    feeder_setting_node[0]["src_path"] = "0";

    feeder_setting_node[1]["feeder_type"] = "IPCAMERA";
    feeder_setting_node[1]["src_path"] = "rtsp://<ID>:<PW>@<IP>:554/trackID=1";

    feeder_setting_node[2]["feeder_type"] = "YOUTUBE";
    feeder_setting_node[2]["src_path"] = "https://www.youtube.com/watch?v=4MUEJ7w-A9U";

    feeder_setting_node[3]["feeder_type"] = "VIDEO";
    feeder_setting_node[3]["src_path"] = "../assets/video/1.mp4";

    YAML::Node feeder_settings_node;
    feeder_settings_node.push_back(feeder_setting_node[0]);
    feeder_settings_node.push_back(feeder_setting_node[1]);
    feeder_settings_node.push_back(feeder_setting_node[2]);
    feeder_settings_node.push_back(feeder_setting_node[3]);

    std::ofstream fout(path);
    YAML::Emitter emitter(fout);
    emitter << YAML::Comment("RTSP는 다음과 같이 설정합니다.") << YAML::Newline;
    emitter << YAML::Comment("<ID> - IP Camera 계정") << YAML::Newline;
    emitter << YAML::Comment("<PW> - IP Camera 암호") << YAML::Newline;
    emitter << YAML::Comment("<IP> - IP Camera IP") << YAML::Newline << YAML::Newline;

    fout << feeder_settings_node;
    fout.close();
}

void generateDefaultModelSettingYAML(const std::string& path) {
    YAML::Node core_id_node[8];
    core_id_node[0]["cluster"] = "Cluster0";
    core_id_node[0]["core"] = "Core0";
    core_id_node[1]["cluster"] = "Cluster0";
    core_id_node[1]["core"] = "Core1";
    core_id_node[2]["cluster"] = "Cluster0";
    core_id_node[2]["core"] = "Core2";
    core_id_node[3]["cluster"] = "Cluster0";
    core_id_node[3]["core"] = "Core3";

    core_id_node[4]["cluster"] = "Cluster1";
    core_id_node[4]["core"] = "Core0";
    core_id_node[5]["cluster"] = "Cluster1";
    core_id_node[5]["core"] = "Core1";
    core_id_node[6]["cluster"] = "Cluster1";
    core_id_node[6]["core"] = "Core2";
    core_id_node[7]["cluster"] = "Cluster1";
    core_id_node[7]["core"] = "Core3";

    YAML::Node model_setting_node[1];
    model_setting_node[0]["model_type"] = "YOLO26";
    model_setting_node[0]["mxq_path"] = "../assets/mxq/yolo26s-weapon_uint8_input_260513.mxq";
    model_setting_node[0]["labels"].push_back("gun");
    model_setting_node[0]["labels"].push_back("knife");
    model_setting_node[0]["pipeline_config"]["labels"].push_back("gun");
    model_setting_node[0]["pipeline_config"]["labels"].push_back("knife");
    model_setting_node[0]["pipeline_config"]["conf_threshold"] = 0.25f;
    model_setting_node[0]["pipeline_config"]["iou_threshold"] = 0.45f;
    model_setting_node[0]["pipeline_config"]["display_confidence_threshold"] = 0.25f;
    model_setting_node[0]["pipeline_config"]["bbox_thickness"] = 2;
    model_setting_node[0]["pipeline_config"]["draw_label_text"] = true;
    model_setting_node[0]["pipeline_config"]["draw_score_text"] = true;
    model_setting_node[0]["pipeline_config"]["draw_detection_border"] = true;
    model_setting_node[0]["dev_no"] = 0;
    model_setting_node[0]["core_id"].push_back(core_id_node[0]);
    model_setting_node[0]["core_id"].push_back(core_id_node[1]);

    YAML::Node model_settings_node;
    model_settings_node.push_back(model_setting_node[0]);

    std::ofstream fout(path);
    fout << model_settings_node;
    fout.close();
}

void generateDefaultLayoutSettingYAML(const std::string& path) {
    YAML::Node image_layout_node;
    image_layout_node[0]["path"] = "../assets/layout/Top_bnr.png";
    image_layout_node[0]["roi"].SetStyle(YAML::EmitterStyle::Flow);
    image_layout_node[0]["roi"][0] = 649;
    image_layout_node[0]["roi"][1] = 0;
    image_layout_node[0]["roi"][2] = 621;
    image_layout_node[0]["roi"][3] = 200;

    YAML::Node feeder_layout_node;
    int feeder_x = 0;
    int feeder_y = 216;
    int feeder_w = 384;
    int feeder_h = 216;
    for (int i = 0; i < 4; i++) {
        feeder_layout_node[i].SetStyle(YAML::EmitterStyle::Flow);
        feeder_layout_node[i][0] = feeder_x;
        feeder_layout_node[i][1] = feeder_y + i * feeder_h;
        feeder_layout_node[i][2] = feeder_w;
        feeder_layout_node[i][3] = feeder_h;
    }

    YAML::Node worker_layout_node;
    int model_x = 384;
    int model_y = 216;
    int model_w = 384;
    int model_h = 216;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            worker_layout_node[y * 4 + x].SetStyle(YAML::EmitterStyle::Flow);
            worker_layout_node[y * 4 + x]["feeder_index"] = y;
            worker_layout_node[y * 4 + x]["model_index"] = x;
            worker_layout_node[y * 4 + x]["roi"].SetStyle(YAML::EmitterStyle::Flow);
            worker_layout_node[y * 4 + x]["roi"][0] = model_x + x * model_w;
            worker_layout_node[y * 4 + x]["roi"][1] = model_y + y * model_h;
            worker_layout_node[y * 4 + x]["roi"][2] = model_w;
            worker_layout_node[y * 4 + x]["roi"][3] = model_h;
        }
    }

    YAML::Node layout_node;
    layout_node["image_layout"] = image_layout_node;
    layout_node["feeder_layout"] = feeder_layout_node;
    layout_node["worker_layout"] = worker_layout_node;

    std::ofstream fout(path);
    fout << layout_node;
    fout.close();
}
}  // namespace

std::vector<FeederSetting> Demo::loadFeederSettingYAML(const std::string& path,
                                                       bool generate_default) {
    if (generate_default) {
        generateDefaultFeederSettingYAML(path);
    }
    std::map<std::string, FeederType> feeder_type_map;
    feeder_type_map["CAMERA"] = FeederType::CAMERA;
    feeder_type_map["IPCAMERA"] = FeederType::IPCAMERA;
    feeder_type_map["YOUTUBE"] = FeederType::YOUTUBE;
    feeder_type_map["VIDEO"] = FeederType::VIDEO;

    std::vector<FeederSetting> feeder_settings;

    YAML::Node fs_node = YAML::LoadFile(path);
    for (int i = 0; i < fs_node.size(); i++) {
        FeederSetting fs;
        fs.feeder_type = feeder_type_map[fs_node[i]["feeder_type"].as<std::string>()];
        fs.src_path = fs_node[i]["src_path"].as<std::string>();

        feeder_settings.push_back(fs);
    }
    return feeder_settings;
}

std::vector<ModelSetting> Demo::loadModelSettingYAML(const std::string& path,
                                                     bool generate_default) {
    if (generate_default) {
        generateDefaultModelSettingYAML(path);
    }
    std::map<std::string, ModelType> model_type_map;
    model_type_map["YOLO11"] = ModelType::YOLO11;
    model_type_map["YOLO26"] = ModelType::YOLO26;

    std::vector<ModelSetting> model_settings;

    YAML::Node ms_node = YAML::LoadFile(path);
    for (int i = 0; i < ms_node.size(); i++) {
        ModelSetting ms;
        ms.model_type = model_type_map[ms_node[i]["model_type"].as<std::string>()];
        ms.mxq_path = ms_node[i]["mxq_path"].as<std::string>();
        if (ms_node[i]["category"]) {
            ms.category = ms_node[i]["category"].as<std::string>();
        }
        YAML::Node labels_node = ms_node[i]["labels"];
        if (labels_node && labels_node.IsSequence()) {
            for (int j = 0; j < labels_node.size(); j++) {
                ms.labels.push_back(labels_node[j].as<std::string>());
            }
        }
        ms.pipeline_config.labels = ms.labels;
        applyPipelineConfigYAML(ms_node[i], ms.pipeline_config);
        if (ms_node[i]["pipeline_config"]) {
            applyPipelineConfigYAML(ms_node[i]["pipeline_config"], ms.pipeline_config);
        }
        normalizePipelineConfig(ms.pipeline_config);
        ms.labels = ms.pipeline_config.labels;
        ms.dev_no = ms_node[i]["dev_no"].as<int>();

        // core_id is deprecated in favour of core_allocation.yaml; silently ignore.
        ms.core_id.clear();

        YAML::Node num_core_node = ms_node[i]["num_core"];
        if (num_core_node) {
            ms.num_core = num_core_node.as<int>();
            ms.is_num_core = true;
            if (ms.num_core <= 0) {
                std::cerr << "[WARNING] Model index " << i << ": num_core is "
                          << ms.num_core <<". num_core will be 0" << std::endl;
                ms.num_core = 0;
            }
        } else {
            ms.num_core = 0;
            ms.is_num_core = false;
        }
        model_settings.push_back(ms);
    }
    return model_settings;
}

LayoutSetting Demo::loadLayoutSettingYAML(const std::string& path,
                                          bool generate_default) {
    if (generate_default) {
        generateDefaultLayoutSettingYAML(path);
    }

    LayoutSetting layout_setting;

    YAML::Node layout_node = YAML::LoadFile(path);

    YAML::Node image_layout_node = layout_node["image_layout"];
    for (int i = 0; i < image_layout_node.size(); i++) {
        std::string path = image_layout_node[i]["path"].as<std::string>();
        int x = image_layout_node[i]["roi"][0].as<int>();
        int y = image_layout_node[i]["roi"][1].as<int>();
        int w = image_layout_node[i]["roi"][2].as<int>();
        int h = image_layout_node[i]["roi"][3].as<int>();
        cv::Mat img = cv::imread(path);
        cv::resize(img, img, {w, h});

        ImageLayout image_layout = {img, {x, y, w, h}};
        layout_setting.image_layout.push_back(image_layout);
    }

    YAML::Node feeder_layout_node = layout_node["feeder_layout"];
    for (int i = 0; i < feeder_layout_node.size(); i++) {
        int x = feeder_layout_node[i][0].as<int>();
        int y = feeder_layout_node[i][1].as<int>();
        int w = feeder_layout_node[i][2].as<int>();
        int h = feeder_layout_node[i][3].as<int>();
        FeederLayout feeder_layout = {{x, y, w, h}};
        layout_setting.feeder_layout.push_back(feeder_layout);
    }

    YAML::Node worker_layout_node = layout_node["worker_layout"];
    for (int i = 0; i < worker_layout_node.size(); i++) {
        int feeder_index = worker_layout_node[i]["feeder_index"].as<int>();
        int model_index = worker_layout_node[i]["model_index"].as<int>();
        int x = worker_layout_node[i]["roi"][0].as<int>();
        int y = worker_layout_node[i]["roi"][1].as<int>();
        int w = worker_layout_node[i]["roi"][2].as<int>();
        int h = worker_layout_node[i]["roi"][3].as<int>();
        WorkerLayout worker_layout = {feeder_index, model_index, {x, y, w, h}};
        layout_setting.worker_layout.push_back(worker_layout);
    }

    return layout_setting;
}
