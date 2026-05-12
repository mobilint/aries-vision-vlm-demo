#include <fstream>
#include <map>
#include <vector>

#include "demo/define.h"
#include "demo/demo.h"
#include "yaml-cpp/yaml.h"

namespace {
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
    model_setting_node[0]["mxq_path"] = "../assets/mxq/yolo26s-weapon_uint8_input.mxq";
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

    std::map<std::string, mobilint::Cluster> cluster_map;
    cluster_map["Cluster0"] = mobilint::Cluster::Cluster0;
    cluster_map["Cluster1"] = mobilint::Cluster::Cluster1;

    std::map<std::string, mobilint::Core> core_map;
    core_map["Core0"] = mobilint::Core::Core0;
    core_map["Core1"] = mobilint::Core::Core1;
    core_map["Core2"] = mobilint::Core::Core2;
    core_map["Core3"] = mobilint::Core::Core3;

    std::vector<ModelSetting> model_settings;

    YAML::Node ms_node = YAML::LoadFile(path);
    for (int i = 0; i < ms_node.size(); i++) {
        ModelSetting ms;
        ms.model_type = model_type_map[ms_node[i]["model_type"].as<std::string>()];
        ms.mxq_path = ms_node[i]["mxq_path"].as<std::string>();
        ms.dev_no = ms_node[i]["dev_no"].as<int>();

        YAML::Node core_id_node = ms_node[i]["core_id"];
        for (int j = 0; j < core_id_node.size(); j++) {
            mobilint::Cluster cluster =
                cluster_map[core_id_node[j]["cluster"].as<std::string>()];
            mobilint::Core core = core_map[core_id_node[j]["core"].as<std::string>()];
            ms.core_id.push_back({cluster, core});
        }

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
            ms.num_core = ms.core_id.size();
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
