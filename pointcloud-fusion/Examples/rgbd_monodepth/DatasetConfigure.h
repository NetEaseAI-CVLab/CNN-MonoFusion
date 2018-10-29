#pragma once
#include<string>
#include<io.h>
#include<opencv2/core/core.hpp>
#include<direct.h>
#include<iostream>

struct DatasetConfigure
{
public:
    DatasetConfigure(std::string dataset_configure_path)
    {
        std::cout << "---------------------------" << std::endl;
        std::cout << "Dataset Configure " << std::endl;

        cv::FileStorage settings(dataset_configure_path, cv::FileStorage::READ);

        sequences = settings["DatasetConf.sequences"];
        voc_path = settings["DatasetConf.voc_path"];
        setting_path = settings["DatasetConf.camera_params_yaml"];;

        pose_save_dir = settings["DatasetConf.pose_save_dir"];
        dataset_type = settings["DatasetConf.type"];
        rgb_extend_name = settings["DatasetConf.rgb_extend_name"];
        depth_extend_name = settings["DatasetConf.depth_extend_name"];
        label_extend_name = settings["DatasetConf.label_extend_name"];

        depth_factor = settings["DatasetConf.depth_factor"];

        image_begin = settings["DatasetConf.image_begin"];
        image_end = settings["DatasetConf.image_end"];
        image_name_zfill = settings["DatasetConf.image_name_zfill"];
        densemap_skip_frames = settings["DatasetConf.densemap_skip_frames"];

        rgb_dir = settings["DatasetConf.rgb_dir"];
        depth_dir = settings["DatasetConf.depth_dir"];
        label_dir = settings["DatasetConf.label_dir"];

        is_use_dataset_image = settings["DatasetConf.USE_DATASET_IMAGE"];
        is_use_dataset_depth = settings["DatasetConf.USE_DATASET_DEPTH"];

        if (is_use_dataset_depth)
        {
            // use dataset depth must use align image_dataset
            is_use_dataset_image = 1;
        }

        if(!dirExists(pose_save_dir)) mkdir(pose_save_dir.c_str());
        pose_save_dir = pose_save_dir + sequences;
        if(!dirExists(pose_save_dir)) mkdir(pose_save_dir.c_str());
        tum_pose_save_dir = pose_save_dir + "tum_format_pose.txt";

        std::cout << "- dataset type: " << dataset_type << std::endl;
        std::cout << "- camera params yaml path: " << setting_path << std::endl;
        std::cout << "- voc path: " << voc_path << std::endl;
        std::cout << "- pose save dir: " << pose_save_dir << std::endl;
        std::cout << "- rgb dir: " << rgb_dir << std::endl;
        std::cout << "- depth dir: " << depth_dir << std::endl;
        std::cout << "- label dir: " << label_dir << std::endl;
        std::cout << "- image name zfill: " << image_name_zfill << std::endl;
        std::cout << "- rgb extend name: " << rgb_extend_name << std::endl;
        std::cout << "- depth extend name: " << depth_extend_name << std::endl;
        std::cout << "- label extend name: " << label_extend_name << std::endl;
        std::cout << "- depth factor: " << depth_factor << std::endl;
        std::cout << "- image begin: " << image_begin << std::endl;
        std::cout << "- image end: " << image_end << std::endl;
        std::cout << "- dense map skip frame nums: " << densemap_skip_frames << std::endl;
        std::cout << "- is use dataset image: " << is_use_dataset_image << std::endl;
        std::cout << "- is use dataset depth: " << is_use_dataset_depth << std::endl;

        settings.release();
    }

    bool dirExists(const std::string& dirName_in)
    {
        int ftyp = _access(dirName_in.c_str(), 0);

        if(0 == ftyp)
            return true;   // this is a directory!  
        else
            return false;    // this is not a directory!  
    }

    std::string voc_path = "E:/orbslam_windows/orbslam_semantic/Examples/Monocular/ORBvoc/ORBvoc.txt";
    std::string setting_path = "E:/orbslam_windows/orbslam_semantic/Examples/Stereo/zed_stereo.yaml";
    std::string pose_save_dir = "F:/study/slam/Dataset/netease/zed_stereo/";
    std::string tum_pose_save_dir = "F:/study/slam/Dataset/netease/zed_stereo/";
    std::string sequences = "2017-8-30-10-19-17/";
    std::string dataset_type = "KINECT";
    std::string depth_dir = "F:/study/slam/Dataset/netease/zed_stereo/";
    std::string label_dir = "F:/study/slam/Dataset/netease/zed_stereo/";
    std::string rgb_dir = "F:/study/slam/Dataset/netease/zed_stereo/";
    std::string rgb_extend_name = "_rgb.png";
    std::string depth_extend_name = "_depth.png";
    std::string label_extend_name = "_label.png";
    int image_name_zfill = 5; //kinect: str(5).zfill(), zed: str(6).zfill()
    float depth_factor = 0.0002;
    int image_begin = 0;
    int image_end = 900;
    int densemap_skip_frames = 20;
    int is_use_dataset_image = 0;
    int is_use_dataset_depth = 0;

    
};
