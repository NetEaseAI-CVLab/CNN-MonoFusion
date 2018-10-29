/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include "DepthPredictCommunicator.h"
#include "KinectDataProvider.h"
//#include "OctomapViewer.h"

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<direct.h>
#include<chrono>
#include<sstream>
#include<array>
#include<io.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui.hpp>
#include "DatasetConfigure.h"

#include<System.h>

#define WITH_SLAM
#define IMAGE_NUMS 2000

#define SERVER_PORT 6666
//#define SERVER_IP "10.200.129.201"
#define SERVER_IP "10.200.129.187"
//#define SERVER_IP "127.0.0.1"

using namespace std;


void save_pose(const string &filename, cv::Mat Tcw, int trackingState, bool isKeyframe);
bool dirExists(const std::string &dirName_in);


int main(int argc, char **argv)
{
    std::string dataset_configure_path = "../Examples/rgbd_monodepth/dataset_configure.yaml";
    DatasetConfigure dataset_conf(dataset_configure_path);

    cout << "\n--------------" << endl;
    cout << "Start & Initialize ORB ..." << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(dataset_conf.voc_path, dataset_conf.setting_path, ORB_SLAM2::System::RGBD, true);
    DepthPredictCommunicator depth_predict_com(dataset_conf.setting_path, dataset_conf.is_use_dataset_depth);

    cv::waitKey(20);

    cv::FileStorage fs(dataset_conf.setting_path, cv::FileStorage::READ);
    cv::Mat K(cv::Matx33f(
        fs["Camera.fx"], 0, fs["Camera.cx"],
        0, fs["Camera.fy"], fs["Camera.cy"],
        0, 0, 1));
    float bf = fs["Camera.bf"];
    int m_img_width = fs["Camera.width"];
    int m_img_height = fs["Camera.height"];
    int save_image_enable = fs["Camera.Save_Image"];
    int cv_wait_delay = fs["Camera.cv_wait_delay"];
    fs.release();
    
    // kinect initial
    KinectDataProvider kinect_reader;
    if(!dataset_conf.is_use_dataset_image)
    {
        std::cout << "Initialize Kinect!" << std::endl;
        if(!kinect_reader.initialize())
        {
            std::cout << "kinect connect failed!" << std::endl;
            return -1;
        }
        kinect_reader.start(cv::Size(960, 540));
        std::cout << "Kinect init done!" << std::endl;
    }


    // mkdir name in current time
    std::string data_path = "../dataset/";
    time_t t = time(0);
    tm *now = localtime(&t);
    std::stringstream save_path_ss;
    save_path_ss << data_path << now->tm_year + 1900 << "-" << now->tm_mon + 1 << "-" << now->tm_mday << "-"
        << now->tm_hour << "-" << now->tm_min << "-" << now->tm_sec << "/";
    std::string save_path = save_path_ss.str();
    if(save_image_enable)
    {
        //std::string mkdir_cmd = "mkdir " + save_path;
        if(!dirExists(save_path))
        {
            mkdir(save_path.c_str());
        }
    }


    // Main loop
    cv::Mat imRGB, imD,imD_float;
    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32FC1);
    cv::Mat Twc = cv::Mat::eye(4, 4, CV_32FC1);
    int ni = 0;
    if (dataset_conf.is_use_dataset_image)
    {
        ni = dataset_conf.image_begin;
        std::cout << "Use dataset image and begin at: " << ni << std::endl;
    }
    while(1)
    {

        stringstream ss;
        ss << setfill('0') << setw(dataset_conf.image_name_zfill) << ni;

        std::string frame_id_str = ss.str();
        //std::cout << "Image : " << frame_id_str << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
        //std::cout << "Image : " << frame_id_str << std::endl;
        std::string pose_save_path = dataset_conf.pose_save_dir + frame_id_str + "-pose.txt";

        // Read image and depthmap from file
        std::string label_path = dataset_conf.label_dir + frame_id_str + dataset_conf.label_extend_name;
        
        if(dataset_conf.is_use_dataset_image)
        {

            std::string color_path = dataset_conf.rgb_dir + frame_id_str + dataset_conf.rgb_extend_name;
              
            cv::Mat imRGB_origin;
            if((_access(color_path.c_str(), 0)) != -1)
            {
                imRGB_origin = cv::imread(color_path, CV_LOAD_IMAGE_COLOR);
                if(imRGB_origin.empty())
                {
                    std::cout << "\nFailed to load image at: " << color_path << endl;
                    std::cout << "Press esc to exit! " << endl;
                    char key = cv::waitKey();
                    if(key == 27) break;
                }
                cv::imshow("img_color", imRGB_origin);
                cv::resize(imRGB_origin, imRGB, cv::Size(m_img_width, m_img_height), 0, 0, cv::INTER_NEAREST);
            }
            else
            {
                std::cout << "\n" << color_path << " not exist!" << endl;
                std::cout << "Press esc to exit! " << endl;
                char key = cv::waitKey();
                if(key == 27) break;
            }
            
        }
        else
        {
            
            bool frame_read_ok = false; 
            cv::Mat kinect_img, kinect_depth;
            while(!frame_read_ok)
            {
                frame_read_ok = kinect_reader.get_aligned_frame(kinect_img, kinect_depth);
                if(!frame_read_ok)
                {
                    //std::cout << "Kinect no data!" << std::endl;
                    char key = cv::waitKey(5);
                    if(key == 27) break;
                }
            }


            if(save_image_enable)
            {
                std::string image_save_path = save_path + frame_id_str + "_color.png";
                std::string depth_save_path = save_path + frame_id_str + "_depth.png";
                cv::imwrite(image_save_path, kinect_img);
                cv::imwrite(depth_save_path, kinect_depth);
            }

            cv::Mat kinect_img_bgr;
            cv::cvtColor(kinect_img, kinect_img_bgr, CV_BGRA2BGR);
            cv::resize(kinect_img_bgr, imRGB, cv::Size(m_img_width, m_img_height), 0, 0, cv::INTER_NEAREST);
            //cv::imshow("img_color", imRGB);
        }

        if(dataset_conf.is_use_dataset_depth)
        {
            std::string depth_path = dataset_conf.depth_dir + frame_id_str + dataset_conf.depth_extend_name;
            if((_access(depth_path.c_str(), 0)) != -1)
            {
                imD = cv::imread(depth_path, CV_LOAD_IMAGE_UNCHANGED);
                if(imD.empty())
                {
                    std::cout << "\nFailed to load depth at: " << depth_path << endl;
                    std::cout << "Press esc to exit! " << endl;
                    char key = cv::waitKey();
                    if(key == 27) break;
                }
            }
            else
            {
                std::cout << "\n" << depth_path << " not exist!" << endl;
                std::cout << "Press esc to exit! " << endl;
                char key = cv::waitKey();
                if(key == 27) break;
            }
        }
        else
        {
            bool recv_ok = false;
            while(!recv_ok)
            {
                recv_ok = depth_predict_com.getNetworkPredictDepth(imRGB, imD);

                if(!recv_ok)
                {
                    char key = cv::waitKey(5);
                    if(key == 27) break;
                }
            }
        }
        

        int rgb_width = imRGB.cols;
        int rgb_height = imRGB.rows;

        cv::Mat imD_resize;
        cv::resize(imD, imD_resize, cv::Size(rgb_width, rgb_height), 0,0, cv::INTER_NEAREST);
        imD_resize.convertTo(imD_float, CV_32F);
            
        imD_float = imD_float * dataset_conf.depth_factor;

        #ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        #else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
        #endif

        //// Pass the images to the SLAM system
        bool isKeyframe = false;
        int trackingState = 2;
        int frame_id = ni;
        Tcw = SLAM.TrackRGBD(imRGB, imD_float, time(0), isKeyframe, trackingState, frame_id);

        if((trackingState == 2) && ni > dataset_conf.densemap_skip_frames)
        {

            SLAM.densemap_frame_enqueue(imRGB, imD_float, K, Tcw, bf, frame_id, isKeyframe);
        }
        #ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        #else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
        #endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        ni = ni + 1;

        cv::imshow("kinect rgb", imRGB);
        cv::imshow("cnn predict depth", imD_resize);
        
        char key = cv::waitKey(cv_wait_delay);
        if(key == 27) break;

        if (dataset_conf.is_use_dataset_image && frame_id > dataset_conf.image_end)
        {
            char key = cv::waitKey();
            if(key == 27) break;
        }
    }


    SLAM.densemap_request_finish();

    std::cout << "\ndone!" << std::endl;

    // create a new thread to show the octomap
    /*cout << "Initializing Octomap..." << endl;
    OctomapViewer octviewer(setting_path);
    vector<Vec3f> points;
    vector<Vec3b> colors;
    dense_map.get_dense_points(points, colors);
    std::thread th_octviewer(&OctomapViewer::run, &octviewer, points, colors);*/
    //octviewer.run(points, colors);

    //system("pause");

    //map_viewer.request_finish();
    //octviewer.request_finish();
    //th_map_viewer.join();
    //th_octviewer.join();

    // Stop all threads
    #ifdef WITH_SLAM
    SLAM.Shutdown();
    #endif

    // Save camera trajectory
    #ifdef WITH_SLAM
    //SLAM.SaveTrajectoryKITTI("CameraTrajectory_kitti.txt");
    //SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_kitti_stereo_orbslam.txt");
    #endif

    kinect_reader.stop();

    return 0;
}

void save_pose(const string &filename, cv::Mat Tcw, int trackingState, bool isKeyframe)
{
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    int frame_state = isKeyframe ? 1 : 0;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);
    f << "trackingState " << trackingState << endl;
    f << "isKeyframe " << frame_state << endl;
    f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << "\n" <<
        Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << "\n" <<
        Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;

    f.close();
}

bool dirExists(const std::string &dirName_in)
{
    int ftyp = _access(dirName_in.c_str(), 0);
    if(ftyp == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}
