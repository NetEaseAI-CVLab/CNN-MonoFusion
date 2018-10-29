#pragma once

#include<opencv2\core.hpp>
#include<list>
#include<mutex>
#include<vector>
#include "OrbKeyFrame.h"
#include <iostream>
#include <condition_variable>    // std::condition_variable
#include <KeyFrame.h>

#include "LabelClasses.h"



#define CVLIBS_FUSION 0
#define INIT_UNCERTAINTY 1.0f
#define DIST_THR 0.1f
#define WIDTH_BODER_LENGTH 70
#define TOP_HEIGHT_BODER_LENGTH 20
#define BOTTOM_HEIGHT_BODER_LENGTH 2
#define MAX_REF_NEIGHS 5

struct DenseMapParams
{
    DenseMapParams(float dist_thr, float init_uncertainty, float max_uncertainty,float _maxdepth, float _stable_thr,float _mindepth,float _focal_scale,int max_fusion_weight_, float _delay_loop_thr) :
        point_dist_thr(dist_thr),
        point_init_uncertainty(init_uncertainty),
        max_uncertainty_limit(max_uncertainty),
        point_stable_thr(_stable_thr),
        max_depth(_maxdepth),
        min_depth(_mindepth),
        focal_scale(_focal_scale),
        max_fusion_weight(max_fusion_weight_),
        stable_delay_loop_thr(_delay_loop_thr)
    {
    }

    DenseMapParams() :
        point_dist_thr(0.1f),
        point_init_uncertainty(0.1f),
        max_uncertainty_limit(1.0f), 
        max_depth(4.0f),
        min_depth(0.1f),
        point_stable_thr(0.001f),
        focal_scale(1.0f),
        stable_delay_loop_thr(0.001f),
        max_fusion_weight(100)
    {
        #ifdef WITH_LABEL
        label_fusion_thr = 255;
        #endif
    }

    float point_dist_thr;
    float point_init_uncertainty;
    float max_uncertainty_limit;
    int max_fusion_weight;
    float max_depth;
    float focal_scale;
    float min_depth;
    float point_stable_thr;
    float stable_delay_loop_thr;
    float uncertainty_weight;

    #ifdef WITH_LABEL
    int label_fusion_thr;
    DenseMapParams(float dist_thr, float init_uncertainty, float max_uncertainty, float _maxdepth, float _stable_thr, int _label_thr) :
        point_dist_thr(dist_thr),
        point_init_uncertainty(init_uncertainty),
        max_uncertainty_limit(max_uncertainty),
        max_depth(_maxdepth),
        point_stable_thr(_stable_thr),
        label_fusion_thr(_label_thr)
    {
    }

    #endif
};

class DenseMap
{
public:
	DenseMap(std::string setting_file, std::string classes_txt = " ");
	void add_points_from_depth_map(OrbKeyFrame &curr_keyframe);
	void clear_points();
    uchar mergeSimilarClass(uchar label_predict);
	void get_current_Tcw(cv::Mat& current_Tcw);
    void set_current_Tcw(cv::Mat & Tcw);
	void get_current_depth_map(cv::Mat& depth);
	void get_dense_points(std::vector<cv::Vec3f>& points, 
        std::vector<cv::Vec3b>& colors, bool draw_label, bool draw_single_frame, bool draw_color);
    bool reprojection_3d2d(const cv::Vec3f &pointw, const cv::Mat &K, const cv::Mat &Tcw, const int cols, const int rows, FeatureUV &uvcoord);
    cv::Vec3f projection_2d3d(const FeatureUV &uvcoord, float Zc, const cv::Mat &K, const cv::Mat &Tcw);
    void cvlibs_greedy_fusion(OrbKeyFrame &ref_keyframe, OrbKeyFrame &curr_keyframe);
    void cvlibs_greedy_fusion_covisible(std::vector<OrbKeyFrame*> &ref_neighs, OrbKeyFrame &curr_keyframe);
    void cvlibs_greedy_fusion_framewise_covisible(std::vector<OrbKeyFrame*> &ref_neighs, OrbKeyFrame &curr_keyframe);
    void cvlibs_greedy_fusion_framewise(OrbKeyFrame &ref_keyframe, OrbKeyFrame &curr_frame);
    cv::Mat calc_transfrom_matrix_2d2d(cv::Mat &Tc1w, cv::Mat &Tc2w);
    cv::Vec3b label_colorful(uchar label);

    void cvlibs_fusion_and_propagate_incweight(cv::Vec3f &ref_pointcloud, cv::Vec3f &curr_pointcloud,
        float ref_weight, float curr_weight, MapPoint &point);
    void cvlibs_fusion_and_propagate(cv::Vec3f &ref_pointcloud, cv::Vec3f &curr_pointcloud, float ref_weight, float curr_weight, MapPoint &point);
    void label_fusion(uchar curr_label, MapPoint &point);

    //control system status
    std::mutex system_running_mtx;
    std::condition_variable system_running_cv;
    bool system_running;

    bool get_system_running_status()
    {
        bool status = system_running;
        return status;
    }

    void set_system_running_status(bool status)
    {
        std::unique_lock <std::mutex> lck(system_running_mtx);
        if(system_running != status)
        {
            system_running = status;
            if(status)
            {
                std::cout << "System Running!" << std::endl;
                system_running_cv.notify_all();
            }
            else
            {
                std::cout << "System Stopping!" << std::endl;
            }
        }
        lck.unlock();
    }

protected:
	std::mutex m_depth_mutex;
	std::mutex m_points_mutex;
    std::mutex m_current_keyframe_mutex;


    // record current keyframe
    OrbKeyFrame m_current_keyframe;
	cv::Mat m_current_Tcw;
	cv::Mat m_current_depth;
    cv::Mat m_current_image;
	std::vector<MapPoint> m_points;
	std::vector<cv::Vec3b> m_colors;
    
    int invisible_point_size;
    int point_nums;

    #ifdef WITH_LABEL
    cv::Mat m_current_label;
    #endif

    DenseMapParams densemap_params;
};
