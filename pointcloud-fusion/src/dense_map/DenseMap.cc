#include "DenseMap.h"
#include<opencv2/imgproc.hpp>
#include <iostream>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<chrono>

using namespace std;
using namespace cv;

DenseMap::DenseMap(std::string setting_file, std::string classes_txt)
{
    point_nums = 0;
    invisible_point_size = 0;
    system_running = true;

    cv::FileStorage settings(setting_file, cv::FileStorage::READ);

    densemap_params.point_dist_thr = settings["DenseMap.PointDistThr"];
    densemap_params.max_uncertainty_limit = settings["DenseMap.MaxUncertaintyLimit"];
    densemap_params.point_init_uncertainty = settings["DenseMap.PointInitUncertainty"];
    densemap_params.max_depth = settings["DenseMap.MaxDepth"];
    densemap_params.min_depth = settings["DenseMap.MinDepth"];
    densemap_params.point_stable_thr = settings["DenseMap.PointStableThr"];
    densemap_params.stable_delay_loop_thr = settings["DenseMap.StableDelayLoopThr"];
    densemap_params.uncertainty_weight = settings["DenseMap.UncertaintyWeight"];
    densemap_params.focal_scale = settings["Camera.focal_scale"];
    densemap_params.max_fusion_weight = settings["DenseMap.MaxFusionWeight"];

    #ifdef WITH_LABEL
    densemap_params.label_fusion_thr = settings["DenseMap.LabelFusionThr"];
    #endif

    std::cout << "------------------------\nDenseMap Params" << std::endl;

    std::cout << "- dist thr: " << densemap_params.point_dist_thr << "\n"
        << "- max uncertainty: " << densemap_params.max_uncertainty_limit << "\n"
        << "- init uncertainty: " << densemap_params.point_init_uncertainty << "\n"
        << "- max depth: " << densemap_params.max_depth << "\n"
        << "- min depth: " << densemap_params.min_depth << "\n"
        << "- stable thr: " << densemap_params.point_stable_thr << "\n"
        << "- uncertainty weight: " << densemap_params.uncertainty_weight << "\n"
        << "- max fusion weight: " << densemap_params.max_fusion_weight << "\n"
        #ifndef WITH_LABEL
        << std::endl;
        #else
        << "- label fusion thr: " << densemap_params.label_fusion_thr << "\n" << std::endl;
        #endif
    settings.release();
}

void DenseMap::add_points_from_depth_map(OrbKeyFrame &curr_keyframe)
{	

    if(curr_keyframe.depth.empty() || curr_keyframe.image.empty()) return;
    CV_Assert(curr_keyframe.depth.size() == curr_keyframe.image.size());

    cv::Mat intrincs = curr_keyframe.K;
    float fx = intrincs.at<float>(0, 0);
    float fy = intrincs.at<float>(1, 1);
    float cx = intrincs.at<float>(0, 2);
    float cy = intrincs.at<float>(1, 2);
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    float inv_cx = -cx * inv_fx;
    float inv_cy = -cy * inv_fy;

    unique_lock<mutex> depth_locker(m_depth_mutex, defer_lock);
    unique_lock<mutex> points_locker(m_points_mutex, defer_lock);


    depth_locker.lock();
    m_current_image = curr_keyframe.image;
    m_current_Tcw = curr_keyframe.Tcw.clone();	// deep copy
    m_current_depth = curr_keyframe.depth;		// shallow copy

    #ifdef WITH_LABEL
    m_current_label = curr_keyframe.label;
    uchar * ptr_label = m_current_label.ptr<uchar>();
    #endif

    depth_locker.unlock();

    //GaussianBlur(m_current_image, m_current_image, Size(5, 5), 0);

    int width = m_current_depth.cols;
    int height = m_current_depth.rows;
    float* ptr_depth = m_current_depth.ptr<float>();
    Vec3b* ptr_color = m_current_image.ptr<Vec3b>();


    int step = 1;
    int stride = step * m_current_depth.cols;

    ptr_depth += stride*TOP_HEIGHT_BODER_LENGTH;
    ptr_color += stride*TOP_HEIGHT_BODER_LENGTH;

    Matx33f Rcw = m_current_Tcw.rowRange(0, 3).colRange(0, 3);
    Vec3f tcw = m_current_Tcw.rowRange(0, 3).col(3);
	for (size_t v = TOP_HEIGHT_BODER_LENGTH; v < height- BOTTOM_HEIGHT_BODER_LENGTH; v += step)
	{
		for (size_t u = WIDTH_BODER_LENGTH; u < width- WIDTH_BODER_LENGTH; u += step)
		{
			float Z = ptr_depth[u];
			Vec3b c = ptr_color[u];
			if (Z <= densemap_params.min_depth || Z >= densemap_params.max_depth) continue;
            if(Z != Z) continue;

            FeatureUV curr_uvcoord(u, v);
            cv::Vec3f P = projection_2d3d(curr_uvcoord, Z, intrincs, m_current_Tcw);

			points_locker.lock();
            int pointIdx = m_points.size();
            MapPoint map_point;
            map_point.idx = pointIdx;
            map_point.point = P;
            map_point.isStable = false;
            map_point.uncertainty = densemap_params.point_init_uncertainty;

            #ifdef WITH_LABEL
            map_point.label = mergeSimilarClass(ptr_label[u]);
            map_point.label_confidence = 0;
            map_point.label_color = label_colorful(map_point.label);
            #endif

            m_colors.emplace_back(c[2], c[1], c[0]);
            m_points.emplace_back(map_point);
	
			
			points_locker.unlock();
          
		}
		ptr_depth += stride;
		ptr_color += stride;

        #ifdef WITH_LABEL
        ptr_label += stride;
        #endif
	}

    unique_lock<mutex> curr_keyframe_locker(m_current_keyframe_mutex, defer_lock);
    curr_keyframe_locker.lock();
    m_current_keyframe = curr_keyframe;
    curr_keyframe_locker.unlock();

    //std::cout << "points size: " << m_points.size() << std::endl;
}


cv::Vec3b DenseMap::label_colorful(uchar label)
{
    cv::Vec3b color;
    int r_mod = (label_name::sofa + 1) / 2;
    int g_mod = (label_name::sofa + 2) / 5;
    int b_mod = (label_name::sofa + 2) / 3;
    
    color[0] = (uchar)((label % r_mod + 1) * 30 + 40);
    color[1] = (uchar)(((label + 2) % g_mod + 1) * 60 + 30);
    color[2] = (uchar)(((label + 2) % b_mod + 1) * 30 + 20);
    return color;
}

void DenseMap::cvlibs_greedy_fusion(OrbKeyFrame &ref_keyframe, OrbKeyFrame &curr_keyframe)
{

    if(curr_keyframe.depth.empty() || curr_keyframe.image.empty()) return;
    CV_Assert(curr_keyframe.depth.size() == curr_keyframe.image.size());

    cv::Mat intrincs = curr_keyframe.K;

    unique_lock<mutex> depth_locker(m_depth_mutex, defer_lock);
    unique_lock<mutex> points_locker(m_points_mutex, defer_lock);

    depth_locker.lock();
    m_current_image = curr_keyframe.image;
    m_current_Tcw = curr_keyframe.Tcw.clone();	// deep copy
    m_current_depth = curr_keyframe.depth;		// shallow copy
    cv::Mat ref_Tcw = ref_keyframe.Tcw.clone();	// deep copy

    #ifdef WITH_LABEL
    m_current_label = curr_keyframe.label;
    uchar *ptr_label = m_current_label.ptr<uchar>();
    #endif

    cv::Mat prev_depth = ref_keyframe.depth;
    cv::Mat prev_uncertainty_map = ref_keyframe.uncertainty_map;

    depth_locker.unlock();

    int width = m_current_depth.cols;
    int height = m_current_depth.rows;
    float* ptr_depth = m_current_depth.ptr<float>();
    Vec3b* ptr_color = m_current_image.ptr<Vec3b>();

    int step = 1;
    int stride = step * m_current_depth.cols;

    ptr_depth += stride*TOP_HEIGHT_BODER_LENGTH;
    ptr_color += stride*TOP_HEIGHT_BODER_LENGTH;

    Matx33f Rcw = m_current_Tcw.rowRange(0, 3).colRange(0, 3);
    Vec3f tcw = m_current_Tcw.rowRange(0, 3).col(3);

    cv::Matx33f ref_Rcw = ref_Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Vec3f ref_tcw = ref_Tcw.rowRange(0, 3).col(3).clone();
    float fx = intrincs.at<float>(0, 0);
    float fy = intrincs.at<float>(1, 1);
    float cx = intrincs.at<float>(0, 2);
    float cy = intrincs.at<float>(1, 2);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    //v --> row, height
    for(size_t v = TOP_HEIGHT_BODER_LENGTH; v < (height- BOTTOM_HEIGHT_BODER_LENGTH); v += step)
    {
        //u --> col, width
        for(size_t u = WIDTH_BODER_LENGTH; u < (width- WIDTH_BODER_LENGTH); u += step)
        {
            float Z = ptr_depth[u];
            Vec3b c = ptr_color[u];
            
            if(Z <= densemap_params.min_depth || Z >= densemap_params.max_depth) continue;
            if(Z != Z) continue;

            // project curr_frame to world coord
            FeatureUV curr_uvcoord(u, v);
            float x = float(curr_uvcoord.u);
            float y = float(curr_uvcoord.v);
            float Xw = Z*(x - cx) / fx;
            float Yw = Z*(y - cy) / fy;
            cv::Vec3f P(Xw, Yw, Z);
            P = Rcw.t() * P - Rcw.t()*tcw;

            // project world coord to ref_keyframe coord
            FeatureUV ref_uvcoord;
            bool isInImagePlane = true;
            cv::Vec3f pointc = ref_Rcw * P + ref_tcw;
            float ref_X = pointc[0] * fx + pointc[2] * cx;
            float ref_Y = pointc[1] * fy + pointc[2] * cy;
            float ref_Z = pointc[2];
            float ref_u = ref_X / ref_Z;
            float ref_v = ref_Y / ref_Z;
            if(ref_u<WIDTH_BODER_LENGTH || ref_u>(width - WIDTH_BODER_LENGTH)) isInImagePlane = false;
            if(ref_v<TOP_HEIGHT_BODER_LENGTH || ref_v>(height - BOTTOM_HEIGHT_BODER_LENGTH)) isInImagePlane = false;
            int ref_iu = round(ref_u);
            int ref_iv = round(ref_v);
            ref_uvcoord.u = ref_iu;
            ref_uvcoord.v = ref_iv;

            int pointIdx = 0;

            bool isUVCreatedPoint = ref_keyframe.getMapPointIdx(ref_uvcoord, pointIdx);

            #ifdef CVLIBS_FUSION
            if(isInImagePlane && isUVCreatedPoint && pointIdx < m_points.size())
            {
                //greedy fusion
                MapPoint ref_map_point = m_points[pointIdx];

                float max_weight = densemap_params.max_uncertainty_limit;
                cv::Vec3f ref_pointcloud = ref_map_point.point;
                float ref_weight = ref_map_point.uncertainty;

                cv::Vec3f dist = ref_pointcloud - P;
                float dist_len = dist.dot(dist);
                float curr_weight = (dist_len<max_weight) ? dist_len:max_weight;
                points_locker.lock();
                float fusion_weight = ref_weight;
                if(dist_len < densemap_params.point_dist_thr)
                {
                    bool last_stable_status = m_points[pointIdx].isStable;
                    //cvlibs_fusion_and_propagate(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);                      
                    cvlibs_fusion_and_propagate_incweight(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);
                    bool fusion_stable_status = m_points[pointIdx].isStable;
                    if(last_stable_status == false && fusion_stable_status == true)
                    {
                        cv::Vec3b color(c[2], c[1], c[0]);
                        m_colors[pointIdx] = color;
                    }

                    // update label
                    #ifdef WITH_LABEL
                    uchar curr_label = mergeSimilarClass(ptr_label[u]);
                    label_fusion(curr_label, m_points[pointIdx]);
                    #endif
                }

                // add observes
                curr_keyframe.addMapPoint(curr_uvcoord, pointIdx);

                points_locker.unlock();
            }
            else
            {
                pointIdx = m_points.size();

                // not exist, then create new pointcloud 
                points_locker.lock();

                MapPoint map_point;
                map_point.idx = pointIdx;
                map_point.point = P;
                map_point.isStable = false;
                map_point.uncertainty = densemap_params.point_init_uncertainty;

                #ifdef WITH_LABEL
                map_point.label = mergeSimilarClass(ptr_label[u]);
                map_point.label_confidence = 0;
                map_point.label_color = label_colorful(map_point.label);
                #endif

                m_points.emplace_back(map_point);
                m_colors.emplace_back(c[2], c[1], c[0]);

                curr_keyframe.addMapPoint(curr_uvcoord, pointIdx);
                points_locker.unlock();
            }
            #else
            if(isInImagePlane && isUVCreatedPoint)
            {
                //greedy fusion
                float ref_weight = update_uncertainty_curr;
                float curr_weight = update_uncertainty_ref;
                cv::Vec3f pointcloud = m_points[pointIdx];
                cv::Vec3f dist = pointcloud - P;
            }
            else
            {
                pointIdx = m_points.size();
                //std::cout << "point size" << pointIdx << std::endl;
                // not exist, then create new pointcloud 
                points_locker.lock();
                m_points.emplace_back(P);

                // transport rgb mat 
                m_colors.emplace_back(c[2], c[1], c[0]);
                points_locker.unlock();

                FeatureUV curr_uvcoord(i, j);
                curr_keyframe.addMapPoint(curr_uvcoord, pointIdx);
            }
            #endif
        }

       
        ptr_depth += stride;
        ptr_color += stride;
        #ifdef WITH_LABEL
        ptr_label += stride;
        #endif
    }
    
    unique_lock<mutex> curr_keyframe_locker(m_current_keyframe_mutex, defer_lock);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "keyframe id:" << curr_keyframe.frame_idx << ", cost time: " << ttrack << std::endl;

    curr_keyframe_locker.lock();
    m_current_keyframe = curr_keyframe;
    curr_keyframe_locker.unlock();

}

void DenseMap::cvlibs_greedy_fusion_covisible(std::vector<OrbKeyFrame*> &ref_neighs, OrbKeyFrame &curr_keyframe)
{

    if(curr_keyframe.depth.empty() || curr_keyframe.image.empty()) return;
    CV_Assert(curr_keyframe.depth.size() == curr_keyframe.image.size());

    cv::Mat intrincs = curr_keyframe.K;

    unique_lock<mutex> depth_locker(m_depth_mutex, defer_lock);
    unique_lock<mutex> points_locker(m_points_mutex, defer_lock);

    depth_locker.lock();
    m_current_image = curr_keyframe.image;
    m_current_Tcw = curr_keyframe.Tcw.clone();	// deep copy
    m_current_depth = curr_keyframe.depth;		// shallow copy

    #ifdef WITH_LABEL
    m_current_label = curr_keyframe.label;
    uchar *ptr_label = m_current_label.ptr<uchar>();
    #endif
    depth_locker.unlock();

    int width = m_current_depth.cols;
    int height = m_current_depth.rows;
    float* ptr_depth = m_current_depth.ptr<float>();
    Vec3b* ptr_color = m_current_image.ptr<Vec3b>();

    int step = 1;
    int stride = step * m_current_depth.cols;


    ptr_depth += stride*TOP_HEIGHT_BODER_LENGTH;
    ptr_color += stride*TOP_HEIGHT_BODER_LENGTH;

    Matx33f Rcw = m_current_Tcw.rowRange(0, 3).colRange(0, 3);
    Vec3f tcw = m_current_Tcw.rowRange(0, 3).col(3);


    float fx = intrincs.at<float>(0, 0);
    float fy = intrincs.at<float>(1, 1);
    float cx = intrincs.at<float>(0, 2);
    float cy = intrincs.at<float>(1, 2);

    int ref_neighs_num = ref_neighs.size();
    ref_neighs_num = ref_neighs_num < MAX_REF_NEIGHS ? ref_neighs_num : MAX_REF_NEIGHS;
    OrbKeyFrame* ref_keyframe = ref_neighs[0];

    // for debug count neighs exist point
    int count_neighs0_exist = 0;
    int count_neighs1_exist = 0;
    int count_neighs2_exist = 0;
    int count_neighs3_exist = 0;
    int count_neighs0_time = 0;
    int count_neighs1_time = 0;
    int count_neighs2_time = 0;
    int count_neighs3_time = 0;
    
    cv::Mat ref_Tcw[MAX_REF_NEIGHS];
    cv::Matx33f ref_Rcw[MAX_REF_NEIGHS];
    cv::Vec3f ref_tcw[MAX_REF_NEIGHS];

    for (int neigh_idx = 0; neigh_idx<ref_neighs_num; neigh_idx++)
    {
        ref_Tcw[neigh_idx] = (ref_neighs[neigh_idx]->Tcw);
        ref_Rcw[neigh_idx] = ref_Tcw[neigh_idx].rowRange(0, 3).colRange(0, 3);
        ref_tcw[neigh_idx] = ref_Tcw[neigh_idx].rowRange(0, 3).col(3);
    }

    FeatureUV ref_uvcoord;
    MapPoint ref_map_point;
    MapPoint map_point;
    FeatureUV curr_uvcoord;
    cv::Vec3f pointc;
    cv::Vec3f ref_pointcloud;
    cv::Vec3f dist;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    //v --> row, height
    for(size_t v = TOP_HEIGHT_BODER_LENGTH; v < (height - BOTTOM_HEIGHT_BODER_LENGTH); v += step)
    {
        //u --> col, width
        for(size_t u = WIDTH_BODER_LENGTH; u < (width - WIDTH_BODER_LENGTH); u += step)
        {
            float Z = ptr_depth[u];
            Vec3b c = ptr_color[u];

            if(Z <= densemap_params.min_depth || Z >= densemap_params.max_depth) continue;
            if(Z != Z) continue;

            // project curr_frame to world coord
            curr_uvcoord.u = u;
            curr_uvcoord.v = v;
            float x = float(curr_uvcoord.u);
            float y = float(curr_uvcoord.v);
            float Xw = Z*(x - cx) / fx;
            float Yw = Z*(y - cy) / fy;
            cv::Vec3f P(Xw, Yw, Z);
            P = Rcw.t() * P - Rcw.t()*tcw;

            // project world coord to ref_keyframe coord
            bool is_mappoint_exist = false;
            int count_neighs_search = 0;

            //for(auto iter_neigh= ref_neighs.begin(); iter_neigh!=ref_neighs.end(); iter_neigh++)
            for(int neigh_idx = 0; neigh_idx<ref_neighs_num; neigh_idx++)
            {
                //ref_keyframe = *iter_neigh;
                ref_keyframe = ref_neighs[neigh_idx];

                bool isInImagePlane = true;
                pointc = ref_Rcw[neigh_idx] * P + ref_tcw[neigh_idx];
                float ref_X = pointc[0] * fx + pointc[2] * cx;
                float ref_Y = pointc[1] * fy + pointc[2] * cy;
                float ref_Z = pointc[2];
                float ref_u = ref_X / ref_Z;
                float ref_v = ref_Y / ref_Z;
                if(ref_u<WIDTH_BODER_LENGTH || ref_u>(width - WIDTH_BODER_LENGTH)) isInImagePlane = false;
                if(ref_v<TOP_HEIGHT_BODER_LENGTH || ref_v>(height - BOTTOM_HEIGHT_BODER_LENGTH)) isInImagePlane = false;
                //int ref_iu = round(ref_u);
                //int ref_iv = round(ref_v);

                ref_uvcoord.u = int(ref_u);
                ref_uvcoord.v = int(ref_v);

                // for debug count neighs exist point
                //if(count_neighs_search == 3)
                //{
                //    count_neighs3_time = count_neighs3_time + 1;
                //}
                //else if(count_neighs_search == 2)
                //{
                //    count_neighs2_time = count_neighs2_time + 1;
                //}
                //else if(count_neighs_search == 1)
                //{
                //    count_neighs1_time = count_neighs1_time + 1;
                //}
                //else if(count_neighs_search == 0)
                //{
                //    count_neighs0_time = count_neighs0_time + 1;
                //}
                

                int pointIdx = 0;
                bool isUVCreatedPoint = ref_keyframe->getMapPointIdx(ref_uvcoord, pointIdx);
                if(isInImagePlane && isUVCreatedPoint && pointIdx < m_points.size())
                {
                    //greedy fusion
                    ref_map_point = m_points[pointIdx];

                    float max_weight = densemap_params.max_uncertainty_limit;
                    ref_pointcloud = ref_map_point.point;
                    float ref_weight = ref_map_point.uncertainty;

                    dist = ref_pointcloud - P;
                    float dist_len = dist.dot(dist);
                    float curr_weight = (dist_len < max_weight) ? dist_len : max_weight;
                    points_locker.lock();
                    if(dist_len < densemap_params.point_dist_thr)
                    {
                        bool last_stable_status = m_points[pointIdx].isStable;
                        //cvlibs_fusion_and_propagate(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);
                        cvlibs_fusion_and_propagate_incweight(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);
                        bool fusion_stable_status = m_points[pointIdx].isStable;
                        if(last_stable_status == false && fusion_stable_status == true)
                        {
                            cv::Vec3b color(c[2], c[1], c[0]);

                            m_colors[pointIdx] = color;
                        }

                        // update label
                        #ifdef WITH_LABEL
                        uchar curr_label = mergeSimilarClass(ptr_label[u]);
                        label_fusion(curr_label, m_points[pointIdx]);
                        #endif
                    }

                    // for debug count neighs exist point
                    //if(count_neighs_search == 3)
                    //{
                    //    count_neighs3_exist = count_neighs3_exist + 1;
                    //}
                    //else if(count_neighs_search == 2)
                    //{
                    //    count_neighs2_exist = count_neighs2_exist + 1;
                    //}
                    //else if(count_neighs_search == 1)
                    //{
                    //    count_neighs1_exist = count_neighs1_exist + 1;
                    //}
                    //else if(count_neighs_search == 0)
                    //{
                    //    count_neighs0_exist = count_neighs0_exist + 1;
                    //}


                    // add observes
                    curr_keyframe.addMapPoint(curr_uvcoord, pointIdx);
                    points_locker.unlock();

                    // mappoint exist, just fusion
                    is_mappoint_exist = true;

                    
                    break;
                    
                }
                count_neighs_search = count_neighs_search + 1;
            }
          
            if(!is_mappoint_exist)
            {
                int pointIdx = m_points.size();

                // not exist, then create new pointcloud 
                points_locker.lock();

                map_point.idx = pointIdx;
                map_point.point = P;
                map_point.isStable = false;
                map_point.uncertainty = densemap_params.point_init_uncertainty;

                #ifdef WITH_LABEL
                map_point.label = mergeSimilarClass(ptr_label[u]);
                map_point.label_confidence = 0;
                map_point.label_color = label_colorful(map_point.label);
                #endif

                m_points.emplace_back(map_point);
                m_colors.emplace_back(c[2], c[1], c[0]);

                curr_keyframe.addMapPoint(curr_uvcoord, pointIdx);
                points_locker.unlock();
            }
        }


        ptr_depth += stride;
        ptr_color += stride;
        #ifdef WITH_LABEL
        ptr_label += stride;
        #endif
    }

    unique_lock<mutex> curr_keyframe_locker(m_current_keyframe_mutex, defer_lock);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "keyframe id:" << curr_keyframe.frame_idx << ", cost time: " << ttrack << std::endl;
    curr_keyframe_locker.lock();
    m_current_keyframe = curr_keyframe;
    curr_keyframe_locker.unlock();
    //std::cout << "neighs search times: " << count_neighs0_time << ", " << count_neighs1_time << ", " << count_neighs2_time << ", " << count_neighs3_time << std::endl;
    //std::cout << "neighs exist points: " << count_neighs0_exist << ", " << count_neighs1_exist << ", " << count_neighs2_exist << ", " << count_neighs3_exist << std::endl;
}


void DenseMap::cvlibs_greedy_fusion_framewise_covisible(std::vector<OrbKeyFrame*> &ref_neighs, OrbKeyFrame &curr_frame)
{

    if(curr_frame.depth.empty() || curr_frame.image.empty()) return;
    CV_Assert(curr_frame.depth.size() == curr_frame.image.size());

    cv::Mat intrincs = curr_frame.K;

    unique_lock<mutex> depth_locker(m_depth_mutex, defer_lock);
    unique_lock<mutex> points_locker(m_points_mutex, defer_lock);

    depth_locker.lock();
    m_current_image = curr_frame.image;
    m_current_Tcw = curr_frame.Tcw.clone();	// deep copy
    m_current_depth = curr_frame.depth;		// shallow copy

    #ifdef WITH_LABEL
    m_current_label = curr_frame.label;
    uchar *ptr_label = m_current_label.ptr<uchar>();
    #endif
    depth_locker.unlock();

    int width = m_current_depth.cols;
    int height = m_current_depth.rows;
    float* ptr_depth = m_current_depth.ptr<float>();
    Vec3b* ptr_color = m_current_image.ptr<Vec3b>();

    int step = 1;
    int stride = step * m_current_depth.cols;


    ptr_depth += stride*TOP_HEIGHT_BODER_LENGTH;
    ptr_color += stride*TOP_HEIGHT_BODER_LENGTH;

    Matx33f Rcw = m_current_Tcw.rowRange(0, 3).colRange(0, 3);
    Vec3f tcw = m_current_Tcw.rowRange(0, 3).col(3);


    float fx = intrincs.at<float>(0, 0);
    float fy = intrincs.at<float>(1, 1);
    float cx = intrincs.at<float>(0, 2);
    float cy = intrincs.at<float>(1, 2);

    int ref_neighs_num = ref_neighs.size();
    ref_neighs_num = ref_neighs_num < MAX_REF_NEIGHS ? ref_neighs_num : MAX_REF_NEIGHS;
    OrbKeyFrame* ref_keyframe = ref_neighs[0];

    // for debug count neighs exist point
    int count_neighs0_exist = 0;
    int count_neighs1_exist = 0;
    int count_neighs2_exist = 0;
    int count_neighs3_exist = 0;
    int count_neighs0_time = 0;
    int count_neighs1_time = 0;
    int count_neighs2_time = 0;
    int count_neighs3_time = 0;

    cv::Mat ref_Tcw[MAX_REF_NEIGHS];
    cv::Matx33f ref_Rcw[MAX_REF_NEIGHS];
    cv::Vec3f ref_tcw[MAX_REF_NEIGHS];

    for(int neigh_idx = 0; neigh_idx<ref_neighs_num; neigh_idx++)
    {
        ref_Tcw[neigh_idx] = (ref_neighs[neigh_idx]->Tcw);
        ref_Rcw[neigh_idx] = ref_Tcw[neigh_idx].rowRange(0, 3).colRange(0, 3);
        ref_tcw[neigh_idx] = ref_Tcw[neigh_idx].rowRange(0, 3).col(3);
    }

    FeatureUV ref_uvcoord;
    MapPoint ref_map_point;
    MapPoint map_point;
    FeatureUV curr_uvcoord;
    cv::Vec3f pointc;
    cv::Vec3f ref_pointcloud;
    cv::Vec3f dist;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    //v --> row, height
    for(size_t v = TOP_HEIGHT_BODER_LENGTH; v < (height - BOTTOM_HEIGHT_BODER_LENGTH); v += step)
    {
        //u --> col, width
        for(size_t u = WIDTH_BODER_LENGTH; u < (width - WIDTH_BODER_LENGTH); u += step)
        {
            float Z = ptr_depth[u];
            Vec3b c = ptr_color[u];

            if(Z <= densemap_params.min_depth || Z >= densemap_params.max_depth) continue;
            if(Z != Z) continue;

            // project curr_frame to world coord
            curr_uvcoord.u = u;
            curr_uvcoord.v = v;
            float x = float(curr_uvcoord.u);
            float y = float(curr_uvcoord.v);
            float Xw = Z*(x - cx) / fx;
            float Yw = Z*(y - cy) / fy;
            cv::Vec3f P(Xw, Yw, Z);
            P = Rcw.t() * P - Rcw.t()*tcw;

            // project world coord to ref_keyframe coord
            bool is_mappoint_exist = false;
            int count_neighs_search = 0;

            //for(auto iter_neigh= ref_neighs.begin(); iter_neigh!=ref_neighs.end(); iter_neigh++)
            for(int neigh_idx = 0; neigh_idx<ref_neighs_num; neigh_idx++)
            {
                //ref_keyframe = *iter_neigh;
                ref_keyframe = ref_neighs[neigh_idx];

                bool isInImagePlane = true;
                pointc = ref_Rcw[neigh_idx] * P + ref_tcw[neigh_idx];
                float ref_X = pointc[0] * fx + pointc[2] * cx;
                float ref_Y = pointc[1] * fy + pointc[2] * cy;
                float ref_Z = pointc[2];
                float ref_u = ref_X / ref_Z;
                float ref_v = ref_Y / ref_Z;
                if(ref_u<WIDTH_BODER_LENGTH || ref_u>(width - WIDTH_BODER_LENGTH)) isInImagePlane = false;
                if(ref_v<TOP_HEIGHT_BODER_LENGTH || ref_v>(height - BOTTOM_HEIGHT_BODER_LENGTH)) isInImagePlane = false;

                ref_uvcoord.u = int(ref_u);
                ref_uvcoord.v = int(ref_v);

                // for debug count neighs exist point
                //if(count_neighs_search == 3)
                //{
                //    count_neighs3_time = count_neighs3_time + 1;
                //}
                //else if(count_neighs_search == 2)
                //{
                //    count_neighs2_time = count_neighs2_time + 1;
                //}
                //else if(count_neighs_search == 1)
                //{
                //    count_neighs1_time = count_neighs1_time + 1;
                //}
                //else if(count_neighs_search == 0)
                //{
                //    count_neighs0_time = count_neighs0_time + 1;
                //}

                int pointIdx = 0;
                bool isUVCreatedPoint = ref_keyframe->getMapPointIdx(ref_uvcoord, pointIdx);
                if(isInImagePlane && isUVCreatedPoint && pointIdx < m_points.size())
                {
                    //greedy fusion
                    ref_map_point = m_points[pointIdx];

                    float max_weight = densemap_params.max_uncertainty_limit;
                    ref_pointcloud = ref_map_point.point;
                    float ref_weight = ref_map_point.uncertainty;

                    dist = ref_pointcloud - P;
                    float dist_len = dist.dot(dist);
                    float curr_weight = (dist_len < max_weight) ? dist_len : max_weight;
                    points_locker.lock();
                    if(dist_len < densemap_params.point_dist_thr)
                    {
                        bool last_stable_status = m_points[pointIdx].isStable;
                        //cvlibs_fusion_and_propagate(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);
                        cvlibs_fusion_and_propagate_incweight(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);

                        bool fusion_stable_status = m_points[pointIdx].isStable;
                        //if(last_stable_status == false && fusion_stable_status == true)
                        //{
                        //    cv::Vec3b color(c[2], c[1], c[0]);
                        //    m_colors[pointIdx] = color;
                        //}

                        // update label
                        #ifdef WITH_LABEL
                        uchar curr_label = mergeSimilarClass(ptr_label[u]);
                        label_fusion(curr_label, m_points[pointIdx]);
                        #endif
                    }

                    //for debug count neighs exist point
                    //if(count_neighs_search == 3)
                    //{
                    //    count_neighs3_exist = count_neighs3_exist + 1;
                    //}
                    //else if(count_neighs_search == 2)
                    //{
                    //    count_neighs2_exist = count_neighs2_exist + 1;
                    //}
                    //else if(count_neighs_search == 1)
                    //{
                    //    count_neighs1_exist = count_neighs1_exist + 1;
                    //}
                    //else if(count_neighs_search == 0)
                    //{
                    //    count_neighs0_exist = count_neighs0_exist + 1;
                    //}

                    // add observes
                    points_locker.unlock();
                    break;
                }
                count_neighs_search = count_neighs_search + 1;
            }
        }


        ptr_depth += stride;
        ptr_color += stride;
        #ifdef WITH_LABEL
        ptr_label += stride;
        #endif
    }

    unique_lock<mutex> curr_keyframe_locker(m_current_keyframe_mutex, defer_lock);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "frame id:" << curr_frame.frame_idx << ", cost time: " << ttrack << std::endl;

    //std::cout << "neighs search times: " << count_neighs0_time << ", " << count_neighs1_time << ", " << count_neighs2_time << ", " << count_neighs3_time << std::endl;
    //std::cout << "neighs exist points: " << count_neighs0_exist << ", " << count_neighs1_exist << ", " << count_neighs2_exist << ", " << count_neighs3_exist << std::endl;
}


void DenseMap::cvlibs_greedy_fusion_framewise(OrbKeyFrame &ref_keyframe, OrbKeyFrame &curr_frame)
{

    if(curr_frame.depth.empty() || curr_frame.image.empty()) return;
    CV_Assert(curr_frame.depth.size() == curr_frame.image.size());

    cv::Mat intrincs = curr_frame.K;

    unique_lock<mutex> depth_locker(m_depth_mutex, defer_lock);
    unique_lock<mutex> points_locker(m_points_mutex, defer_lock);

    depth_locker.lock();
    cv::Mat image = curr_frame.image;
    cv::Mat Tcw = curr_frame.Tcw.clone();	// deep copy
    cv::Mat ref_Tcw = ref_keyframe.Tcw.clone();	// deep copy
    cv::Mat depth = curr_frame.depth;		// shallow copy


    depth_locker.unlock();

    int width = depth.cols;
    int height = depth.rows;
    float* ptr_depth = depth.ptr<float>();
    cv::Vec3b* ptr_color = image.ptr<Vec3b>();

    int step = 1;
    int stride = step * depth.cols;

    ptr_depth += stride*TOP_HEIGHT_BODER_LENGTH;
    ptr_color += stride*TOP_HEIGHT_BODER_LENGTH;

    cv::Matx33f Rcw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Vec3f tcw = Tcw.rowRange(0, 3).col(3).clone();

    cv::Matx33f ref_Rcw = ref_Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Vec3f ref_tcw = ref_Tcw.rowRange(0, 3).col(3).clone();

    //std::cout << Tcw << std::endl;
    float fx = intrincs.at<float>(0, 0);
    float fy = intrincs.at<float>(1, 1);
    float cx = intrincs.at<float>(0, 2);
    float cy = intrincs.at<float>(1, 2);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int count_disp = 0;
    cv::Vec3f ref_pointcloud;
    cv::Vec3f dist;
    MapPoint ref_map_point;
    cv::Vec3f pointc;
    FeatureUV ref_uvcoord;

    //v --> row, height
    for(size_t v = TOP_HEIGHT_BODER_LENGTH; v < (height - BOTTOM_HEIGHT_BODER_LENGTH); v += step)
    {
        //u --> col, width
        for(size_t u = WIDTH_BODER_LENGTH; u < (width - WIDTH_BODER_LENGTH); u += step)
        {
            float Z = ptr_depth[u];
            Vec3b c = ptr_color[u];

            if(Z <= densemap_params.min_depth || Z >= densemap_params.max_depth) continue;
            if(Z != Z) continue;

            // project curr_frame to world coord
            float x = float(u);
            float y = float(v);
            float Xw = Z*(x - cx) / fx;
            float Yw = Z*(y - cy) / fy;
            cv::Vec3f P(Xw, Yw, Z);
            P = Rcw.t() * P - Rcw.t()*tcw;

            // project world coord to ref_keyframe coord
            bool isInImagePlane = true;
            pointc = ref_Rcw * P + ref_tcw;
            float ref_X = pointc[0] * fx + pointc[2] * cx;
            float ref_Y = pointc[1] * fy + pointc[2] * cy;
            float ref_Z = pointc[2];
            float ref_u = ref_X / ref_Z;
            float ref_v = ref_Y / ref_Z;
            if(ref_u<WIDTH_BODER_LENGTH || ref_u>(width - WIDTH_BODER_LENGTH)) isInImagePlane = false;
            if(ref_v<TOP_HEIGHT_BODER_LENGTH || ref_v>(height - BOTTOM_HEIGHT_BODER_LENGTH)) isInImagePlane = false;
            //int ref_iu = round(ref_u);
            //int ref_iv = round(ref_v);
            ref_uvcoord.u = int(ref_u);
            ref_uvcoord.v = int(ref_v);

            //// cvlibs fusion    
            int pointIdx = 0;
            bool isUVCreatedPoint = ref_keyframe.getMapPointIdx(ref_uvcoord, pointIdx);
            if(isInImagePlane && isUVCreatedPoint && pointIdx < m_points.size())
            {
                //greedy fusion
                ref_map_point = m_points[pointIdx];

                float max_weight = densemap_params.max_uncertainty_limit;
                ref_pointcloud = ref_map_point.point;
                float ref_weight = ref_map_point.uncertainty;

                dist = ref_pointcloud - P;
                float dist_len = dist.dot(dist);
                float curr_weight = (dist_len < max_weight) ? dist_len : max_weight;

                points_locker.lock();

                // more strictly require for stable
                if(dist_len < densemap_params.point_dist_thr)
                {
                    bool last_stable_status = m_points[pointIdx].isStable;
                    //cvlibs_fusion_and_propagate(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);
                    cvlibs_fusion_and_propagate_incweight(ref_pointcloud, P, ref_weight, curr_weight, m_points[pointIdx]);
                    bool fusion_stable_status = m_points[pointIdx].isStable;
                    if(last_stable_status == false && fusion_stable_status == true)
                    {
                        cv::Vec3b color(c[2], c[1], c[0]);
                        m_colors[pointIdx] = color;
                        count_disp++;
                    }
                }

                points_locker.unlock();
            }
        }

        
        ptr_depth += stride;
        ptr_color += stride;
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    //std::cout << "frame-wise time: " << ttrack << std::endl;

    //int pointsize = m_points.size();
    //std::cout << "refine frame id: " << curr_frame.frame_idx << ", new stable point: " << count_disp << ", pointsize " << pointsize << std::endl;
}


uchar DenseMap::mergeSimilarClass(uchar label_predict)
{
    uchar merge_label;
    merge_label = label_predict;

    //merge cusion & seat to sofa
    if(label_predict == label_name::seat || label_predict == label_name::cushion)
    {
        merge_label = label_name::sofa;
    }

    // merge rug to floor
    if(label_predict == label_name::rug)
    {
        merge_label = label_name::floor;
    }

    //merge computer & televison to screen
    if(label_predict == label_name::computer || label_predict == label_name::television)
    {
        merge_label = label_name::screen;
    }

    return merge_label;
}

void DenseMap::cvlibs_fusion_and_propagate(cv::Vec3f &ref_pointcloud, cv::Vec3f &curr_pointcloud,
    float ref_weight, float curr_weight, MapPoint &point)
{
    bool last_stable_status = point.isStable;

    // propagate pointclouds & weighted-average point
    point.point = (curr_weight*ref_pointcloud + ref_weight*curr_pointcloud) / (ref_weight + curr_weight);

    // propagate weight
    float update_weight = (densemap_params.uncertainty_weight*curr_weight + 1.0f*ref_weight) / (densemap_params.uncertainty_weight+1.0f);
    update_weight = (update_weight < densemap_params.max_uncertainty_limit) ? update_weight : densemap_params.max_uncertainty_limit;
    point.uncertainty = update_weight;

    // delay_loop judge stable
    if(last_stable_status)
    {
        // delay_loop thr
        float stable_thr = densemap_params.point_stable_thr + densemap_params.stable_delay_loop_thr;

        // more strictly require for stable
        if(update_weight < stable_thr)
        {
            point.isStable = true;
        }
        else
        {
            point.isStable = false;
        }
    }
    else
    {
        // delay_loop thr
        float stable_thr = densemap_params.point_stable_thr - densemap_params.stable_delay_loop_thr;

        // more strictly require for stable
        if(update_weight < stable_thr)
        {
            point.isStable = true;
        }
        else
        {
            point.isStable = false;
        }
    }
}

void DenseMap::cvlibs_fusion_and_propagate_incweight(cv::Vec3f &ref_pointcloud, cv::Vec3f &curr_pointcloud,
    float ref_weight, float curr_weight, MapPoint &point)
{
    bool last_stable_status = point.isStable;
    int old_weight = point.fusion_weight;
    int new_weight = 1;

    //std::cout << "weighted" << std::endl;
    // propagate pointclouds & weighted-average point
    point.point = (old_weight*ref_pointcloud + new_weight*curr_pointcloud) / float(old_weight + new_weight);

    // propagate weight
    /*float update_weight = (densemap_params.uncertainty_weight*curr_weight + 1.0f*ref_weight) / (densemap_params.uncertainty_weight + 1.0f);
    update_weight = (update_weight < densemap_params.max_uncertainty_limit) ? update_weight : densemap_params.max_uncertainty_limit;*/
    
    float update_uncertainty = (new_weight*curr_weight + old_weight*ref_weight) / (old_weight + new_weight);
    update_uncertainty = (update_uncertainty < densemap_params.max_uncertainty_limit) ? update_uncertainty : densemap_params.max_uncertainty_limit;

    point.uncertainty = update_uncertainty;

    new_weight = old_weight + new_weight;
    point.fusion_weight = std::min(new_weight, densemap_params.max_fusion_weight);

    // delay_loop judge stable
    if(last_stable_status)
    {
        // delay_loop thr
        float stable_thr = densemap_params.point_stable_thr + densemap_params.stable_delay_loop_thr;

        // more strictly require for stable
        if(update_uncertainty < stable_thr)
        {
            point.isStable = true;
        }
        else
        {
            point.isStable = false;
        }
    }
    else
    {
        // delay_loop thr
        float stable_thr = densemap_params.point_stable_thr - densemap_params.stable_delay_loop_thr;

        // more strictly require for stable
        if(update_uncertainty < stable_thr)
        {
            point.isStable = true;
        }
        else
        {
            point.isStable = false;
        }
    }
}


void DenseMap::clear_points()
{
	unique_lock<mutex> lock(m_points_mutex);
	m_points.clear();
}

void DenseMap::set_current_Tcw(cv::Mat & frame_Tcw)
{
    unique_lock<mutex> lock(m_depth_mutex);
    m_current_Tcw = frame_Tcw.clone();
}

void DenseMap::get_current_Tcw(cv::Mat & current_Tcw)
{
	unique_lock<mutex> lock(m_depth_mutex);
	current_Tcw = m_current_Tcw.clone();
}

void DenseMap::get_current_depth_map(cv::Mat & depth)
{
	unique_lock<mutex> lock(m_depth_mutex);
	depth = m_current_depth.clone();
}

void DenseMap::get_dense_points(std::vector<cv::Vec3f>& points, 
    std::vector<cv::Vec3b>& colors, bool draw_label, bool draw_single_frame, bool draw_color)
{
	unique_lock<mutex> lock(m_points_mutex);

    if(m_points.empty()) return;

    int not_visible_count = 0;
    int pointsize = m_points.size();

    if(draw_single_frame)
    {
        unique_lock<mutex> curr_keyframe_locker(m_current_keyframe_mutex, defer_lock);
        curr_keyframe_locker.lock();
        cv::Mat intrincs = m_current_keyframe.K;
        cv::Mat img = m_current_keyframe.image;
        cv::Mat Tcw = m_current_keyframe.Tcw.clone();	// deep copy
        cv::Mat depth = m_current_keyframe.depth;		// shallow copy
        curr_keyframe_locker.unlock();

        int width = depth.cols;
        int height = depth.rows;
        float* ptr_depth = depth.ptr<float>();
        Vec3b* ptr_color = img.ptr<Vec3b>();

        int step = 1;
        int stride = step * depth.cols;

        ptr_depth += stride*TOP_HEIGHT_BODER_LENGTH;
        ptr_color += stride*TOP_HEIGHT_BODER_LENGTH;
        //v --> row, height
        for(size_t v = TOP_HEIGHT_BODER_LENGTH; v < (height - BOTTOM_HEIGHT_BODER_LENGTH); v += step)
        {
            //u --> col, width
            for(size_t u = WIDTH_BODER_LENGTH; u < (width - WIDTH_BODER_LENGTH); u += step)
            {
                float Z = ptr_depth[u];
                Vec3b c = ptr_color[u];

                if(Z <= densemap_params.min_depth || Z >= densemap_params.max_depth) continue;
                if(Z != Z) continue;

                // projection & reprojection
                FeatureUV curr_uvcoord(u, v);
                cv::Vec3f P = projection_2d3d(curr_uvcoord, Z, intrincs, Tcw);

                MapPoint map_point;
                map_point.point = P;
                map_point.isStable = true;
                map_point.uncertainty = densemap_params.point_init_uncertainty;

                points.emplace_back(map_point.point);

                if (draw_color)
                {
                    colors.emplace_back(c[2], c[1], c[0]);
                }
                else
                {
                    colors.emplace_back(100, 10, 10);
                }
                
            }

            ptr_depth += stride;
            ptr_color += stride;

        }
               
        cv::waitKey(10);
        //v --> row, height  u --> col, width
        /*for(size_t v = 10; v < (height - 10); v += 1)
        {
            for(size_t u = 10; u < (width - 10); u += 1)
            {
                FeatureUV uvcoord(u, v);
                int pointIdx;

                unique_lock<mutex> curr_keyframe_locker(m_current_keyframe_mutex);
                bool isExistMapPoint = m_current_keyframe.getMapPointIdx(uvcoord, pointIdx);
                curr_keyframe_locker.unlock();

                if(isExistMapPoint)
                {
                    MapPoint map_point = m_points[pointIdx];
                    if(map_point.isStable)
                    {
                        points.emplace_back(map_point.point);

                        #ifdef WITH_LABEL
                        if(draw_label) colors.emplace_back(map_point.label_color);
                        else colors.emplace_back(m_colors[pointIdx]);
                        #else
                        colors.emplace_back(m_colors[pointIdx]);
                        #endif
                    }
                }
            }
        }*/
    }
    else
    {
        for(int i = 0; i < pointsize; i++)
        {
            MapPoint map_point = m_points[i];
            if(map_point.isStable)
            {
                points.emplace_back(map_point.point);

                #ifdef WITH_LABEL
                if(draw_label) colors.emplace_back(map_point.label_color);
                else colors.emplace_back(m_colors[i]);
                #else
                

                if(draw_color)
                {
                    colors.emplace_back(m_colors[i]);
                }
                else
                {
                    //float conf = map_point.uncertainty;
                    //float r_norm_scale = 10.0 / densemap_params.point_stable_thr;
                    //float g_norm_scale = 30.0 / densemap_params.point_stable_thr;
                    //float b_norm_scale = 40.0 / densemap_params.point_stable_thr;
                    //int r_conf_norm = int(conf * r_norm_scale + 200);
                    //int g_conf_norm = int(conf * g_norm_scale + 80);
                    //int b_conf_norm = int(conf * b_norm_scale + 90);

                    cv::Vec3f point_value = map_point.point;
                    float dist = point_value.dot(point_value);
                    float r_norm_scale = 200.0 / 10;
                    float g_norm_scale = 200.0 / 10;
                    float b_norm_scale = 200.0 / 10;
                    int r_conf_norm = int(dist * r_norm_scale + 40);
                    int g_conf_norm = int(dist * g_norm_scale + 40);
                    int b_conf_norm = int(dist * b_norm_scale + 40);

                    if(r_conf_norm > 255 || g_conf_norm > 255 || b_conf_norm > 255)
                    {
                        r_conf_norm = 255;
                        g_conf_norm = 255;
                        b_conf_norm = 255;
                    }
                        
                    colors.emplace_back(r_conf_norm, g_conf_norm, b_conf_norm);
                }

                #endif
            }
            else
            {
                not_visible_count++;
            }
        }
    }


    if(point_nums == pointsize && invisible_point_size == not_visible_count) return;
    else
    {
        point_nums = pointsize;
        invisible_point_size = not_visible_count; 
    }
}

// Tc2c1 : c1 position in c2 coordinate, transform c1 to c2
// return : K * Tc2w * Tc1w.t() * K.t()
cv::Mat DenseMap::calc_transfrom_matrix_2d2d(cv::Mat &Tc1w, cv::Mat &Tc2w)
{
    // calc Twc1 --> camera pose , c1 in word coord
    cv::Mat Rc1w = Tc1w.rowRange(0, 3).colRange(0, 3);
    cv::Mat tc1w = Tc1w.rowRange(0, 3).col(3);
    cv::Mat Rwc1 = Rc1w.t();
    cv::Mat Ow = -Rwc1*tc1w;

    cv::Mat Twc1= cv::Mat::eye(4, 4, Tc1w.type());
    Rwc1.copyTo(Twc1.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(Twc1.rowRange(0, 3).col(3));

    cv::Mat Tc2c1 = cv::Mat::eye(4, 4, Tc1w.type());
    Tc2c1 = Tc2w * Twc1;
    return Tc2c1;
}

// projection 3d pointclouds to 2d 
bool DenseMap::reprojection_3d2d(const cv::Vec3f &pointw, const cv::Mat &K, const cv::Mat &Tcw, const int cols, const int rows, FeatureUV &uvcoord)
{
    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);


    Matx33f Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    Vec3f tcw = Tcw.rowRange(0, 3).col(3);

    cv::Vec3f pointc = Rcw * pointw + tcw;


    float X = pointc[0] * fx + pointc[2] * cx;
    float Y = pointc[1] * fy + pointc[2] * cy;
    float Z = pointc[2];
    
    float u = X / Z;
    float v = Y / Z;

    if(u<WIDTH_BODER_LENGTH || u>(cols - WIDTH_BODER_LENGTH)) return false;
    if(v<TOP_HEIGHT_BODER_LENGTH || v>(rows - BOTTOM_HEIGHT_BODER_LENGTH)) return false;

    int iu = round(u);
    int iv = round(v);


    uvcoord.u = iu;
    uvcoord.v = iv;

    return true;
}


// projection 3d pointclouds to 2d 
cv::Vec3f DenseMap::projection_2d3d(const FeatureUV &uvcoord, const float Zc, const cv::Mat &K, const cv::Mat &Tcw)
{
    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);


    Matx33f Rcw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
    Vec3f tcw = Tcw.rowRange(0, 3).col(3).clone();

    float x = float(uvcoord.u);
    float y = float(uvcoord.v);

    float Xw = Zc*(x - cx) / fx;
    float Yw = Zc*(y - cy) / fy;

    // transform point by Tcw
    Vec3f P(Xw, Yw, Zc);
    P = Rcw.t() * P - Rcw.t()*tcw;

    return P;
}


void DenseMap::label_fusion(uchar curr_label, MapPoint &point)
{
    #ifdef WITH_LABEL
    int16_t ref_label_conf = point.label_confidence;
    uchar ref_label = point.label;

    if(ref_label == curr_label)
    {
        if(ref_label_conf < densemap_params.label_fusion_thr)
        {
            ref_label_conf = ref_label_conf + 1;
        }
    }
    else
    {
        if(ref_label_conf > 0) ref_label_conf = ref_label_conf - 1;
        if(ref_label_conf == 0) point.label = curr_label;
    }

    point.label_confidence = ref_label_conf;
    point.label_color = label_colorful(curr_label);
    #endif
}