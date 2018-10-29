#include "DepthPredictor.h"
#include<iostream>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgproc.hpp>


using namespace cv;

DepthPredictor::DepthPredictor(ORB_SLAM2::Map* pMap, KeyFrameQueue* frame_queue,
    DenseMap* dense_map, std::string setting_file):mpOrbSlamMap(pMap),
	m_success(false), m_frame_queue(frame_queue), m_dense_map(dense_map), m_should_finish(false),
	m_is_finished(true)
{
	FileStorage fs(setting_file, FileStorage::READ);

	m_img_width = fs["Camera.width"];
	m_img_height = fs["Camera.height"];
	m_focal_length = fs["Camera.fx"];
    m_depth_grad_thr = fs["Camera.depth_grad_thr"];
    m_net_width = fs["Camera.net_width"];
    m_net_height = fs["Camera.net_height"];
    m_covisible_keyframes = fs["DenseMap.CovisibleForKeyFrames"];
    m_covisible_frames = fs["DenseMap.CovisibleForFrames"];

    std::cout << "------------------------\nDenseMap DepthPredictor Params" << std::endl;
    std::cout << "- Covisible For KeyFrames: " << m_covisible_keyframes << std::endl;
    std::cout << "- Covisible For Frames: " << m_covisible_frames << std::endl;
	fs.release();

    m_first_frameid = 0;
	m_success = true;
}

DepthPredictor::~DepthPredictor()
{
	request_finish();
}

void DepthPredictor::run()
{
	set_finished(false);
    while(!should_finish())
    {
        if(m_frame_queue->size() == 0)
            continue;

        std::unique_lock <std::mutex> lck(m_dense_map->system_running_mtx);
        while(!m_dense_map->get_system_running_status())
        {
            m_dense_map->system_running_cv.wait(lck);
        }
        lck.unlock();

		OrbKeyFrame curr_frame = m_frame_queue->dequeue();
        if(vOrbKeyframes.empty())
        {
            addKeyframe(curr_frame.frame_idx, curr_frame);
            m_dense_map->add_points_from_depth_map(curr_frame);
            m_first_frameid = curr_frame.frame_idx;
            std::cout << "First keyframe id in densemap: " << m_first_frameid << std::endl;
        }
        else
        {
            auto iter = vOrbKeyframes.end() - 1;
            OrbKeyFrame ref_keyframe = *iter;
            std::vector<OrbKeyFrame*> ref_neighs;
            ref_neighs.emplace_back(&ref_keyframe);
            if(curr_frame.isKeyframe)
            {
                //std::cout << "\nDenseMap keyframe size: " << vOrbKeyframes.size() 
                //          << ", keyframe id->" << ref_keyframe.frame_idx
                //          << ", curr_frame id->" << curr_frame.frame_idx 
                //          << ", this is keyframe" << std::endl;
                
                ORB_SLAM2::KeyFrame *orbslam_kf = mpOrbSlamMap->getKeyframeByframeId(ref_keyframe.frame_idx);
                if(orbslam_kf != nullptr)
                {
                    std::vector<ORB_SLAM2::KeyFrame*> vNeighs = orbslam_kf->GetBestCovisibilityKeyFrames(m_covisible_keyframes);
                    for(auto iter_neigh = vNeighs.begin(); iter_neigh != vNeighs.end(); iter_neigh++)
                    {
                        long unsigned int neigh_id = (*iter_neigh)->mnFrameId;
                        if(neigh_id > m_first_frameid && neigh_id < curr_frame.frame_idx)
                        {
                            // get OrbKeyFrame use orbslam neighs_frame_id
                            long unsigned int neigh_kf_id = map_frameid_keyframeid[neigh_id];
                            ref_neighs.emplace_back(&vOrbKeyframes[neigh_kf_id]);
                        }
                    }
                }
                else
                {
                    //std::cout << "**keyframe has delete by Map**" << std::endl;
                }

                //std::cout << "ref_neighs size: " << ref_neighs.size() << ", neigh id:";
                //for(auto iter_neigh = ref_neighs.begin(); iter_neigh != ref_neighs.end(); iter_neigh++)
                //{
                //    std::cout << (*iter_neigh)->frame_idx << " ";
                //}
                //std::cout << "\n";
                //if(ref_neighs.size() < 2)
                //{
                //    m_dense_map->cvlibs_greedy_fusion(ref_keyframe, curr_frame);
                //}
                //else
                //{
                //    m_dense_map->cvlibs_greedy_fusion_covisible(ref_neighs, curr_frame);
                //}

                // covisible
                m_dense_map->cvlibs_greedy_fusion_covisible(ref_neighs, curr_frame);

                // add keyframe to densemap keyframe manager
                addKeyframe(curr_frame.frame_idx, curr_frame);
            }
            else
            {
                //std::cout << "keyframe size: " << orbkeyframes.size() << ", keyframe id->" << ref_keyframe.frame_idx 
                //    << ", curr_frame id->" << curr_frame.frame_idx << std::endl;
                
                //m_dense_map->cvlibs_greedy_fusion_framewise(ref_keyframe, curr_frame);

                ORB_SLAM2::KeyFrame *orbslam_kf = mpOrbSlamMap->getKeyframeByframeId(ref_keyframe.frame_idx);
                if(orbslam_kf != nullptr)
                {
                    std::vector<ORB_SLAM2::KeyFrame*> vNeighs = orbslam_kf->GetBestCovisibilityKeyFrames(m_covisible_frames);
                    for(auto iter_neigh = vNeighs.begin(); iter_neigh != vNeighs.end(); iter_neigh++)
                    {
                        long unsigned int neigh_id = (*iter_neigh)->mnFrameId;
                        if(neigh_id > m_first_frameid && neigh_id < curr_frame.frame_idx)
                        {
                            // get OrbKeyFrame use orbslam neighs_frame_id
                            long unsigned int neigh_kf_id = map_frameid_keyframeid[neigh_id];
                            ref_neighs.emplace_back(&vOrbKeyframes[neigh_kf_id]);
                        }
                    }
                }
                else
                {
                    //std::cout << "**keyframe has delete by Map**" << std::endl;
                }

                //std::cout << "refneighs num: "<< ref_neighs.size() << std::endl;
                m_dense_map->cvlibs_greedy_fusion_framewise_covisible(ref_neighs, curr_frame);
                m_dense_map->set_current_Tcw(curr_frame.Tcw);
                
            }
        }
        
	}
	set_finished(true);
}

void DepthPredictor::set_finished(bool status)
{
    std::unique_lock<std::mutex> lock(m_mutex);
	m_is_finished = status;
	m_should_finish = status;
}

void DepthPredictor::request_finish()
{
    std::unique_lock<std::mutex> lock(m_mutex);
	m_should_finish = true;
}

bool DepthPredictor::should_finish()
{
    std::unique_lock<std::mutex> lock(m_mutex);
	return m_should_finish;
}
