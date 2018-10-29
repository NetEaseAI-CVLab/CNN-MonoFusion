#pragma once

#include<string>
#include<opencv2\core.hpp>
#include<mutex>
#include "KeyFrameQueue.h"
#include "DenseMap.h"
#include "Map.h"

class DepthPredictor
{
public:
	DepthPredictor(ORB_SLAM2::Map* pMap, KeyFrameQueue* frame_queue, DenseMap* dense_map, std::string setting_file);
	~DepthPredictor();
	void run();

	void request_finish();
	bool should_finish();
    void addKeyframe(long unsigned int frame_id, OrbKeyFrame &keyframe)
    {
        vOrbKeyframes.emplace_back(keyframe);
        long unsigned int kf_id = vOrbKeyframes.size() - 1;
        map_frameid_keyframeid[frame_id] = kf_id;

        //std::cout << "keyframe idx: " << vOrbKeyframes[map_frameid_keyframeid[frame_id]].frame_idx << ", frame id->" << frame_id << std::endl;
    }
protected:

	int			m_img_width;
	int			m_img_height;
    int			m_net_width;
    int			m_net_height;
	float		m_focal_length;
    float       m_depth_grad_thr;
    int         m_covisible_frames;
    int         m_covisible_keyframes;


	cv::Mat		m_depth;

	bool		m_success;
	bool		m_should_finish;
	bool		m_is_finished;
	std::mutex	m_mutex;

	KeyFrameQueue* m_frame_queue;
	DenseMap*	   m_dense_map;
    std::vector<OrbKeyFrame> vOrbKeyframes;
    std::map<long unsigned int, long unsigned int> map_frameid_keyframeid;
    long unsigned int m_first_frameid;
    ORB_SLAM2::Map* mpOrbSlamMap;
	void set_finished(bool status);

};
