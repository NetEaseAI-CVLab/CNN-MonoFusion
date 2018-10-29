#pragma once
#include "DenseMap.h"
#include <pangolin/pangolin.h>
#include <string>

class DenseMapDrawer
{
public:
	DenseMapDrawer(DenseMap* dense_map, std::string setting_file);
    void get_current_opengl_camera_matrix(pangolin::OpenGlMatrix & M, cv::Mat &cv_Tc);
	void draw_dense_points(vector<cv::Vec3f> &points, bool draw_label, bool draw_single_frame, bool draw_color);
	//void draw_depth_map();


    void set_system_status(bool _status)
    {
        m_dense_map->set_system_running_status(_status);
    }

protected:
	DenseMap* m_dense_map;
	float m_point_size;
};
