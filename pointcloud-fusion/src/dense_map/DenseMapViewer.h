#pragma once

#include"DenseMapDrawer.h"
#include <thread>                // std::thread
#include <mutex>                // std::mutex, std::unique_lock
class Plane
{
public:
    Plane(const std::vector<cv::Vec3f> &vMPs, const cv::Mat &Tcw);
    Plane(const float &nx, const float &ny, const float &nz, const float &ox, const float &oy, const float &oz);

    void Recompute();

    //normal
    cv::Mat n;
    //origin
    cv::Mat o;
    //arbitrary orientation along normal
    float rang;
    //transformation from world to the plane
    cv::Mat Tpw;
    pangolin::OpenGlMatrix glTpw;
    //MapPoints that define the plane
    std::vector<cv::Vec3f> mvMPs;
    //camera pose when the plane was first observed (to compute normal direction)
    cv::Mat mTcw, XC;
};

class DenseMapViewer
{
public:
	DenseMapViewer(DenseMapDrawer* drawer, std::string setting_file);
	void run();
	void request_finish();
	bool should_finish();
	bool is_finished();
    bool system_running_status()
    {
        return system_running;
    }

protected:
	void set_finished(bool status);
   
    Plane* DetectPlane(const cv::Mat Tcw, std::vector<cv::Vec3f> &vMPs, const int iterations);
    void DrawCube(const float &size, const float x = 0, const float y = 0, const float z = 0);
    void DrawPlane(int ndivs, float ndivsize);
    void DrawPlane(Plane* pPlane, int ndivs, float ndivsize);
	bool m_should_finish;
	bool m_is_finished;
	std::mutex m_mutex;
	DenseMapDrawer* m_map_drawer;

	float m_viewpoint_x;
	float m_viewpoint_y;
	float m_viewpoint_z;
	float m_viewpoint_f;

    bool system_running;
};

