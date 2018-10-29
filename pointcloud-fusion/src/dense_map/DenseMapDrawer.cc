#include "DenseMapDrawer.h"

using namespace std;
using namespace cv;

DenseMapDrawer::DenseMapDrawer(DenseMap* dense_map, std::string setting_file):
	m_dense_map(dense_map)
{
	cv::FileStorage settings(setting_file, cv::FileStorage::READ);
	m_point_size = settings["Viewer.PointSize"];
	settings.release();
}

void DenseMapDrawer::get_current_opengl_camera_matrix(pangolin::OpenGlMatrix & M, cv::Mat &cv_Tcw)
{

	cv::Mat Rwc(3, 3, CV_32F);
	cv::Mat twc(3, 1, CV_32F);

	m_dense_map->get_current_Tcw(cv_Tcw);
	if (cv_Tcw.empty())
	{
		M.SetIdentity();
		return;
	}

	Rwc = cv_Tcw.rowRange(0, 3).colRange(0, 3).t();
	twc = -Rwc*cv_Tcw.rowRange(0, 3).col(3);

	M.m[0] = Rwc.at<float>(0, 0);
	M.m[1] = Rwc.at<float>(1, 0);
	M.m[2] = Rwc.at<float>(2, 0);
	M.m[3] = 0.0;

	M.m[4] = Rwc.at<float>(0, 1);
	M.m[5] = Rwc.at<float>(1, 1);
	M.m[6] = Rwc.at<float>(2, 1);
	M.m[7] = 0.0;

	M.m[8] = Rwc.at<float>(0, 2);
	M.m[9] = Rwc.at<float>(1, 2);
	M.m[10] = Rwc.at<float>(2, 2);
	M.m[11] = 0.0;

	M.m[12] = twc.at<float>(0);
	M.m[13] = twc.at<float>(1);
	M.m[14] = twc.at<float>(2);
	M.m[15] = 1.0;
}

void DenseMapDrawer::draw_dense_points(vector<cv::Vec3f> &points, bool draw_label, bool draw_single_frame, bool draw_color)
{

	//vector<Vec3f> points;
	vector<Vec3b> colors;
	m_dense_map->get_dense_points(points, colors, draw_label, draw_single_frame, draw_color);
	CV_Assert(points.size() == colors.size());

	if (points.empty()) return;

	glPointSize(m_point_size);
	glBegin(GL_POINTS);
	glColor3f(0.0, 0.0, 0.0);

	for (size_t i = 0; i < points.size(); i++)
	{	
		Vec3f& point = points[i];
		Vec3b& color = colors[i];
		float r = min(color[0] / 255.0f, 1.0f);
		float g = min(color[1] / 255.0f, 1.0f);
		float b = min(color[2] / 255.0f, 1.0f);
		glColor3f(r, g, b);		// glColor3b has some bugs to draw wrong colors.
		glVertex3f(point[0], point[1], point[2]);
	}
	glEnd();
}

//void DenseMapDrawer::draw_depth_map()
//{
//
//}
