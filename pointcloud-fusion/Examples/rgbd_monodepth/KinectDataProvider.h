#pragma once

#include <Kinect.h>
#include <opencv2/core.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>

class KinectDataProvider
{
public:
	static const int ORIGIN_COLOR_WIDTH = 1920;
	static const int ORIGIN_COLOR_HEIGHT = 1080;
	static const int ORIGIN_DEPTH_WIDTH = 512;
	static const int ORIGIN_DEPTH_HEIGHT = 424;
	static const int MAX_CACHED_FRAMES = 1;	// large cache will cause large delay!

	struct AlignedFrame
	{
		AlignedFrame() = default;
		AlignedFrame(cv::Mat& _color, cv::Mat& _depth);

		cv::Mat color;
		cv::Mat depth;
	};

private:
	IKinectSensor* sensor_;             // Kinect sensor
	IMultiSourceFrameReader* reader_;   // Kinect data source
	ICoordinateMapper* mapper_;         // Converts between depth, color, and 3d coordinates
	bool initialized_;
	bool started_;
	std::atomic<bool> should_stop_;

	std::queue<AlignedFrame> aligned_frames_;
	cv::Mat origin_color_;
	cv::Mat origin_depth_;
	
	cv::Mat color2depth_;
	cv::Mat resized_color2depth_;
	cv::Size required_size_;

	std::thread frame_process_thread_;
	std::mutex mutex_;

public:
	KinectDataProvider();
	~KinectDataProvider();
	bool initialize();
	void start(cv::Size required_size = cv::Size(0, 0));
	void stop();
	bool is_started();
	bool get_aligned_frame(cv::Mat& color, cv::Mat& depth);

private:
	void release();
	void process_frame();
	bool get_origin_depth(IMultiSourceFrame* frame);
	bool get_origin_color(IMultiSourceFrame* frame);
	void align_frame_and_resize(cv::Mat& out_aligned_color, cv::Mat& out_aligned_depth);

	void push_frame(cv::Mat& color, cv::Mat& depth);
	bool pop_frame(cv::Mat& color, cv::Mat& depth);
};
