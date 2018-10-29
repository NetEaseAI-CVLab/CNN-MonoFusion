#include "KinectDataProvider.h"
#include <opencv2/imgproc.hpp>

KinectDataProvider::KinectDataProvider():
	sensor_(nullptr), reader_(nullptr), mapper_(nullptr),
	initialized_(false), should_stop_(false), started_(false)
{
}

KinectDataProvider::~KinectDataProvider()
{
	stop();
	release();
}

bool KinectDataProvider::initialize()
{
	if (initialized_) return true;

	if (FAILED(GetDefaultKinectSensor(&sensor_))) {
		sensor_ = nullptr;
		initialized_ = false;
		return false;
	}

	if (sensor_) {

		sensor_->OpenMultiSourceFrameReader(
			FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
			&reader_);
		sensor_->get_CoordinateMapper(&mapper_);
		
		if (reader_ && mapper_)
		{
			initialized_ = true;
			return true;
		}
		else
		{
			release();
		}
	}
	initialized_ = false;
	return false;
}

void KinectDataProvider::start(cv::Size required_size)
{
	if (!initialized_ || !sensor_ || started_) return;

	if (required_size.width == 0 || required_size.height == 0)
	{
		required_size_ = cv::Size(ORIGIN_COLOR_WIDTH, ORIGIN_COLOR_HEIGHT);
	}
	else
	{
		required_size_ = required_size;
	}

	sensor_->Open();
	should_stop_ = false;
	started_ = true;
	frame_process_thread_ = std::thread(&KinectDataProvider::process_frame, this);
}

void KinectDataProvider::stop()
{
	if (!started_) return;

	sensor_->Close();
	should_stop_ = true;
	frame_process_thread_.join();
	started_ = false;
}

bool KinectDataProvider::is_started()
{
	return started_;
}

bool KinectDataProvider::get_aligned_frame(cv::Mat & color, cv::Mat & depth)
{
	return pop_frame(color, depth);
	
	/*IMultiSourceFrame* frame = nullptr;
	reader_->AcquireLatestFrame(&frame);
	if (!frame) return false;
	
	if (get_origin_depth(frame) && get_origin_color(frame))
	{
		color2depth_.create(origin_color_.size(), CV_32FC2);
		mapper_->MapColorFrameToDepthSpace(origin_depth_.total(), origin_depth_.ptr<UINT16>(), origin_color_.total(), (DepthSpacePoint*)color2depth_.data);
		align_frame_and_resize(color, depth, requied_size);

		frame->Release();
		return true;
	}

	frame->Release();
	return false;*/
}

void KinectDataProvider::release()
{
	if (reader_)
	{
		reader_->Release();
		reader_ = nullptr;
	}

	if (mapper_)
	{
		mapper_->Release();
		mapper_ = nullptr;
	}

	if (sensor_)
	{
		sensor_->Close();
		sensor_->Release();
		sensor_ = nullptr;
	}
	initialized_ = false;
}

void KinectDataProvider::process_frame()
{
	IMultiSourceFrame* frame = nullptr;

	while (!should_stop_)
	{
		{
			std::unique_lock<std::mutex> lock(mutex_);
			if (aligned_frames_.size() >= MAX_CACHED_FRAMES)
				continue;
		}

		reader_->AcquireLatestFrame(&frame);
		if (!frame) continue;

		if (get_origin_depth(frame) && get_origin_color(frame))
		{
			color2depth_.create(origin_color_.size(), CV_32FC2);
			mapper_->MapColorFrameToDepthSpace(origin_depth_.total(), origin_depth_.ptr<UINT16>(), origin_color_.total(), (DepthSpacePoint*)color2depth_.data);

			cv::Mat color, depth;
			align_frame_and_resize(color, depth);
			push_frame(color, depth);
		}

		frame->Release();
		frame = nullptr;
	}
}

bool KinectDataProvider::get_origin_depth(IMultiSourceFrame* frame)
{
	IDepthFrame* depth_frame;
	IDepthFrameReference* frame_ref = nullptr;

	frame->get_DepthFrameReference(&frame_ref);
	if (!frame_ref) return false;
	frame_ref->AcquireFrame(&depth_frame);
	frame_ref->Release();
	if (!depth_frame) return false;

	origin_depth_.create(ORIGIN_DEPTH_HEIGHT, ORIGIN_DEPTH_WIDTH, CV_16UC1);
	UINT buffer_size = origin_depth_.total();	// use the element count, not the size in bytes.
	HRESULT res = depth_frame->CopyFrameDataToArray(buffer_size, origin_depth_.ptr<UINT16>());
	depth_frame->Release();

	if (FAILED(res))
		return false;
	return true;
}

bool KinectDataProvider::get_origin_color(IMultiSourceFrame* frame)
{
	IColorFrame* color_frame;
	IColorFrameReference* frame_ref = nullptr;

	frame->get_ColorFrameReference(&frame_ref);
	if (!frame_ref) return false;
	frame_ref->AcquireFrame(&color_frame);
	frame_ref->Release();
	if (!color_frame) return false;

	origin_color_.create(ORIGIN_COLOR_HEIGHT, ORIGIN_COLOR_WIDTH, CV_8UC4);
	UINT buffer_size = origin_color_.step[0] * origin_color_.rows;	// use size in bytes here.
	HRESULT res = color_frame->CopyConvertedFrameDataToArray(buffer_size, origin_color_.data, ColorImageFormat_Bgra);
	color_frame->Release();

	if (FAILED(res))
		return false;
	return true;
}

void KinectDataProvider::align_frame_and_resize(cv::Mat& out_aligned_color, cv::Mat& out_aligned_depth)
{
	cv::resize(origin_color_, out_aligned_color, required_size_);
	cv::resize(color2depth_, resized_color2depth_, required_size_);

	out_aligned_depth.create(required_size_, origin_depth_.type());
	cv::remap(origin_depth_, out_aligned_depth, resized_color2depth_, cv::noArray(), CV_INTER_LINEAR);
}

void KinectDataProvider::push_frame(cv::Mat & color, cv::Mat & depth)
{
	std::unique_lock<std::mutex> lock(mutex_);

	aligned_frames_.emplace(color, depth);
}

bool KinectDataProvider::pop_frame(cv::Mat & color, cv::Mat & depth)
{
	std::unique_lock<std::mutex> lock(mutex_);

	if (aligned_frames_.size() == 0) return false;

	color = aligned_frames_.front().color;
	depth = aligned_frames_.front().depth;
	aligned_frames_.pop();
	return true;
}

KinectDataProvider::AlignedFrame::AlignedFrame(cv::Mat & _color, cv::Mat & _depth)
{
	color = _color;
	depth = _depth;
}
