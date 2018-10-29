#pragma once


#include<WinSock2.h>
#include<string>
#include<opencv2\core.hpp>
#include<mutex>
#include "KeyFrameQueue.h"
#include "DenseMap.h"

class DepthPredictCommunicator
{
public:
    DepthPredictCommunicator(std::string setting_file, int is_use_dataset_depth);
	~DepthPredictCommunicator();
    bool predict(cv::Mat &imgColor, cv::Mat& depth);
    bool getNetworkPredictDepth(cv::Mat &imgColor, cv::Mat & depth);
protected:
	//cv::Size	m_server_output_size;
	int			m_img_width;
	int			m_img_height;
    int			m_net_width;
    int			m_net_height;
	float		m_focal_length;
    float       m_depth_grad_thr;
    float       m_focal_scale;

	std::string	m_server_ip;
	int			m_server_port;

	SOCKET		m_socket;

	cv::Mat		m_depth;

	bool		m_success;



	SOCKET connect_server(std::string ip, ushort port);
	void disconnect_server(SOCKET sock);

	void send_number(int data, SOCKET sock);
	void send_number(float data, SOCKET sock);
	void send_image(cv::Mat& img, SOCKET sock, std::string encode = ".png");
	int recv_depth(SOCKET sock, cv::Mat& disp);
	int recv_all(SOCKET sock, char* buf, int length, int flags);
	int send_all(SOCKET sock, char* buf, int length, int flags);

};
