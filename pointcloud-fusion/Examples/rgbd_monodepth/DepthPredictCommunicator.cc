#include "DepthPredictCommunicator.h"
#include<iostream>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgproc.hpp>
#pragma comment(lib, "ws2_32.lib")

using namespace cv;

DepthPredictCommunicator::DepthPredictCommunicator(std::string setting_file, int is_use_dataset_depth):
	m_success(false)
{
	FileStorage fs(setting_file, FileStorage::READ);
	m_server_ip = fs["Server.IP"];
	m_server_port = fs["Server.Port"];
	m_img_width = fs["Camera.width"];
	m_img_height = fs["Camera.height"];
	m_focal_length = fs["Camera.fx"];
    m_focal_scale = fs["Camera.focal_scale"];
    m_depth_grad_thr = fs["Camera.depth_grad_thr"];
    m_net_width = fs["Camera.net_width"];
    m_net_height = fs["Camera.net_height"];
    
	fs.release();

    std::cout << "server ip -> " << m_server_ip << std::endl;
    std::cout << "server port -> " << m_server_port << std::endl;
    std::cout << "network input -> " << m_net_width << "x" << m_net_height << std::endl;
    if(!is_use_dataset_depth)
    {	
        std::cout << "\nInit Socket for communication ...." << std::endl;
        //// Startup socket
        WSAData wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
        m_socket = connect_server(m_server_ip, m_server_port);
        if(m_socket == 0)
        {
            std::cout << "Connect socket failed!" << std::endl;
            return;
        }

        std::cout << "Socket init done!" << std::endl;
        send_number(m_depth_grad_thr, m_socket);
        send_number(m_focal_scale, m_socket);
    }
    m_success = true;

	//send_number(m_img_width, m_socket);
	//send_number(m_img_height, m_socket);

	//// Confirm server output size
	//char str_buf[16];
	//// receive width
	//int recv_len = recv_all(m_socket, str_buf, 16, 0);
	//if (recv_len <= 0)
	//{
	//	cout << "Cannot confirm server output size!" << endl;
	//	return;
	//}
	//string str_width(str_buf);
	//int width = stoi(str_width);
	//// receive height
	//recv_len = recv_all(m_socket, str_buf, 16, 0);
	//if (recv_len <= 0)
	//{
	//	cout << "Cannot confirm server output size!" << endl;
	//	return;
	//}
	//string str_height(str_buf);
	//int height = stoi(str_height);
	//m_server_output_size = Size(width, height);
}

DepthPredictCommunicator::~DepthPredictCommunicator()
{
	// Cleanup socket
	disconnect_server(m_socket);
	WSACleanup();
}



bool DepthPredictCommunicator::predict(cv::Mat &imgColor, cv::Mat & depth)
{
	if (!m_success)
	{
        std::cout << "Prediction server unavailable." << std::endl;
		return false;
	}

    // img resize

    cv::Mat  img_resize;
    cv::resize(imgColor, img_resize, Size(m_net_width, m_net_height),0,0, cv::INTER_NEAREST);

	// Send camera pose and keyframe image to server
	send_image(img_resize, m_socket, ".png");
	// Receive disparity from server
	int recv_len = recv_depth(m_socket, depth);
	if (recv_len <= 0)
	{
        std::cout << "Receive disparity failed!" << std::endl;
		return false;
	}

    return true;

	//resize(m_disp, m_scaled_disp, keyframe.image.size(), 0, 0, CV_INTER_NN);
	//m_scaled_disp *= keyframe.image.cols;	// the disparities returned by server is normalized from 0 to 1.
	//
	////spatialGradient(m_scaled_disp, m_dx, m_dy);
	//Sobel(m_scaled_disp, m_dx, CV_32FC1, 1, 0, 3);
	//Sobel(m_scaled_disp, m_dy, CV_32FC1, 0, 1, 3);
	//cv::sqrt(m_dx.mul(m_dx) + m_dy.mul(m_dy), m_gradients);
	//threshold(m_gradients, m_gradients, 1.6, 0, THRESH_TOZERO_INV);
	//threshold(m_gradients, m_gradients, 0, 1, THRESH_BINARY);
	//
	//// Convert disparity to depth map;
	//divide(keyframe.bf, m_scaled_disp, depth);
	//threshold(depth, depth, m_min_depth, 0, THRESH_TOZERO);
	//threshold(depth, depth, m_max_depth, 0, THRESH_TOZERO_INV);
	//depth = depth.mul(m_gradients);
}


SOCKET DepthPredictCommunicator::connect_server(std::string ip, ushort port)
{
	sockaddr_in server;
	hostent *hp;
	in_addr *hipaddr;
	SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
	server.sin_family = AF_INET;
	server.sin_port = htons(port);
	server.sin_addr.s_addr = inet_addr(ip.c_str());
	sockaddr* server_addr = (sockaddr*)&server;
	int addr_length = sizeof(sockaddr_in);
	if (connect(sock, server_addr, addr_length) == INVALID_SOCKET)
	{
		disconnect_server(sock);
		return 0;
	}
	return sock;
}

void DepthPredictCommunicator::disconnect_server(SOCKET sock)
{
	closesocket(sock);
}

void DepthPredictCommunicator::send_number(int data, SOCKET sock)
{
	char str_buf[16];
    std::stringstream ss;
	ss << data;
	std::string str_data = ss.str();
	memset(str_buf, ' ', 16);
	CV_Assert(str_data.length() <= 16);
	memcpy(str_buf, str_data.c_str(), str_data.length());

	send_all(sock, str_buf, 16, 0);
}

void DepthPredictCommunicator::send_number(float data, SOCKET sock)
{
	char str_buf[16];
    std::stringstream ss;
	ss << data;
	std::string str_data = ss.str();
	memset(str_buf, ' ', 16);
	CV_Assert(str_data.length() <= 16);
	memcpy(str_buf, str_data.c_str(), str_data.length());

	send_all(sock, str_buf, 16, 0);
}

void DepthPredictCommunicator::send_image(cv::Mat & img, SOCKET sock, std::string encode)
{
    std::vector<uchar> encode_buf;
	static char str_length_buf[16];

	cv::imencode(encode, img, encode_buf);
	int length = encode_buf.size();
    std::stringstream ss;
	ss << length;
	std::string str_length = ss.str();
	memset(str_length_buf, ' ', 16);
	memcpy(str_length_buf, str_length.c_str(), str_length.length());

	send_all(sock, str_length_buf, 16, 0);
	send_all(sock, (char*)encode_buf.data(), length, 0);
}

int DepthPredictCommunicator::recv_depth(SOCKET sock, cv::Mat & depth)
{
	static char str_buf[16];

	// receive encoded image length
	int recv_len = recv_all(sock, str_buf, 16, 0);
	if (recv_len <= 0) return recv_len;
    std::string str_length(str_buf);
	int length = std::stoi(str_length);

    std::vector<char> encode_buf(length);
	recv_len = recv_all(sock, encode_buf.data(), length, 0);
    depth = imdecode(encode_buf, CV_LOAD_IMAGE_UNCHANGED);
    //cv::resize(recv_depth, depth, Size(m_img_width, m_img_height), CV_INTER_NEAREST);

	return recv_len;

	// receive data
	/*disp.create(m_server_output_size.height, m_server_output_size.width, CV_32FC1);
	recv_len = recv_all(sock, (char*)disp.data, length, 0);
	return recv_len;*/
}

int DepthPredictCommunicator::recv_all(SOCKET sock, char * buf, int length, int flags)
{
	int toread = length;
	char  *bufptr = buf;

	while (toread > 0)
	{
		int rsz = recv(sock, bufptr, toread, flags);
		if (rsz <= 0)
			return rsz;  /* Error or other end closed cnnection */

		toread -= rsz;  /* Read less next time */
		bufptr += rsz;  /* Next buffer position to read into */
	}

	return length;
}

int DepthPredictCommunicator::send_all(SOCKET sock, char * buf, int length, int flags)
{
	int tosend = length;
	char  *bufptr = buf;

	while (tosend > 0)
	{
		int rsz = send(sock, bufptr, tosend, flags);
		if (rsz <= 0)
			return rsz;  /* Error or other end closed cnnection */

		tosend -= rsz;  /* Read less next time */
		bufptr += rsz;  /* Next buffer position to read into */
	}

	return length;
}

bool DepthPredictCommunicator::getNetworkPredictDepth(cv::Mat &imgColor, cv::Mat & depth)
{
   bool recv_ok = predict(imgColor, depth);
   return recv_ok;
}
