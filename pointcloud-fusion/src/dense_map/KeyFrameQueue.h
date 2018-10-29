#pragma once

#include <mutex>
#include <queue>
#include "OrbKeyFrame.h"

class KeyFrameQueue
{
public:
	void enqueue(OrbKeyFrame& frame);
	OrbKeyFrame dequeue();
	size_t size();

protected:
	std::mutex m_mutex;
	std::queue<OrbKeyFrame> m_queue;
};
