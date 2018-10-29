#include "KeyFrameQueue.h"

using namespace std;

void KeyFrameQueue::enqueue(OrbKeyFrame & frame)
{
	unique_lock<mutex> lock(m_mutex);
	m_queue.push(frame);
}

OrbKeyFrame KeyFrameQueue::dequeue()
{
	unique_lock<mutex> lock(m_mutex);
	OrbKeyFrame frame = m_queue.front();
	m_queue.pop();
	return frame;
}

size_t KeyFrameQueue::size()
{
	unique_lock<mutex> lock(m_mutex);
	return m_queue.size();
}
