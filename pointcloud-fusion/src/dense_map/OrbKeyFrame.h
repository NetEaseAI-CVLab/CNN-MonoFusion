#pragma once
#include <opencv2\core.hpp>
#include <map>
#include <iostream>

//#define WITH_LABEL

struct FeatureUV
{
    int u;
    int v;
    FeatureUV() :u(0), v(0)
    {
    }

    FeatureUV(int _u, int _v) : u(_u), v(_v)
    {
    }

    
    bool operator<(const FeatureUV& other_feature) const
    {
        if(this->u < other_feature.u)
        {
            return true;
        }
        else if(this->u == other_feature.u)
        {
            if(this->v < other_feature.v)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
        return false;
    }
};

struct MapPoint
{
    int idx;
    cv::Vec3f point;
    float uncertainty;
    bool isStable;
    uchar fusion_weight;
    #ifdef WITH_LABEL
    uchar label;
    int16_t label_confidence;
    cv::Vec3b label_color;
    #endif

    bool operator<(const MapPoint& other_feature) const
    {
        if(this->idx < other_feature.idx) return true;
        else return false;
    }
};

class OrbKeyFrame
{
public:
    OrbKeyFrame()
    {
    }

    cv::Mat K;
    cv::Mat Tcw;
    cv::Mat image;
    cv::Mat depth;
    bool isKeyframe;
    
    cv::Mat uncertainty_map;
    int frame_idx;
    float bf;  // baseline * focal_length

    std::map<FeatureUV, int> map_featureuv_pointidx;

    void addMapPoint(FeatureUV &uvcoord, int point_idx)
    {
        // if key exist, will not insert
        //map_uvcoord_featureidx.insert(pair<FeatureUV, int>(uvcoord, feature_idx));
        //map_featureidx_pointidx.insert(pair<int, int>(feature_idx, point_idx));

        // if key exist, will overwrite
        map_featureuv_pointidx[uvcoord] = point_idx;
    }

    bool getMapPointIdx(const FeatureUV &uvcoord, int &pointidx) const
    {
        auto iter1 = map_featureuv_pointidx.find(uvcoord);
        if(iter1 != map_featureuv_pointidx.end())
        {
            pointidx = iter1->second;
            return true;
        }

        return false;
    }

    size_t getFeatureuvSize()
    {
        return map_featureuv_pointidx.size();
    }

    #ifdef WITH_LABEL
    cv::Mat label;
    OrbKeyFrame(cv::Mat& _image, cv::Mat &_depth, cv::Mat &_label, cv::Mat& _K, cv::Mat& _Tcw, float _bf, float _idx, bool _isKeyframe)
    {
        //image = _image.clone();		// deep copy
        image = _image;					// shallow copy
        depth = _depth;
        label = _label;
        K = _K.clone();
        Tcw = _Tcw.clone();
        bf = _bf;
        frame_idx = _idx;
        uncertainty_map = cv::Mat::zeros(image.rows, image.cols, CV_32F);
        isKeyframe = _isKeyframe;
    }
    #else
    OrbKeyFrame(cv::Mat& _image, cv::Mat &_depth, cv::Mat& _K, cv::Mat& _Tcw, float _bf, float _idx, bool _isKeyframe)
    {
        //image = _image.clone();		// deep copy
        image = _image.clone();					// shallow copy
        depth = _depth.clone();
        K = _K.clone();
        Tcw = _Tcw.clone();
        bf = _bf;
        frame_idx = _idx;
        uncertainty_map = cv::Mat::zeros(image.rows, image.cols, CV_32F);
        isKeyframe = _isKeyframe;
    }
    #endif

private:

};

