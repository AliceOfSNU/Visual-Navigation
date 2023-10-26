// orb cv
#include "orb_feature_tracker.h"
#include <opencv2/features2d.hpp>

//add
#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <cassert>

using namespace cv;
using namespace cv::xfeatures2d;

OrbFeatureTracker::OrbFeatureTracker()
    : detector(ORB::create(400,//n_features
		1.2f,//scalefactor
		8//n_levels
    )) {}

void OrbFeatureTracker::detectKeypoints(const cv::Mat &img,
                                         std::vector<cv::KeyPoint> *keypoints) const {
    detector->detect(img, *keypoints);
}

void OrbFeatureTracker::detectKeypoints(const cv::Mat &img,
                                         std::vector<cv::KeyPoint> *keypoints,
                                         const cv::Mat& mask) const{
  detector->detect(img, *keypoints, mask);
}

void OrbFeatureTracker::describeKeypoints(const cv::Mat &img,
                                           std::vector<cv::KeyPoint> *keypoints,
                                           cv::Mat *descriptors) const {
    detector->compute(img, *keypoints, *descriptors);
}

void OrbFeatureTracker::matchDescriptors(const cv::Mat &descriptors_1,
                                          const cv::Mat &descriptors_2,
                                          std::vector<std::vector<cv::DMatch>> *matches,
                                          std::vector<cv::DMatch> *good_matches) const {


  //using brute force matcher with HAMMING distance as recommended
  //hamming is the count of equal elements in each vector
  cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
  matcher.match(descriptors_1, descriptors_2, *good_matches);

}
