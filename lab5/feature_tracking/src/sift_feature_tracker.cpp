
#include "sift_feature_tracker.h"
#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

/**
   Sift feature tracker Constructor.
*/
SiftFeatureTracker::SiftFeatureTracker()
  : FeatureTracker(),
    detector(SIFT::create()) {}

void SiftFeatureTracker::detectKeypoints(const cv::Mat& img,
                                         std::vector<KeyPoint>* keypoints) const {
  CHECK_NOTNULL(keypoints);
  detector->detect(img, OUT *keypoints);
}

void SiftFeatureTracker::describeKeypoints(const cv::Mat& img,
                                           std::vector<KeyPoint>* keypoints,
                                           cv::Mat* descriptors) const {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  detector->compute(img, *keypoints, OUT *descriptors);

  if(descriptors->type() != CV_32F){
    descriptors->convertTo(*descriptors, CV_32F);
  }
}

void SiftFeatureTracker::matchDescriptors(
                                          const cv::Mat& descriptors_1,
                                          const cv::Mat& descriptors_2,
                                          std::vector<std::vector<DMatch>>* matches,
                                          std::vector<cv::DMatch>* good_matches) const {
  CHECK_NOTNULL(matches);

  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  cout << descriptors_1.type() << descriptors_1.size() << endl;
  cout << descriptors_2.type() << descriptors_2.size() << endl;
  try{
    matcher->knnMatch(descriptors_1, descriptors_2, *matches, 2);
  }catch(cv::Exception& e){
    cout << e.what() << endl;
  }

  for(vector<DMatch> match: *matches){
    if(match.size() == 1 || match[0].distance < 0.8 * match[1].distance){
      //good match if not ambiguous, or only one is matched
      good_matches->push_back(match[0]);
    }
  }
}
