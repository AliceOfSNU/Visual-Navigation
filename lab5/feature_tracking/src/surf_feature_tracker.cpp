#include "surf_feature_tracker.h"
#include "defines.h"
#include <vector>
#include <glog/logging.h>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


using namespace cv;
using namespace cv::xfeatures2d;

/**
   Surf feature tracker Constructor.
*/
SurfFeatureTracker::SurfFeatureTracker()
  : FeatureTracker(),
    detector(SURF::create()) {

}

/** TODO: this function detects keypoints in an image.
    @param[in] img Image input where to detect keypoints.
    @param[out] keypoints List of keypoints detected on the given image.
*/
void SurfFeatureTracker::detectKeypoints(const cv::Mat& img,
                                         std::vector<KeyPoint>* keypoints) const {
  CHECK_NOTNULL(keypoints);
  detector->detect(img, OUT *keypoints);
}

/** TODO: this function describes keypoints in an image.
    @param[in] img Image used to detect the keypoints.
    @param[in, out] keypoints List of keypoints detected on the image. Depending
    on the detector used some keypoints might be added or removed.
    @param[out] descriptors List of descriptors for the given keypoints.
*/
void SurfFeatureTracker::describeKeypoints(const cv::Mat& img,
                                           std::vector<KeyPoint>* keypoints,
                                           cv::Mat* descriptors) const {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  detector->compute(img, *keypoints, OUT *descriptors);
}

/** TODO: this function matches descriptors.
    @param[in] descriptors_1 First list of descriptors.
    @param[in] descriptors_2 Second list of descriptors.
    @param[out] matches List of k best matches between descriptors.
    @param[out] good_matches List of descriptors classified as "good"
*/
void SurfFeatureTracker::matchDescriptors(
                                          const cv::Mat& descriptors_1,
                                          const cv::Mat& descriptors_2,
                                          std::vector<std::vector<DMatch>>* matches,
                                          std::vector<cv::DMatch>* good_matches) const {
  CHECK_NOTNULL(matches);
  FlannBasedMatcher flann_matcher;
  flann_matcher.knnMatch(descriptors_1, descriptors_2, *matches, 2);

  for(vector<DMatch> match: *matches){
    if(match.size() == 1 || match[0].distance < 0.8 * match[1].distance){
      //good match if not ambiguous, or only one is matched
      good_matches->push_back(match[0]);
    }
  }

}
