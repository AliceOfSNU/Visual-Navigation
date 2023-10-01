#include "lk_feature_tracker.h"

#include <numeric>
#include <vector>
#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

using namespace cv;
using namespace cv::xfeatures2d;

/**
   LK feature tracker Constructor.
*/
LKFeatureTracker::LKFeatureTracker() {
  // Feel free to modify if you want!
  cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
}

LKFeatureTracker::~LKFeatureTracker() {
  // Feel free to modify if you want!
  cv::destroyWindow(window_name_);
}


/**
 @param[in] frame Current image frame
*/
void LKFeatureTracker::trackFeatures(const cv::Mat& frame) {

  cv::Mat frame_gry;
  cv::cvtColor( frame, frame_gry, cv::COLOR_BGR2GRAY );

  const int max_corners = 1<<7;
  //consider using corner subpixel to get even more accurate results.
  
  vector<cv::Mat> curr_pyr;
  cv::Size winsize = cv::Size(21, 21);
  cv::buildOpticalFlowPyramid(frame_gry, curr_pyr, winsize, 3, true );

  if(num_samples_ < 1){
    //init.
    prev_frame_ = frame;
    prev_frame_gry_ = frame_gry;
    prev_pyr_ = curr_pyr;
    prev_corners_.reserve(max_corners); //still empty, though.
    cv::goodFeaturesToTrack(frame_gry, OUT prev_corners_, max_corners, 0.02, 5, noArray(), 3, false);
    tracks_overlay_ = frame * 0.0;
  }

  // detection
  cv::Mat err; vector<uchar> ismatched;
  vector<cv::Point2f> corners;
  corners.reserve(max_corners);

  if(prev_corners_.size() > 0){
    cv::calcOpticalFlowPyrLK(prev_pyr_, curr_pyr, prev_corners_, OUT corners, OUT ismatched, OUT err, winsize);
  }

  // matching in O(N)
  int match_cnt = 0;
  for(int i = 0; i < ismatched.size(); ++i){
    if(ismatched[i]){
      corners[match_cnt] = corners[i];
      prev_corners_[match_cnt] = prev_corners_[i];
      match_cnt++;
    }
  }
  corners.erase(corners.begin()+match_cnt, corners.end());
  prev_corners_.erase(prev_corners_.begin()+match_cnt, prev_corners_.end());
  
  // outlier rejection
  vector<uchar> mask;
  assert(prev_corners_.size() == corners.size());
  mask.resize(corners.size());
  if(mask.size() > 0){
    inlierMaskComputation(prev_corners_, corners, OUT &mask);
  }
  int num_inliers = 0;
  for(uchar x: mask) num_inliers+=x;


#if 1 //display
  show(frame, prev_corners_, corners, mask);
#endif

  //stats
  float new_num_samples = num_samples_ + 1.0f;
  float old_num_samples = num_samples_;
  avg_num_keypoints_img1_ = (avg_num_keypoints_img1_ * old_num_samples + prev_corners_.size()) / new_num_samples;
  //update prev_corners
  prev_corners_.clear();
  cv::goodFeaturesToTrack(frame_gry, OUT prev_corners_, max_corners, 0.3, 10, noArray(), 3, true);

  avg_num_keypoints_img2_ = (avg_num_keypoints_img2_ * old_num_samples + prev_corners_.size()) / new_num_samples;
  avg_num_matches_ = (avg_num_matches_ * old_num_samples + match_cnt) / new_num_samples;
  avg_num_inliers_ = (avg_num_inliers_ * old_num_samples + num_inliers) / new_num_samples;
  avg_inlier_ratio_ =
      (avg_inlier_ratio_ * old_num_samples + num_inliers) / new_num_samples;
  num_samples_++;

  if(num_samples_%100 == 0){
    printStats();
  }
  // assignment does no copy.
  prev_pyr_ = curr_pyr; 
  prev_frame_ = frame;
  prev_frame_gry_ = frame_gry;
  
}

void LKFeatureTracker::printStats() const {
    std::cout << "Avg. Keypoints 1 Size: " << avg_num_keypoints_img1_ << std::endl;
    std::cout << "Avg. Keypoints 2 Size: " << avg_num_keypoints_img2_ << std::endl;
    std::cout << "Avg. Number of matches: " << avg_num_matches_ << std::endl;
    std::cout << "Avg. Number of Inliers: " << avg_num_inliers_ << std::endl;
    std::cout << "Avg. Inliers ratio: " << avg_inlier_ratio_ << std::endl;
    std::cout << "Num. of samples: " << num_samples_ << std::endl;
  }

/** TODO Display image with tracked features from prev to curr on the image
 * corresponding to 'frame'
 * @param[in] frame The current image frame, to draw the feature track on
 * @param[in] prev The previous set of keypoints
 * @param[in] curr The set of keypoints for the current frame. keypoints of the same index are matched.
 */
void LKFeatureTracker::show(const cv::Mat& frame, vector<cv::Point2f>& prev,
                            vector<cv::Point2f>& curr, const vector<uchar>& mask) {
  cv::Mat frame_cpy; frame.copyTo(frame_cpy);
  if(mask.size() > 0){
    for(int i = 0; i < curr.size(); ++i){
      cv::Scalar color = cv::Scalar(0, 255, 0);
      if(mask[i]){
        color = cv::Scalar(0, 255, 0);
      }else{
        color = cv::Scalar(0, 0, 255);
      }
      cv::line(tracks_overlay_, prev[i], curr[i], color, 1);
      cv::circle(frame_cpy, curr[i], 10, color, 1);
    }
  }

  tracks_overlay_ *= 0.95;
  frame_cpy += tracks_overlay_;
  imshow(window_name_, frame_cpy);
  cv::waitKey(10);
}

/** Compute Inlier Mask out of the given matched keypoints.
 @param[in] pts1 List of keypoints detected on the first image.
 @param[in] pts2 List of keypoints detected on the second image.
 @param[out] inlier_mask Mask indicating inliers (1) from outliers (0).
*/
void LKFeatureTracker::inlierMaskComputation(const std::vector<cv::Point2f>& pts1,
                                             const std::vector<cv::Point2f>& pts2,
                                             std::vector<uchar>* inlier_mask) const {
  CHECK_NOTNULL(inlier_mask);

  static constexpr double max_dist_from_epi_line_in_px = 3.0;
  static constexpr double confidence_prob = 0.99;
  try {
    findFundamentalMat(pts1, pts2, CV_FM_RANSAC,
                       max_dist_from_epi_line_in_px, confidence_prob,
                       *inlier_mask);
  } catch(...) {
    ROS_WARN("Inlier Mask could not be computed, this can happen if there"
             "are not enough features tracked.");
  }
}
