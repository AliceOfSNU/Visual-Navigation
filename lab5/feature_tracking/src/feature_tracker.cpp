#include "feature_tracker.h"

#include <vector>
#include <numeric>
#include <cassert>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <ros/ros.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  16.485 - Fall 2021  - Lab 5 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement a generic feature tracker base class
//  which you will augment with several derived classes for SIFT, SURF, ORB, and
//  FAST+BRIEF feature tracking.
//
// NOTE: Deliverables for the TEAM portion of this assignment start at number 3
// and end at number 7. If you have completed the parts labeled Deliverable 3-7,
// you are done with the TEAM portion of the lab. Deliverables 1-2 are
// individual.

using namespace cv;
using namespace cv::xfeatures2d;

/** TODO This is the main tracking function, given two images, it detects,
 * describes and matches features.
 * We will be modifying this function incrementally to plot different figures
 * and compute different statistics.
 @param[in] img_1, img_2 Images where to track features.
 @param[out] matched_kp_1_kp_2 pair of vectors of keypoints with the same size
 so that matched_kp_1_kp_2.first[i] matches with matched_kp_1_kp_2.second[i].
*/
void FeatureTracker::trackFeatures(const cv::Mat &img_1,
                                   const cv::Mat &img_2,
                                   std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> *matched_kp_1_kp_2) {

  
  vector<KeyPoint> keypoints_1, keypoints_2; 
  detectKeypoints(img_1, &keypoints_1);
  detectKeypoints(img_2, &keypoints_2);

  cv::Mat desc_1, desc_2; //size [kpt_count * 128]
  describeKeypoints(img_1, &keypoints_1, &desc_1);
  describeKeypoints(img_2, &keypoints_2, &desc_2);

  std::vector<std::vector<DMatch>> matches;
  std::vector<DMatch> good_matches;

  //assert(desc_1.size()==desc_2.size() && desc_1.size[0] > 0);
  matchDescriptors(desc_1, desc_2, &matches, &good_matches);
  cv::Mat match_img;

#if 0 //DRAW
  cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, match_img);
  cv::imshow("matches", match_img);
  while (ros::ok() && waitKey(10) == -1) {} //draw
#endif

  vector<uchar> mask;
  vector<KeyPoint> mkpts_1(good_matches.size()), mkpts_2(good_matches.size());
  for(int i = 0; i < good_matches.size(); ++i){
    mkpts_1[i] = keypoints_1[good_matches[i].queryIdx];
    mkpts_2[i] = keypoints_2[good_matches[i].trainIdx];
  }
  matched_kp_1_kp_2->first = mkpts_1, matched_kp_1_kp_2->second = mkpts_2;  
  assert(matched_kp_1_kp_2 != nullptr);
  //inlierMaskComputation(mkpts_1, mkpts_2, OUT &mask);

  
#if 0 //draw
  vector<char> cmask{mask.begin(), mask.end()}; 
  cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, match_img, 
    cv::Scalar(0, 255, 0), cv::Scalar::all(-1), cmask, 
    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
  );
  std::transform(cmask.begin(), cmask.end(), cmask.begin(), [](const char& x) {return !x;});
  cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, match_img, 
    cv::Scalar(0, 0, 255), cv::Scalar::all(-1), cmask, 
    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG
  );
  cv::imshow("matches", match_img);
  while (ros::ok() && waitKey(10) == -1) {} 
#endif

  unsigned int num_inliers = 0;
  for(uchar x: mask) num_inliers += x;

  float const new_num_samples = static_cast<float>(num_samples_) + 1.0f;
  float const old_num_samples = static_cast<float>(num_samples_);
  avg_num_keypoints_img1_ = (avg_num_keypoints_img1_ * old_num_samples + static_cast<float>(keypoints_1.size())) / new_num_samples;
  avg_num_keypoints_img2_ = (avg_num_keypoints_img2_ * old_num_samples + static_cast<float>(keypoints_2.size())) / new_num_samples;
  avg_num_matches_ = (avg_num_matches_ * old_num_samples + static_cast<float>(matches.size())) / new_num_samples;
  avg_num_good_matches_ = (avg_num_good_matches_ * old_num_samples + static_cast<float>(good_matches.size())) / new_num_samples;
  avg_num_inliers_ = (avg_num_inliers_ * old_num_samples + static_cast<float>(num_inliers)) / new_num_samples;
  avg_inlier_ratio_ =
      (avg_inlier_ratio_ * old_num_samples + (static_cast<float>(num_inliers) / static_cast<float>(good_matches.size()))) / new_num_samples;
  ++num_samples_;
  
}

/** Compute Inlier Mask out of the given matched keypoints.
   *  Both keypoints_1 and keypoints_2 input parameters must be ordered by match
   * i.e. keypoints_1[0] has been matched to keypoints_2[0].
   * Therefore, both keypoints vectors must have the same length.
    @param[in] keypoints_1 List of keypoints detected on the first image.
    @param[in] keypoints_2 List of keypoints detected on the second image.
    @param[out] inlier_mask Mask indicating inliers (1) from outliers (0).
  */
void FeatureTracker::inlierMaskComputation(const std::vector<KeyPoint> &keypoints_1,
                                           const std::vector<KeyPoint> &keypoints_2,
                                           std::vector<uchar> *inlier_mask) const {
  CHECK_NOTNULL(inlier_mask);
  const size_t size = keypoints_1.size();
  CHECK_EQ(keypoints_2.size(), size) << "Size of keypoint vectors "
                                        "should be the same!";

  std::vector<Point2f> pts1(size);
  std::vector<Point2f> pts2(size);
  for (size_t i = 0; i < keypoints_1.size(); i++) {
    pts1[i] = keypoints_1[i].pt;
    pts2[i] = keypoints_2[i].pt;
  }

  static constexpr double max_dist_from_epi_line_in_px = 3.0;
  static constexpr double confidence_prob = 0.99;
  try {
    findFundamentalMat(pts1, pts2, CV_FM_RANSAC, max_dist_from_epi_line_in_px, confidence_prob, *inlier_mask);
  } catch (...) {
    ROS_WARN("Inlier Mask could not be computed, this can happen if there"
             "are not enough features tracked.");
  }
}

void FeatureTracker::drawMatches(const cv::Mat &img_1,
                                 const cv::Mat &img_2,
                                 const std::vector<KeyPoint> &keypoints_1,
                                 const std::vector<KeyPoint> &keypoints_2,
                                 const std::vector<std::vector<DMatch>> &matches) {
  cv::namedWindow("tracked_features", cv::WINDOW_NORMAL);
  cv::Mat img_matches;
  cv::drawMatches(img_1,
                  keypoints_1,
                  img_2,
                  keypoints_2,
                  matches,
                  img_matches,
                  Scalar::all(-1),
                  Scalar::all(-1),
                  std::vector<std::vector<char>>(),
                  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  // Show detected matches
  imshow("tracked_features", img_matches);

  // Wait indefinitely for key pressed, but allow ROS to also kill everything.
  // Otherwise ROS will not die unless we close the window.
  // Use this when wanting to visualize pairs of images, one at a time.
  while (ros::ok() && waitKey(10) == -1) {} // Instead of using waitKey(0);

  // Alternatively, just wait for some seconds. Use this when playing with video
  // sequences.
  // waitKey(10);
}

void FeatureTracker::printStats() const {
  std::cout << "Avg. Keypoints 1 Size: " << avg_num_keypoints_img1_ << std::endl;
  std::cout << "Avg. Keypoints 2 Size: " << avg_num_keypoints_img2_ << std::endl;
  std::cout << "Avg. Number of matches: " << avg_num_matches_ << std::endl;
  std::cout << "Avg. Number of good matches: " << avg_num_good_matches_ << std::endl;
  std::cout << "Avg. Number of Inliers: " << avg_num_inliers_ << std::endl;
  std::cout << "Avg. Inliers ratio: " << avg_inlier_ratio_ << std::endl;
  std::cout << "Num. of samples: " << num_samples_ << std::endl;
}
