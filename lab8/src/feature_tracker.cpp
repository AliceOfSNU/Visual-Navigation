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
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#define NOINCREMENT
#define OUT

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using pii = pair<int, int>;

/** This is the main tracking function, given two images, it detects,
 * describes and matches features.
 * We will be modifying this function incrementally to plot different figures
 * and compute different statistics.
 * useful when a feature is tracked for no more than one frame
 @param[in] img_1, img_2 Images where to track features.
 @param[out] matched_kp_1_kp_2 pair of vectors of keypoints with the same size
 so that matched_kp_1_kp_2.first[i] matches with matched_kp_1_kp_2.second[i].
*/
void FeatureTracker::trackFeatures(const cv::Mat &img_1,
                                   const cv::Mat &img_2,
                                   std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> *matched_kp_1_kp_2,
                                   cv::Mat& match_img) {

  vector<KeyPoint> keypoints_1, keypoints_2; 
  detectKeypoints(img_1, &keypoints_1); detectKeypoints(img_2, &keypoints_2);

  cv::Mat desc_1, desc_2; //size [kpt_count * 128]
  describeKeypoints(img_1, &keypoints_1, &desc_1); describeKeypoints(img_2, &keypoints_2, &desc_2);

  std::vector<std::vector<DMatch>> matches;
  std::vector<DMatch> good_matches;

  matchDescriptors(desc_1, desc_2, &matches, &good_matches);
  vector<KeyPoint> mkpts_1(good_matches.size()), mkpts_2(good_matches.size());
  for(int i = 0; i < good_matches.size(); ++i){
    mkpts_1[i] = keypoints_1[good_matches[i].queryIdx];
    mkpts_2[i] = keypoints_2[good_matches[i].trainIdx];
  }

  //run RANSAC
  vector<uchar> mask;
  size_t num_inliers = filterInliers(mkpts_1, mkpts_2, OUT &mask);
  
  assert(matched_kp_1_kp_2 != nullptr);
  matched_kp_1_kp_2->first = mkpts_1, matched_kp_1_kp_2->second = mkpts_2;  
  
  
#if 1 //draw
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
#endif  
  update_stats(keypoints_1.size(), keypoints_2.size(), good_matches.size(), num_inliers);
  
  //add mask to image

}

/** This is the main tracking function, given two images, it detects,
 * describes and matches features.
 * We will be modifying this function incrementally to plot different figures
 * and compute different statistics.
 * useful when a feature is tracked for no more than one frame
 @param[in] img_1, img_2 Images where to track features.
 @param[out] matched_kp_1_kp_2 pair of vectors of keypoints with the same size
 so that matched_kp_1_kp_2.first[i] matches with matched_kp_1_kp_2.second[i].
*/
void FeatureTracker::trackFeatures(const cv::Mat &img_1,
                                   const cv::Mat &img_2,
                                   std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> *matched_kp_1_kp_2,
                                   cv::Mat& match_img,
                                   const cv::Rect &roi_1,
                                   const cv::Rect &roi_2
                                   ) {
  vector<KeyPoint> keypoints_1, keypoints_2; 
  cv::Mat mask1 = cv::Mat::zeros(img_1.size(), CV_8UC1), mask2 = cv::Mat::zeros(img_2.size(), CV_8UC1);
  mask1(roi_1) = Scalar(255), mask2(roi_2)=Scalar(255);

  detectKeypoints(img_1, &keypoints_1, mask1); detectKeypoints(img_2, &keypoints_2, mask2);
  cout << "detected " << keypoints_1.size() << " keypoints." << endl;


  if(keypoints_1.empty() || keypoints_2.empty()){
    //failed to detect any matches inside.. we can still draw..
    cv::Mat masked[2];
    cv::Mat cmask_1=cv::Mat::zeros(img_1.size(), img_1.type()), cmask_2=cv::Mat::zeros(img_2.size(), img_2.type());
    cmask_1(roi_1) = Scalar(255, 255, 0), cmask_2(roi_2)=Scalar(255, 255, 0);
  
    addWeighted( img_1, 0.5, cmask_1, 0.5, 1.0, masked[0]);
    addWeighted( img_2, 0.5, cmask_2, 0.5, 1.0, masked[1]);
    hconcat(masked, 2, match_img);
    cout << "nothing here!omg" << endl;
    assert(matched_kp_1_kp_2 != nullptr);
    matched_kp_1_kp_2->first.clear(), matched_kp_1_kp_2->second.clear();  
    update_stats(keypoints_1.size(), keypoints_2.size(), 0, 0);
    return;
  }

  cv::Mat desc_1, desc_2; //size [kpt_count * 128]
  describeKeypoints(img_1, &keypoints_1, &desc_1); describeKeypoints(img_2, &keypoints_2, &desc_2);

  std::vector<std::vector<DMatch>> matches;
  std::vector<DMatch> good_matches;

  matchDescriptors(desc_1, desc_2, &matches, &good_matches);
  vector<KeyPoint> mkpts_1(good_matches.size()), mkpts_2(good_matches.size());
  for(int i = 0; i < good_matches.size(); ++i){
    mkpts_1[i] = keypoints_1[good_matches[i].queryIdx];
    mkpts_2[i] = keypoints_2[good_matches[i].trainIdx];
  }

  //run RANSAC
  vector<uchar> mask;
  cout << "detected " << mkpts_1.size() << "matches" << endl;

  size_t num_inliers = filterInliers(mkpts_1, mkpts_2, OUT &mask);
  
  assert(matched_kp_1_kp_2 != nullptr);
  matched_kp_1_kp_2->first = mkpts_1, matched_kp_1_kp_2->second = mkpts_2;  
  
  cout << "before draw" << endl;
#if 1 //draw
  vector<char> cmask{mask.begin(), mask.end()}; 
  cv::Mat masked_1, masked_2, cmask_1=cv::Mat::zeros(img_1.size(), img_1.type()), cmask_2=cv::Mat::zeros(img_2.size(), img_2.type());
  cmask_1(roi_1) = Scalar(255, 255, 0), cmask_2(roi_2)=Scalar(255, 255, 0);
  
  addWeighted( img_1, 0.5, cmask_1, 0.5, 1.0, masked_1);
  addWeighted( img_2, 0.5, cmask_2, 0.5, 1.0, masked_2);
  cv::drawMatches(masked_1, keypoints_1, masked_2, keypoints_2, good_matches, match_img, 
    cv::Scalar(0, 255, 0), cv::Scalar::all(-1), cmask, 
    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
  );
  std::transform(cmask.begin(), cmask.end(), cmask.begin(), [](const char& x) {return !x;});
  cv::drawMatches(masked_1, keypoints_1, masked_2, keypoints_2, good_matches, match_img, 
    cv::Scalar(0, 0, 255), cv::Scalar::all(-1), cmask, 
    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG
  );
#endif  
  update_stats(keypoints_1.size(), keypoints_2.size(), good_matches.size(), num_inliers);
  

}

/** useful if features are tracked for longer times.
  * returns how many frames each landmark has been tracked for.
  * specify region of interest
  @param[in] img_1 the prev image
  @param[in] img_2 the current image
  @param[out] landmarks a vector of struct containing unique landmark id, and number of frames it has been tracked for.
  @param[in] roi_1, roi_2 the region of interest to detect keypoints in.
*/
void FeatureTracker::trackFeaturesContinuous(const cv::Mat &img_1,
                                   const cv::Mat &img_2,
                                   std::vector<observation>& landmarks,
                                   cv::Mat& match_img,
                                   const cv::Rect& roi_1,
                                   const cv::Rect& roi_2) {
  //maximum number of tracks is bounded by the number of detections of the ORB detector
  //which is something like 200-500, very managable.
  static map<size_t, size_t> lookups[2];
  //keeps track of how for how many frames a track was matched.
  static map<size_t, size_t> obsdata;
  static int lookupidx = 0;
  static size_t next_id = 1;

  assert(!img_1.empty());
  assert(!img_2.empty());

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
  //reverse map
  vector<KeyPoint> mkpts_1(good_matches.size()), mkpts_2(good_matches.size());
  for(int i = 0; i < good_matches.size(); ++i){
    mkpts_1[i] = keypoints_1[good_matches[i].queryIdx];
    mkpts_2[i] = keypoints_2[good_matches[i].trainIdx];

  }

  //run RANSAC
  vector<uchar> mask;
  size_t num_inliers = filterInliers(mkpts_1, mkpts_2, OUT &mask);

  //draw
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

  //find all keypoints that are tracked in this frame.
  //and update their pixel and tracking information
  map<size_t, size_t> &lookup = lookups[lookupidx];
  map<size_t, size_t> &lookup_new = lookups[lookupidx ^ 1];
  cout << "lookup" << lookupidx << " size:" << lookup.size() << endl;

  vector<size_t> active_landmarks{};
  landmarks.clear(); //to return

  size_t upd_cnt = 0;
  for(int i = 0; i < mkpts_1.size(); ++i){
    size_t key = point_hash(mkpts_1[i].pt);
    auto lookup_it = lookup.find(key);
    if(lookup_it == lookup.end()){
      //newly initiated track
      lookup_new[point_hash(mkpts_2[i].pt)] = next_id; //insertion
      landmarks.push_back({next_id, mkpts_2[i], 1}); //return
      obsdata[next_id] = 1; //save track info
      active_landmarks.push_back(next_id);
      next_id++;
    }else{
      size_t tid = lookup_it->second;
      lookup_new[point_hash(mkpts_2[i].pt)] = tid; //update pixel-trackid association
      landmarks.push_back({tid, mkpts_2[i], ++obsdata[tid]}); //return
      active_landmarks.push_back(tid);
      upd_cnt++;
    }
  }
  std::cout << upd_cnt << "are existing tracks out of " << mkpts_1.size() << endl;


  //purge all terminated landmark data.
  std::sort(active_landmarks.begin(), active_landmarks.end());
  std::cout << "total " << active_landmarks.size() << " active out of " << obsdata.size() << endl;
  size_t j = 0, delete_cnt = 0;
  for(auto it = obsdata.cbegin(); it != obsdata.cend(); NOINCREMENT){
    if(it->first != active_landmarks[j]){
      //should erase
      delete_cnt++;
      it = obsdata.erase(it);
    }else{
      ++it; ++j;
    }
  }
  cout << "obs data size: " << obsdata.size() <<"," << delete_cnt <<"deleted" << endl;

  //swap
  lookup.clear(); //clear old pixel-track associations
  lookupidx ^= 1;

}

/** compute inliers and erase all outliers from matched keypoints vector
 * This actually modifies the original vector and removes all outliers. 
  @param[in] mkpts_1 List of matched keypoints
  @param[in] mkpts_2 List of matched keypoints
  @param[out] mask 1 if inliers, 0 is outlier.
  @returns[size_t] how many are inliers!
*/
size_t FeatureTracker::filterInliers(vector<KeyPoint> &mkpts_1, vector<KeyPoint> &mkpts_2, vector<uchar> *mask){
  inlierMaskComputation(mkpts_1, mkpts_2, mask);
  size_t cnt = 0;
  for(int i = 0; i < mkpts_1.size(); ++i){
    if((*mask)[i]) {
      continue;
    }
    mkpts_1[cnt] = mkpts_1[i];
    mkpts_2[cnt] = mkpts_2[i];
    ++cnt;
  }
  mkpts_1.resize(cnt); mkpts_2.resize(cnt);
  return cnt;
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

void FeatureTracker::update_stats(size_t num_kp1, size_t num_kp2, size_t num_matches, size_t num_inliers){
  float const new_num_samples = static_cast<float>(num_samples_) + 1.0f;
  float const old_num_samples = static_cast<float>(num_samples_);
  avg_num_keypoints_img1_ = (avg_num_keypoints_img1_ * old_num_samples + static_cast<float>(num_kp1)) / new_num_samples;
  avg_num_keypoints_img2_ = (avg_num_keypoints_img2_ * old_num_samples + static_cast<float>(num_kp2)) / new_num_samples;
  avg_num_matches_ = (avg_num_matches_ * old_num_samples + static_cast<float>(num_matches)) / new_num_samples;
  avg_num_good_matches_ = (avg_num_good_matches_ * old_num_samples + static_cast<float>(num_matches)) / new_num_samples;
  avg_num_inliers_ = (avg_num_inliers_ * old_num_samples + static_cast<float>(num_inliers)) / new_num_samples;
  avg_inlier_ratio_ =
      (avg_inlier_ratio_ * old_num_samples + (static_cast<float>(num_inliers) / static_cast<float>(num_matches))) / new_num_samples;
  ++num_samples_;
}

/**
 * a drawing function that puts lines between matches
*/
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
