#pragma once 

#define OUT
#include <vector>
#include <iostream>
// GTSAM
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point2.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/Marginals.h>

//data types
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <opencv2/core.hpp>
#include "utils.hpp"

using namespace std;
using namespace gtsam;
using ProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;

class PoseGraphOptimizer{
public:
    PoseGraphOptimizer();
    PoseGraphOptimizer(double isotropic_pos_sigma, double isotropic_rot_sigma, double isotropic_pix_sigma);
    ~PoseGraphOptimizer() = default;

    //all data interfaces are via geometry_msgs. gtsam dependencies do not leak  out.
    void initialize_priors(geometry_msgs::PoseConstPtr& cam_init_pose, geometry_msgs::PointConstPtr& obj_init_pos);
    //void add_pose_and_projection_factor(geometry_msgs::PoseConstPtr& cam_pose, cv::Point2f& obj_centroid);
    void add_pose_prior(size_t cam_id,const Pose3& pose);
    void add_projection_factor(size_t cam_id, size_t landmark_id, const Point2& landmark_pixel);
    void set_K(double fx, double fy, double s, double u0, double v0);
    Cal3_S2 get_K(){return *K;};
    void add_point_prior(size_t landmark_id, const Point3& point);
    void print_graph(){
        graph.print();
    }
    bool has_key(Symbol sym){
        return graph.keys().exists(sym.key());
    }
    int optimize_graph();
    Values optimize();
    double getError(Values const& val){
      return graph.error(val);
    }

private:
    Cal3_S2::shared_ptr K;

    // Define the camera observation noise model
    double isotropic_pos_sigma = 0.3;
    double isotropic_rot_sigma = 0.1;
    double isotropic_pix_sigma = 10.0;
    noiseModel::Isotropic::shared_ptr pixel_noise =
        noiseModel::Isotropic::Sigma(2, 10.0);  // one pixel in u and v

    noiseModel::Isotropic::shared_ptr position_noise =
        noiseModel::Isotropic::Sigma(3, 0.3);

      const gtsam::noiseModel::Diagonal::shared_ptr pose_noise = 
      gtsam::noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3)).finished());

    // Create an empty factor graph
    NonlinearFactorGraph graph;

    size_t num_factors = 0;
    Values initialEstimate;

};
