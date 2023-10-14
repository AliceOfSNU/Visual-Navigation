
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
    void add_pose_and_projection_factor(geometry_msgs::PoseConstPtr& cam_pose, cv::Point2f& obj_centroid);
    void set_K(const Matrix44& K);
    void print_graph(){
        graph.print();
    }
    int optimize_graph();
    int get_poses_optimal(OUT vector<geometry_msgs::Pose>& poses);
    int get_centroid_optimal(OUT geometry_msgs::Point centroid);


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
    size_t num_cam_poses, num_obj_pos;

    /**
     * converts a geometry_msg pose to gtsam::Pose3
    */
    Pose3 msgToPose3(geometry_msgs::Pose::ConstPtr& msg){
        geometry_msgs::Quaternion q = msg->orientation;
        geometry_msgs::Point p = msg->position;

        //copy ellision or move.
        return Pose3(Rot3(Quaternion(q.w, q.x, q.y, q.z)), Point3(p.x, p.y, p.z));
    }

    /**
     * converts a geometry_msg point to gtsam::Point3
    */
   Point3 msgToPoint3(geometry_msgs::Point::ConstPtr& msg){
        return Point3(msg->x, msg->y, msg->z);
   }

   /**
    * converts a cv2 Point2f to gtsam::Point2
   */
  Point2 cvToPoint2(cv::Point2f const& cvpt2){
    return Point2(cvpt2.x, cvpt2.y);
  }

};
