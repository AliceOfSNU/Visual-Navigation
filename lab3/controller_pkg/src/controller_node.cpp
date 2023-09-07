#include <ros/ros.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <mav_msgs/Actuators.h>
#include <nav_msgs/Odometry.h>
#include <trajectory_msgs/MultiDOFJointTrajectoryPoint.h>
#include <cmath>

#define PI M_PI
#define EPS 10e-7


#define IN
#define OUT
#define INOUT
#define FALLTRHOUGH

inline bool ALMOST_EQUAL_D(double X, double Y) const{
  return abs(X-Y) < EPS;
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  PART 0 |  16.485 - Fall 2019  - Lab 3 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement a geometric controller for a
//  simulated UAV, following the publication:
//
//  [1] Lee, Taeyoung, Melvin Leoky, N. Harris McClamroch. "Geometric tracking
//      control of a quadrotor UAV on SE (3)." Decision and Control (CDC),
//      49th IEEE Conference on. IEEE, 2010
//
//  We use variable names as close as possible to the conventions found in the
//  paper, however, we have slightly different conventions for the aerodynamic
//  coefficients of the propellers (refer to the lecture notes for these).
//  Additionally, watch out for the different conventions on reference frames


#include <eigen3/Eigen/Dense>
using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using Cross = Eigen::Cross;
using MatrixXd = Eigen::MatrixXd;
using Matrix4d = Eigen::Matrix4d;
using Matrix3d = Eigen::Matrix3d;
using Matrix2d = Eigen::Matrix2d;

#include <tf2_eigen/tf2_eigen.h>

class controllerNode{
  ros::NodeHandle nh;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  PART 1 |  Declare ROS callback handlers
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //
  // In this section, you need to declare:
  //   1. two subscribers (for the desired and current UAVStates)
  //   2. one publisher (for the propeller speeds)
  //   3. a timer for your main control loop
  //
  // ~~~~ begin solution
  //
  //     **** FILL IN HERE ***
  //
  // ~~~~ end solution
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //                                 end part 1
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Controller parameters
  double kx, kv, kr, komega; // controller gains - [1] eq (15), (16)

  // Physical constants (we will set them below)
  double m;              // mass of the UAV
  double g;              // gravity acceleration
  double d;              // distance from the center of propellers to the c.o.m.
  double cf,             // Propeller lift coefficient
         cd;             // Propeller drag coefficient
  Eigen::Matrix3d J;     // Inertia Matrix
  Eigen::Vector3d e3;    // [0,0,1]
  Eigen::MatrixXd F2W;   // Wrench-rotor speeds map
  Eigen::MatrixXd F2W_INV;   // Wrench-rotor speeds map inverse
  
  // Controller internals (you will have to set them below)
  // Current state
  Eigen::Vector3d x;     // current position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d v;     // current velocity of the UAV's c.o.m. in the world frame
  Eigen::Matrix3d R;     // current orientation of the UAV
  Eigen::Vector3d omega; // current angular velocity of the UAV's c.o.m. in the *body* frame

  // Desired state
  Eigen::Vector3d xd;    // desired position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d vd;    // desired velocity of the UAV's c.o.m. in the world frame
  Eigen::Vector3d ad;    // desired acceleration of the UAV's c.o.m. in the world frame
  double yawd;           // desired yaw angle

  double hz;             // frequency of the main control loop


  static Eigen::Vector3d Vee(const Eigen::Matrix3d& in){
    Eigen::Vector3d out;
    out << in(2,1), in(0,2), in(1,0);
    return out;
  }

  static double signed_sqrt(double val){
    return val>0?sqrt(val):-sqrt(-val);
  }

public:
  controllerNode():e3(0,0,1),F2W(4,4),F2W_INV(4, 4),hz(1000.0){

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  PART 2 |  Initialize ROS callback handlers
      // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
      //
      // In this section, you need to initialize your handlers from part 1.
      // Specifically:
      //  - bind controllerNode::onDesiredState() to the topic "desired_state"
      //  - bind controllerNode::onCurrentState() to the topic "current_state"
      //  - bind controllerNode::controlLoop() to the created timer, at frequency
      //    given by the "hz" variable
      //
      // Hints: 
      //  - use the nh variable already available as a class member
      //  - read the lab 3 handout to fnd the message type
      //
      // ~~~~ begin solution
      //

      // fill in F2W
      F2W.row(0) << cf, cf, cf, cf;
      double dd = cf*d/sqrt(2.0);
      F2W.col(0).tail(3) << dd, dd, cd;
      F2W.col(1).tail(3) << -dd, dd, cd;
      F2W.col(2).tail(3) << -dd, -dd, cd;
      F2W.col(3).tail(3) << dd, -dd, cd;
      
      F2W_INV = F2W.inverse();

      // Live the life of a control engineer! Tune these parameters for a fast
      // and accurate controller.
      //
      // Controller gains

      // **** FIDDLE WITH THESE! ***
      // Change them in the provided launch file.
      nh.getParam("kx", kx);
      nh.getParam("kv", kv);
      nh.getParam("kr", kr);
      nh.getParam("komega", komega);
      ROS_INFO("Gain values:\nkx: %f \nkv: %f \nkr: %f \nkomega: %f\n", kx, kv, kr, komega);

      // Initialize constants
      m = 1.0;
      cd = 1e-5;
      cf = 1e-3;
      g = 9.81;
      d = 0.3;
      J << 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0;
  }

  void onDesiredState(const trajectory_msgs::MultiDOFJointTrajectoryPoint& des_state){

      //  xd, vd, ad: You can ignore the angular acceleration.
      xd << msg.transforms[0].translation.x, msg.transforms[0].translation.y, msg.transforms[0].translation.z;
      vd << msg.velocities[0].linear.x, msg.velocities[0].linear.y, msg.velocities[0].linear.z;
      ad << msg.accelerations[0].linear.x, msg.accelerations[0].linear.y, msg.accelerations[0].linear.z;

      //  Hints:
      //    - use the methods tf2::getYaw(...)
      //    - maybe you want to use also tf2::fromMsg(...)

      Eigen::Quaterniond q(msg.transforms[0].rotation.x,
        msg.transforms[0].rotation.y,
        msg.transforms[0].rotation.z,
        msg.transforms[0].rotation.w);
      Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
      yawd = euler.z;
  }

  void onCurrentState(const nav_msgs::Odometry& cur_state){
      // x, v, R and omega
      //  CAVEAT: cur_state.twist.twist.angular is in the world frame, while omega
      //          needs to be in the body frame!

      x << cur_state.pose.pose.position.x,  cur_state.pose.pose.position.y,  cur_state.pose.pose.position.z;
      v << cur_state.twist.twist.linear.x, cur_state.twist.twist.linear.y, cur_state.twist.twist.linear.z;
      Eigen::Quaternionf q; q << cur_state.pose.pose.orientation.w << cur_state.pose.pose.orientation.x << cur_state.pose.pose.orientation.y << cur_state.pose.pose.orientation.z;
      R = q.toRotationMatrix(); //base frame wrt world frame
      omega << cur_state.twist.twist.angular.x << cur_state.twist.twist.angular.y << cur_state.twist.twist.angular.z;
      omega = R.inverse()*omega; //world to base.
     
  }

  void controlLoop(const ros::TimerEvent& t){
    Eigen::Vector3d ex, ev, er, eomega;

    // velocity and position tracking errors
    ex = x - xd, ev = v - vd;

    // Rd matrix.
    Vector3d b1d, b2d, b3d, b1d_tilde, f;
    f = -kx*ex - kv*ev + m*g*e3 + m*ad;
    b3d = f.normalized();
    b1d_tilde << cos(yawd), sin(yawd), 0;
    b2d = b3d.cross(b1d_tilde).normalized();
    b1d = b2d.cross(b3d).normalized();
    Matrix3d Rd << b1d, b2d, b3d;
    assert(ALMOST_EQUAL_D(Rd.determinant(), 1.0))

    // orientation error (er) and the rotation-rate error (eomega)
    er = 0.5*Vee(Rd.transpose()*R-R.transpose()*Rd);
    eomega = omega; //omega_d term is ignored.

    // desired wrench (force + torques) to control the UAV.
    double fz_u = f.dot(R*e3); //1d
    Vector3d tau_u = -kr*er - komega*eomega + omega.cross(J*omega); //3d
    //    - feel free to ignore all the terms involving \Omega_d and its time
    //      derivative as they are of the second order and have negligible
    //      effects on the closed-loop dynamics.

    Vector4d sgn_sq_w; sgn_sq_w << fz_u, tau_u;
    sgn_sq_w = F2W_INV * sgn_sq_w;

    // control message
    double w[4]{0.0};
    for(int prop = 0; prop < 4; ++prop){
      w[prop] = signed_sqrt(sgn_sq_w[prop]);
    }
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "controller_node");
  controllerNode n;
  ros::spin();
}
