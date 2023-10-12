/**
 * @file deliverable_1.cpp
 * @brief Hands-on introduction: Robot Motion + Robot Localization, as in
 * https://smartech.gatech.edu/handle/1853/45226
 */

/**
 * A simple 2D pose slam example with "GPS" measurements
 *  - The robot moves forward 2 meter each iteration
 *  - The robot initially faces along the X axis (horizontal, to the right in
 * 2D)
 *  - We have full odometry between pose
 *  - We have "GPS-like" measurements implemented with a custom factor
 */

// We will use Pose2 variables (x, y, theta) to represent the robot positions
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/PriorFactor.h>

using namespace std;
using namespace gtsam;

// Before we begin the example, we must create a custom unary factor to
// implement a "GPS-like" functionality. Because standard GPS measurements
// provide information only on the position, and not on the orientation, we
// cannot use a simple prior to properly model this measurement.
//
// The factor will be a unary factor, affect only a single system variable. It
// will also use a standard Gaussian noise model. Hence, we will derive our new
// factor from the NoiseModelFactor1.
#include <gtsam/nonlinear/NonlinearFactor.h>

class UnaryFactor : public NoiseModelFactor1<Pose2> {

  double mx_, my_;

 public:
  UnaryFactor(Key j, double x, double y, const SharedNoiseModel& model)
      : NoiseModelFactor1<Pose2>(model, j), mx_(x), my_(y) {}

  virtual ~UnaryFactor() {}

  // Using the NoiseModelFactor1 base class there are two functions that must be
  // overridden. The first is the 'evaluateError' function. This function
  // implements the desired measurement function, returning a vector of errors
  // when evaluated at the provided variable value. It must also calculate the
  // Jacobians for this measurement function, if requested.
  Vector evaluateError(const Pose2& q,
                       boost::optional<Matrix&> H = boost::none) const {
    // y = h(X) where h is nonlinear function,
    // need derivative of h = H
    if (H) {
      *H = (Matrix(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished();
    }
    return (Vector(2) << q.x() - mx_, q.y() - my_).finished();
  }

}; 

int main(int argc, char** argv) {
  // Create a factor graph container and add factors to it
  NonlinearFactorGraph graph;
  Pose2 priorMean(0.0, 0.0, 0.0);
  noiseModel::Diagonal::shared_ptr priorNoise =
      noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
  graph.add(PriorFactor<Pose2>(1, priorMean, priorNoise));

  // 1a. Add odometry factors
  // For simplicity, we will use the same noise model for each odometry
  // factor
  Pose2 odometry(2.0, 0.0, 0.0);
  noiseModel::Diagonal::shared_ptr odometryNoise =
      noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));

  // TODO: Add the above odometry measurement and noise as between factors
  // between nodes 1, 2 and nodes 2, 3
  // Create odometry (Between) factors between consecutive poses
  //
  // Insert code below:
  graph.add(BetweenFactor<Pose2>(1, 2, odometry, odometryNoise));
  graph.add(BetweenFactor<Pose2>(2, 3, odometry, odometryNoise));

  // End of 1a.

  // 1b. Add "GPS-like" measurements
  // We will use our custom UnaryFactor (already defined for you) for this.
  noiseModel::Diagonal::shared_ptr unaryNoise =
      noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1));

  // Add "GPS" like measurement to nodes 1, 2, and 3 with UnaryFactor 
  // (0, 0) for node 1, (2, 0) for node 2, (4, 0) for node 3
  //
  // Insert code below:
  graph.add(UnaryFactor(1, 0.0, 0.0, unaryNoise));
  graph.add(UnaryFactor(2, 2.0, 0.0, unaryNoise));
  graph.add(UnaryFactor(3, 4.0, 0.0, unaryNoise));
  // End of 1b.

  // 1c. Create the data structure to hold the initialEstimate estimate to
  // the solution For illustrative purposes, these have been deliberately set to
  // incorrect values

  // Add initial estimates (Pick incorrect values to demonstrate estimate
  // before and after optimization)
  Values initial;
  //
  // Insert code below:
  initial.insert(1, Pose2(0.5, 0.0, 0.2));
  initial.insert(2, Pose2(2.3, 0.1, -0.2));
  initial.insert(3, Pose2(4.1, 0.1, 0.1));
  // End of 1c.

  // print and report the initial results.
  graph.print();
  initial.print("Initial Estimate \n");

  // Optimize using Levenberg-Marquardt optimization. 
  Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();

  // Print and report the final results.
  result.print("Optimized Estimate \n");
  return 0;
}
