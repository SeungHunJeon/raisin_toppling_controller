//
// Created by donghoon on 8/23/22.
//
#pragma once

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "raisim/World.hpp"
#include "helper/BasicEigenTypes.hpp"
#include "raisin_toppling_controller/raiboController.hpp"
#include "raisin_parameter/parameter_container.hpp"
#include "raisin_controller/controller.hpp"
#include "raisin_interfaces/srv/vector3.hpp"
#include "raisin_data_logger/raisin_data_logger.hpp"
#include "helper/neuralNet.hpp"

/// TODO
#define OBSDIM 87
#define ACTDIM 12
#define ENCOUTDIM 128
#define ENCNUMLAYER 1

namespace raisin{
namespace controller {

class raiboLearningController : public Controller {

 public:
  raiboLearningController(
    raisim::World & world, raisim::RaisimServer & server,
    raisim::World & worldSim, raisim::RaisimServer & serverSim, GlobalResource & globalResource);
  bool create() final;
  bool init() final;
  Eigen::VectorXf obsScalingAndGetAction();
  bool advance() final;
  bool reset() final;
  bool terminate() final;
  bool stop() final;
  void setCommand(raisim::SingleBodyObject* object, Eigen::VectorXd& object_geometry, const int toppling_type);

  

 private:
  void commandCallback(const raisin_interfaces::msg::Command::SharedPtr msg);

  Eigen::VectorXd obj_geometry_;

  raisim::RaiboController raiboController_;
  raisim::ArticulatedSystem* robot_;
  Eigen::VectorXf obs_;
  Eigen::Matrix<float, ENCOUTDIM, 1> latent_;
  int clk_ = 0;
  Eigen::Vector3f command_;

  parameter::ParameterContainer & param_;

  std::chrono::time_point<std::chrono::high_resolution_clock> controlBegin_;
  std::chrono::time_point<std::chrono::high_resolution_clock> controlEnd_;
  std::chrono::time_point<std::chrono::high_resolution_clock> joySubscribeBegin_;
  std::chrono::time_point<std::chrono::high_resolution_clock> joySubscribeEnd_;

  double elapsedTime_ = 0.;
  double joySubscribeTime_;

  Eigen::VectorXf obsMean_;
  Eigen::VectorXf obsVariance_;
  raisim::nn::GRU<float, OBSDIM, ENCOUTDIM> encoder_;
  raisim::nn::Linear<float, ENCOUTDIM + OBSDIM, ACTDIM, raisim::nn::ActivationType::leaky_relu> actor_;

  double control_dt_;
  double communication_dt_;
};

}
}



