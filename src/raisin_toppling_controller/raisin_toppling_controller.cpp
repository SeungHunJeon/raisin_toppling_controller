//
// Created by donghoon on 8/23/22.
// 
 
#include <filesystem>
#include "ament_index_cpp/get_package_prefix.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "raisin_toppling_controller/raisin_toppling_controller.hpp"

namespace raisin {
namespace controller {

using std::placeholders::_1;
using std::placeholders::_2;

raiboLearningController::raiboLearningController(
    raisim::World & world, raisim::RaisimServer & server,
    raisim::World & worldSim, raisim::RaisimServer & serverSim, GlobalResource & globalResource)
: Controller("raisin_toppling_controller", world, server, worldSim, serverSim, globalResource),
      encoder_(ENCNUMLAYER),
      actor_({256, 128}),
      param_(parameter::ParameterContainer::getRoot()["raiboLearningController"])
      {
  RSINFO("Initializer")
  param_.loadFromPackageParameterFile("raisin_toppling_controller");

  rclcpp::QoS qos(rclcpp::KeepLast(1));
}

bool raiboLearningController::create() {
  RSINFO("Controller Create!!")
  
  control_dt_ = 0.01;
  communication_dt_ = 0.00025;
  raiboController_.create(robotHub_);
  

  /// load object geometry
  Eigen::VectorXd obj_geom(3);
  obj_geom = param_("box_geometry");

  obj_geometry_ = obj_geom;

  /// load policy network parameters
  std::string model_itertaion = std::string(param_("model_number"));
  std::string encoder_file_name = std::string("GRU_") + model_itertaion + std::string(".txt");
  std::string actor_file_name = std::string("MLP_") + model_itertaion + std::string(".txt");
  std::string obs_mean_file_name = std::string("mean") + model_itertaion + std::string(".csv");
  std::string obs_var_file_name = std::string("var") + model_itertaion + std::string(".csv");

  std::string network_path = std::string(param_("network_path"));
  std::filesystem::path pack_path(ament_index_cpp::get_package_prefix("raisin_toppling_controller"));
  std::filesystem::path encoder_path = pack_path / network_path / encoder_file_name;
  std::filesystem::path actor_path = pack_path / network_path / actor_file_name;
  std::filesystem::path obs_mean_path = pack_path / network_path / obs_mean_file_name;
  std::filesystem::path obs_var_path = pack_path / network_path / obs_var_file_name;

  encoder_.readParamFromTxt(encoder_path.string());
  actor_.readParamFromTxt(actor_path.string());

  RSINFO(encoder_path.string())
  RSINFO(obs_mean_path.string())
  std::string in_line;
  std::ifstream obsMean_file(obs_mean_path.string());
  std::ifstream obsVariance_file(obs_var_path.string());
  obs_.setZero(raiboController_.getObDim());
  obsMean_.setZero(raiboController_.getObDim());
  obsVariance_.setZero(raiboController_.getObDim());

  /// load observation mean and variance
  if (obsMean_file.is_open()) {
    for (int i = 0; i < obsMean_.size(); ++i) {
      std::getline(obsMean_file, in_line, '\n');
      obsMean_(i) = std::stof(in_line);
    }
  }

  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obsVariance_.size(); ++i) {
      std::getline(obsVariance_file, in_line, '\n');
      obsVariance_(i) = std::stof(in_line);
    }
  }

  obsMean_file.close();
  obsVariance_file.close();

  joySubscribeTime_ = 0.;

  logIdx_ = dataLogger_.initializeAnotherDataGroup(
      "toppling",
      "observation", raiboController_.getObservation(),
      "targetPosition", raiboController_.getJointPTarget()
  );

  RSINFO("Set Command !")
  auto obj = reinterpret_cast<raisim::SingleBodyObject*>(worldSim_.getObject("Object"));
  raiboController_.setCommand(obj, obj_geometry_, 0);

  RSINFO("Create Done")

  return true;
}

bool raiboLearningController::init() {
  return true;
}

bool raiboLearningController::advance() {
  controlEnd_ = std::chrono::high_resolution_clock::now();
  elapsedTime_ = std::chrono::duration_cast<std::chrono::microseconds>(controlEnd_ - controlBegin_).count() / 1.e6;

  if (fabs(fmod(elapsedTime_, 0.0025)) < 1e-6) {
    // RSINFO("clip torque")
    raiboController_.clipTorque();
  }

  if (elapsedTime_ < control_dt_) {
    return true;
  }

  else {
  /// 100Hz controller
  
  controlBegin_ = std::chrono::high_resolution_clock::now();
  // robotHub_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  robotHub_->setPdGains(raiboController_.getJointPGain(), raiboController_.getJointDGain());
  raiboController_.updateObservation();
  raiboController_.advance(obsScalingAndGetAction().head(12));
  dataLogger_.append(logIdx_,
      raiboController_.getObservation(), raiboController_.getJointPTarget());
  }
  return true;
}


Eigen::VectorXf raiboLearningController::obsScalingAndGetAction() {
  /// normalize the obs
  obs_ = raiboController_.getObservation().cast<float>();

  for (int i = 0; i < obs_.size(); ++i) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
  }
  /// forward the obs to the encoder
  Eigen::Matrix<float, OBSDIM, 1> encoder_input = obs_.head(OBSDIM);
  latent_ = encoder_.forward(encoder_input);

  /// concat obs and e_out and forward to the actor
  Eigen::Matrix<float, ENCOUTDIM + OBSDIM, 1> actor_input;
  actor_input << latent_, obs_;

  Eigen::VectorXf action = actor_.forward(actor_input);
  return action;
}

bool raiboLearningController::reset() {
  RSINFO("reset start")
  raiboController_.reset();
  encoder_.initHidden();
  controlEnd_ = std::chrono::high_resolution_clock::now();
  joySubscribeEnd_ = std::chrono::high_resolution_clock::now();
  RSINFO("reset done")
  return true;
}

bool raiboLearningController::terminate() { return true; }

bool raiboLearningController::stop() { return true; }

extern "C" Controller * create(
    raisim::World & world, raisim::RaisimServer & server,
    raisim::World & worldSim, raisim::RaisimServer & serverSim, GlobalResource & globalResource)
{
  return new raiboLearningController(world, server, worldSim, serverSim, globalResource);
}

extern "C" void destroy(Controller *p) {
  delete p;
}

void raiboLearningController::setCommand(raisim::SingleBodyObject* object, Eigen::VectorXd& object_geometry, const int toppling_type) {
  raiboController_.setCommand(object, object_geometry, toppling_type);
}

void raiboLearningController::joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
try {
  joySubscribeBegin_ = std::chrono::high_resolution_clock::now();
  joySubscribeTime_ = std::chrono::duration_cast<std::chrono::microseconds>(
      joySubscribeBegin_ - joySubscribeEnd_).count();

  command_ << msg->axes[0], msg->axes[1], msg->axes[2];
  // raiboController_.setCommand(command_);

  joySubscribeEnd_ = std::chrono::high_resolution_clock::now();
} catch (const std::exception &e) {
  std::cout << e.what();
}

}
}


