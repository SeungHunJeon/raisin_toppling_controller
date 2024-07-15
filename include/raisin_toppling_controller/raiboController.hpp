//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "raisin_toppling_controller/helper/BasicEigenTypes.hpp"
#include <raisim/RaisimServer.hpp>
//
// Created by donghoon on 8/11/22.
//

namespace raisim {

class RaiboController {
 public:
  inline bool create(raisim::ArticulatedSystem * robot) {
    RSINFO("Raibo Controller Create!!")
    raibo_ = robot;
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    gc_init_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_init_.resize(raibo_->getDOF());

//    raibo_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    /// Observation
    nominalJointConfig_.setZero(nJoints_);
    nominalJointConfig_ << 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
    jointTarget_.setZero(nJoints_);
    gc_init_ << 0, 0, 0.5225, 1, 0, 0, 0, nominalJointConfig_;
    gv_init_.setZero();

    /// clip torque
    clippedGenForce_.setZero(gvDim_);
    raibo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);

    actionMean_ << nominalJointConfig_; /// joint target
    actionStd_ << Eigen::VectorXd::Constant(nJoints_, 0.1); /// joint target

    obDouble_.setZero(obDim_);

    /// pd controller
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(50.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(0.5);
    raibo_->setPdGains(jointPgain_, jointDgain_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
    return true;
  };

  void reset() {
    RSINFO("raibo Controller Reset start")
    clippedGenForce_.tail(nJoints_).setZero();
    raibo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
    raibo_->getState(gc_, gv_);
    jointTarget_ = gc_.tail(nJoints_);
    previousAction_ << gc_.tail(nJoints_);
    RSINFO("raibo Controller Reset Done")
  }

  void updateStateVariables() {
    raibo_->getState(gc_, gv_);

    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

    /// Object
    objectRot_.e() = object_->getOrientation().e() * yaw_shift.e();
    objectPos_.e() = object_->getPosition();

    raisim::Vec<3> offset{0, 0, object_geometry_(2) / 2};
    desired_FOOT_Pos.e() = objectPos_.e() + objectRot_.e() * offset.e();
    raibo_->getFramePosition(raibo_->getFrameIdxByLinkName("LF_FOOT"), LF_FOOT_Pos);
    raibo_->getFramePosition(raibo_->getFrameIdxByLinkName("RF_FOOT"), RF_FOOT_Pos);
  }

  bool advance(const Eigen::Ref<EigenVec> &action) {
    /// action scaling

    jointTarget_ = action.cast<double>();
    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;

    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    previousAction_ = jointTarget_;

    return true;
  }

  void clipTorque() {
    jointPos_ = raibo_->getGeneralizedCoordinate().e().tail(nJoints_);
    jointVel_ = raibo_->getGeneralizedVelocity().e().tail(nJoints_);
    clippedGenForce_.tail(nJoints_) = jointPgain_.tail(1) * (jointTarget_ - jointPos_) - jointDgain_.tail(1) * jointVel_;

    for(int i = 0; i < nJoints_; i++) {
      /// torque - w clip
      if (std::abs(jointVel_(i)) > clipStart_) {
        clippedTorque_ = - torqueLimit_ / (jointVelLimit_ - clipStart_) * (std::abs(jointVel_(i)) - jointVelLimit_);
        if (std::abs(clippedGenForce_.tail(nJoints_)(i)) > clippedTorque_) {
          clippedGenForce_.tail(nJoints_)(i) = std::copysign(clippedTorque_, clippedGenForce_.tail(nJoints_)(i));
        }
      }
        /// torque limit clip
      else {
        if (std::abs(clippedGenForce_.tail(nJoints_)(i)) > torqueLimit_) {
          clippedTorque_ = torqueLimit_;
          clippedGenForce_.tail(nJoints_)(i) = std::copysign(clippedTorque_, clippedGenForce_.tail(nJoints_)(i));
        }
      }
    }

    raibo_->setGeneralizedForce(clippedGenForce_);
  }

  void updateObservation() {
    updateStateVariables();
    /// body orientation
    obDouble_.head(3) = baseRot_.e().row(2).transpose();
    /// body ang vel
    obDouble_.segment(3, 3) = bodyAngVel_;
    /// joint pos
    obDouble_.segment(6, nJoints_) = gc_.tail(nJoints_);
    /// joint vel
    obDouble_.segment(18, nJoints_) = gv_.tail(nJoints_);
    /// previous action
    obDouble_.segment(30, nJoints_) = previousAction_;
    /// object position
    obDouble_.segment(42, 3) = baseRot_.e().transpose() * (objectPos_.e() - raibo_->getBasePosition().e());
    obDouble_.segment(45, 3) = baseRot_.e().transpose() * (target_objectPos_.e() - raibo_->getBasePosition().e());
    obDouble_.segment(48, 3) = baseRot_.e().transpose() * (target_objectPos_.e() - objectPos_.e());
    /// object orientation
    obDouble_.segment(51, 3) = (baseRot_.e().transpose() * objectRot_.e()).row(0).transpose();
    obDouble_.segment(54, 3) = (baseRot_.e().transpose() * objectRot_.e()).row(1).transpose();
    obDouble_.segment(57, 3) = (baseRot_.e().transpose() * objectRot_.e()).row(2).transpose();
    obDouble_.segment(60, 3) = (baseRot_.e().transpose() * target_objectRot_.e()).row(0).transpose();
    obDouble_.segment(63, 3) = (baseRot_.e().transpose() * target_objectRot_.e()).row(1).transpose();
    obDouble_.segment(66, 3) = (baseRot_.e().transpose() * target_objectRot_.e()).row(2).transpose();
    obDouble_.segment(69, 3) = (baseRot_.e().transpose() * objectRot_.e().transpose() * target_objectRot_.e()).row(0).transpose();
    obDouble_.segment(72, 3) = (baseRot_.e().transpose() * objectRot_.e().transpose() * target_objectRot_.e()).row(1).transpose();
    obDouble_.segment(75, 3) = (baseRot_.e().transpose() * objectRot_.e().transpose() * target_objectRot_.e()).row(2).transpose();

    /// Foot guiding
    obDouble_.segment(78, 3) = baseRot_.e().transpose() * (desired_FOOT_Pos.e() - LF_FOOT_Pos.e());
    obDouble_.segment(81, 3) = baseRot_.e().transpose() * (desired_FOOT_Pos.e() - RF_FOOT_Pos.e());

    /// object geometry
    obDouble_.segment(84, 3) = object_geometry_;
  }

  Eigen::VectorXd getObservation() {
    return obDouble_;
  }

  void setCommand(raisim::SingleBodyObject* object, Eigen::VectorXd& object_geometry, const int toppling_type) {
    object_ = object;
    object_geometry_ = object_geometry;
    initial_objectRot_ = object->getOrientation();
    toppling_type_ = toppling_type;
    update_target_info();
  }

  void update_target_info() {
    Eigen::Vector3d pos_shift;
    raisim::Mat<3,3> rot_shift;
    raisim::angleAxisToRotMat({0, 1, 0}, -M_PI_2, rot_shift);

    if(initial_objectRot_(2, 2) > 0.9) {
      obj_x = object_geometry_(0);
      obj_y = object_geometry_(1);
      obj_z = object_geometry_(2);
    }
    else if (initial_objectRot_(1, 1) > 0.9) {
      obj_x = object_geometry_(2);
      obj_y = object_geometry_(1);
      obj_z = object_geometry_(0);
    }

    switch (toppling_type_)
    {
      case 0: /// y - axis -M_PI / 2 topple
        pos_shift << -(object_geometry_(2) + object_geometry_(0)) / 2, 0, -(object_geometry_(2) - object_geometry_(0)) / 2;
        raisim::angleAxisToRotMat({0, 0, 1}, 0, yaw_shift);
        break;
      case 1: /// y - axis M_PI / 2 topple
        pos_shift << (object_geometry_(2) + object_geometry_(0)) / 2, 0, -(object_geometry_(2) - object_geometry_(0)) / 2;
        raisim::angleAxisToRotMat({0, 0, 1}, M_PI, yaw_shift);
        break;
      case 2: /// x - axis -M_PI / 2 topple
        pos_shift << 0, (object_geometry_(2) + object_geometry_(1)) / 2, -(object_geometry_(2) - object_geometry_(1)) / 2;
        raisim::angleAxisToRotMat({0, 0, 1}, -M_PI_2, yaw_shift);
        break;
      case 3: /// x - axis M_PI / 2 topple
        pos_shift << 0, -(object_geometry_(2) + object_geometry_(1)) / 2, -(object_geometry_(2) - object_geometry_(1)) / 2;
        raisim::angleAxisToRotMat({0, 0, 1}, M_PI_2, yaw_shift);
        break;
    }

    target_objectRot_.e() = object_->getOrientation().e() * yaw_shift.e() * rot_shift.e();

    RSINFO(target_objectRot_.e())

    ///TODO (only considering yaw orientation)
    target_objectPos_.e() = object_->getPosition() + object_->getOrientation().e() * pos_shift;
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  void getInitState(Eigen::VectorXd &gc, Eigen::VectorXd &gv) {
    gc.resize(gcDim_);
    gv.resize(gvDim_);
    gc << gc_init_;
    gv << gv_init_;
  }

  Eigen::VectorXd getJointPGain() const { return jointPgain_; }
  Eigen::VectorXd getJointDGain() const { return jointDgain_; }
  Eigen::VectorXd getJointPTarget() const { return jointTarget_; }
  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] double getSimDt() { return simDt_; }
  [[nodiscard]] double getConDt() { return conDt_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

  void setSimDt(double dt) { simDt_ = dt; };
  void setConDt(double dt) { conDt_ = dt; };

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  Eigen::VectorXd nominalJointConfig_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t obDim_ = 87;
  double simDt_ = .0025;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;

  // robot state variables
  Eigen::VectorXd gc_, gv_, gc_init_, gv_init_;
  Eigen::Vector3d bodyAngVel_;
  raisim::Mat<3, 3> baseRot_;

  // robot observation variables
  Eigen::VectorXd obDouble_;

  // control variables
  double conDt_ = 0.01;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_;
  Eigen::VectorXd jointPgain_, jointDgain_;
  Eigen::Vector3d command_;

  raisim::SingleBodyObject* object_;
  Eigen::VectorXd target_pos_, target_quat_;
  raisim::Mat<3,3> objectRot_, initial_objectRot_;
  raisim::Mat<3,3> target_objectRot_;
  raisim::Vec<3> LF_FOOT_Pos, RF_FOOT_Pos, desired_FOOT_Pos;
  raisim::Vec<3> objectPos_, target_objectPos_;
  Eigen::VectorXd object_geometry_;
  double obj_x, obj_y, obj_z;
  int toppling_type_ = 0; // 0, 1, 2, 3
  raisim::Mat<3,3> yaw_shift;


  /// joint vel limit: 48V 1560rpm (clip start at 743.5rpm) -> 65V 32.77 rad/s (clip start at 20.1 rad/s)
  Eigen::VectorXd clippedGenForce_, frictionTorque_;
  Eigen::VectorXd jointPos_, jointVel_;
  double clippedTorque_, jointVelLimit_ = 35.0091, clipStart_ = 25.0, torqueLimit_ = 90.8558;
};

}