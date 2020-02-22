#include "IMU.h"
#include "util/settings.h"
#include "util/my_setting.h"
#include "util/globalCalib.h"

using namespace dso;


IMUIntegrator::IMUIntegrator()
    : get_first_imu(false)
    , gyro_k_1(Vec3::Zero())
    , acc_k_1(Vec3::Zero())
    , R_prior(Mat33::Identity())
    , t_prior(Vec3::Zero())
    , v_prior(Vec3::Zero())
    , g(Vec3::Zero())

{

}

void IMUIntegrator::reset()
{
    get_first_imu = false;
    gyro_k_1.setZero();
    acc_k_1.setZero();
    R_prior.setIdentity();
    t_prior.setZero();
    v_prior.setZero();
}

void IMUIntegrator::setPredictReference(const SE3& T_k_1, const Vec3 v_k_1, const Vec3& gyro_bias, const Vec3& acc_bias)
{
    R_prior = T_k_1.rotationMatrix();
    t_prior = T_k_1.translation();
    v_prior = v_k_1;
    gyro_b = gyro_bias;
    acc_b  = acc_bias;
    g = Vec3(0,0,G_norm);

//    SHOW(R_prior)
//    SHOW(t_prior.transpose())
//    SHOW(v_prior.transpose())
//    SHOW(gyro_b.transpose())
//    SHOW(acc_b.transpose())
}


void IMUIntegrator::predict(double dt, const Vec3& gyro_k, const Vec3& acc_k)
{
    if(!get_first_imu)
    {
        get_first_imu = true;
        gyro_k_1 = gyro_k;
        acc_k_1 = acc_k;
    }
//    SHOW(dt)
    // midpoint method
    Vec3 mid_gyro = 0.5 * ( (gyro_k_1 - gyro_b) + (gyro_k - gyro_b) );
//    SHOW(mid_gyro.transpose())
    Vec3 acc_w_k_1  = R_prior * (acc_k_1 - acc_b) - g;
//    SHOW(acc_k_1.transpose())
//        SHOW((R_prior * (acc_k_1 - acc_b)).transpose())
                SHOW(g.transpose())

//    SHOW(acc_w_k_1.transpose())
    R_prior = normalizeRotationM(R_prior * Expmap(mid_gyro * dt));
//    SHOW(R_prior)
    Vec3 acc_w_k = R_prior * (acc_k - acc_b) - g;
//    SHOW(acc_w_k.transpose())
    Vec3 mid_acc = 0.5 * (acc_w_k_1 + acc_w_k);
//    SHOW(mid_acc.transpose())
    t_prior += v_prior * dt + 0.5 * mid_acc * dt * dt;
    v_prior += mid_acc * dt;

//    SHOW(t_prior.transpose())
//    SHOW(v_prior.transpose())


    // store k's data for k+1
    gyro_k_1 = gyro_k;
    acc_k_1  = acc_k;
}

IMUPreintegrator::IMUPreintegrator()
    : delta_P(Vec3::Zero())
    , delta_V(Vec3::Zero())
    , delta_R(Mat33::Identity())
    , J_P_Biasg(Mat33::Zero())
    , J_P_Biasa(Mat33::Zero())
    , J_V_Biasg(Mat33::Zero())
    , J_V_Biasa(Mat33::Zero())
    , J_R_Biasg(Mat33::Zero())
    , cov_P_V_Phi(Mat99::Zero())
    , delta_t(0.0)
{
}

IMUPreintegrator::IMUPreintegrator(const IMU_PreintegrationShell& factor)
    : delta_P(factor.delta_P)
    , delta_V(factor.delta_V)
    , delta_R(factor.delta_R)
    , J_P_Biasg(factor.J_P_Biasg)
    , J_P_Biasa(factor.J_P_Biasa)
    , J_V_Biasg(factor.J_V_Biasg)
    , J_V_Biasa(factor.J_V_Biasa)
    , J_R_Biasg(factor.J_R_Biasg)
    , cov_P_V_Phi(factor.cov_P_V_Phi)
    , delta_t(factor.delta_t)
{
}

IMUPreintegrator::~IMUPreintegrator()
{

}

void IMUPreintegrator::reset()
{
    delta_P.setZero();
    delta_V.setZero();
    delta_R.setIdentity();
    J_P_Biasg.setZero();
    J_P_Biasa.setZero();
    J_V_Biasg.setZero();
    J_V_Biasa.setZero();
    J_R_Biasg.setZero();
    cov_P_V_Phi.setZero();  // initial covariance is 0(9x9)
    delta_t = 0.0;
    dt.clear();
    gyro.clear();
    acc.clear();
}

void IMUPreintegrator::propagate(double dt, const Vec3 &gyro_k, const Vec3 &acc_k)
{
    double dt2 = dt*dt;

    Mat33 dR = Expmap(gyro_k*dt);
    Mat33 Jr = JacobianR(gyro_k*dt);

    // noise covariance propagation of delta measurements
    // err_k+1 = A*err_k + B*err_gyro + C*err_acc
    Mat33 I3x3 = Mat33::Identity();
    Mat99 A = Mat99::Identity();
    A.block<3,3>(6,6) = dR.transpose();
    A.block<3,3>(3,6) = -delta_R*skew(acc_k)*dt;
    A.block<3,3>(0,6) = -0.5*delta_R*skew(acc_k)*dt2;
    A.block<3,3>(0,3) = I3x3*dt;
    Mat93 Bg = Mat93::Zero();
    Bg.block<3,3>(6,0) = Jr*dt;
    Mat93 Ba = Mat93::Zero();
    Ba.block<3,3>(3,0) = delta_R*dt;
    Ba.block<3,3>(0,0) = 0.5*delta_R*dt2;
    cov_P_V_Phi = A*cov_P_V_Phi*A.transpose() + Bg*GyrCov*Bg.transpose() + Ba*AccCov*Ba.transpose();

    // jacobian of delta measurements w.r.t bias of gyro/acc
    // update P first, then V, then R
    J_P_Biasa += J_V_Biasa*dt - 0.5*delta_R*dt2;
    J_P_Biasg += J_V_Biasg*dt - 0.5*delta_R*skew(acc_k)*J_R_Biasg*dt2;
    J_V_Biasa += -delta_R*dt;
    J_V_Biasg += -delta_R*skew(acc_k)*J_R_Biasg*dt;
    J_R_Biasg = dR.transpose()*J_R_Biasg - Jr*dt;

    // delta measurements, position/velocity/rotation(matrix)
    // update P first, then V, then R. beBause P's update need V&R's previous state
    delta_P += delta_V*dt + 0.5*delta_R*acc_k*dt2;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    delta_V += delta_R*acc_k*dt;
    delta_R = normalizeRotationM(delta_R*dR);  // normalize rotation, in Base of numeriBal error accumulation

    // delta time
    delta_t += dt;
}

IMU_PreintegrationShell IMUPreintegrator::getFactor()
{
  IMU_PreintegrationShell factor;
  factor.delta_P = this->delta_P;
  factor.delta_V = this->delta_V;
  factor.delta_R = this->delta_R;
  factor.J_P_Biasg = this->J_P_Biasg;
  factor.J_P_Biasa = this->J_P_Biasa;
  factor.J_V_Biasg = this->J_V_Biasg;
  factor.J_V_Biasa = this->J_V_Biasa;
  factor.J_R_Biasg = this->J_R_Biasg;
  factor.cov_P_V_Phi = this->cov_P_V_Phi;
  factor.delta_t = this->delta_t;

  return factor;
}

