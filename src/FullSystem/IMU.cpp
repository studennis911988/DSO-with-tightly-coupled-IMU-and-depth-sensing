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
    g = gravity_positive;

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
{

}
