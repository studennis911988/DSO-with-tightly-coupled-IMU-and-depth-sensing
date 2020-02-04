#ifndef IMU_H
#define IMU_H

/// There are two IMU class
/// 1. IMU integrator : for motion prior
/// 2. IMU preintegration : for residual optimization

#include "util/NumType.h"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include <Eigen/Dense>


namespace dso
{


class IMU
{
public:
    inline Mat33 normalizeRotationM (const Mat33& R)
    {
        Sophus::Quaterniond qr(R);
        return normalizeRotationQ(qr).toRotationMatrix();
    }

    inline Sophus::Quaterniond normalizeRotationQ(const Sophus::Quaterniond& r)
    {
        Sophus::Quaterniond _r(r);
        if (_r.w()<0)
        {
            _r.coeffs() *= -1;
        }
        return _r.normalized();
    }

    inline Mat33 Expmap(const Vec3& v)
    {
        return Sophus::SO3::exp(v).matrix();
    }

};

class IMUIntegrator : public IMU
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // constructor
    IMUIntegrator();
    ~IMUIntegrator();

    // reset all state
    void reset();

    // set predict reference
    void setPredictReference(const SE3& T_k_1, const Vec3 v_k_1, const Vec3& gyro_bias, const Vec3& acc_bias);

    // predict motion prior for new frame at tracking
    void predict(double dt, const Vec3& gyro_k,  const Vec3& acc_k);

    // get private class member
    inline Mat33 get_R_prior() const
    {
        return R_prior;
    }

    inline Vec3 get_t_prior() const
    {
        return t_prior;
    }

    inline Vec3 get_v_prior() const
    {
        return v_prior;
    }

    inline Vec3 getLastGyro() const
    {
        return gyro_k_1;
    }

    inline Vec3 getLastAcc() const
    {
        return acc_k_1;
    }


private:

    // fist imu did not have k-1's data
    bool get_first_imu;

    // k-1 's angular vel & linear accel  (midpoint method)
    Vec3 gyro_k_1;
    Vec3 acc_k_1;

    // bias
    Vec3 gyro_b;
    Vec3 acc_b;

    // gravity
    Vec3 g;

    // motion prior for new tracking frame (note : relative to last frame)
    // lastKF2newF(motion prior) = newF - lastKF
    // I keep bias same as reference keyframe(lastKF)
    Mat33 R_prior;
    Vec3  t_prior;
    Vec3  v_prior;
};


class IMUPreintegrator : public IMU
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUPreintegrator();
    ~IMUPreintegrator();

    // reset all state
    void reset();

    // preintegration for IMU factor
    void propagate(double dt, const Vec3& gyro_k,  const Vec3& acc_k);

    //

    // delta measurements, position/velocity/rotation(matrix)
    inline Vec3 getDeltaP() const    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    {
        return delta_P;
    }
    inline Vec3 getDeltaV() const    // V_k+1 = V_k + R_k*a_k*dt
    {
        return delta_V;
    }
    inline Mat33 getDeltaR() const   // R_k+1 = R_k*exp(w_k*dt).     NOTE: Rwc, Rwc'=Rwc*[w_body]x
    {
        return delta_R;
    }

    inline Mat33 getJPBiasg() const     // position / gyro
    {
        return J_P_Biasg;
    }
    inline Mat33 getJPBiasa() const     // position / acc
    {
        return J_P_Biasa;
    }
    inline Mat33 getJVBiasg() const     // velocity / gyro
    {
        return J_V_Biasg;
    }
    inline Mat33 getJVBiasa() const     // velocity / acc
    {
        return J_V_Biasa;
    }
    inline Mat33 getJRBiasg() const  // rotation / gyro
    {
        return J_R_Biasg;
    }

    // noise covariance propagation of delta measurements
    // note: the order is rotation-velocity-position here
    inline Mat99 getCovPVPhi() const
    {
        return cov_P_V_Phi;
    }

private:
    // preintegration term : position/velocity/rotation(matrix)
    Vec3 delta_P;
    Vec3 delta_V;
    Mat33 delta_R;

    // Jacobian of preintegration term w.r.t gyro/acc bias for correction
    Mat33 J_P_Biasg;
    Mat33 J_P_Biasa;
    Mat33 J_V_Biasg;
    Mat33 J_V_Biasa;
    Mat33 J_R_Biasg;
    Mat33 J_R_Biasa;

    // noise covariance for weighting matrix in optimization
    Mat99 cov_P_V_Phi;
};














}


#endif // IMU_H
