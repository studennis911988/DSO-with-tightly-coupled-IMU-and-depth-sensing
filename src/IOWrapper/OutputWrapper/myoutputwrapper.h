/**
*  Implementation of my own output wrapper which mainly publish camera pose
*  Auther : Dennis 2019/5/19
*/

#ifndef MYOUTPUTWRAPPER_H
#define MYOUTPUTWRAPPER_H

#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "Eigen/Core"
#include "sophus/se3.hpp"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap
{

class MyOutputWrapper : public Output3DWrapper
{
public:
        inline MyOutputWrapper()
        {
            printf("OUT: Created MyOutputWrapper\n");
            depth_image_ptr = nullptr;
        }

        virtual ~MyOutputWrapper()
        {
            printf("OUT: Destroyed MyOutputWrapper\n");
        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
            /** save data **/
            voTrackingPose = frame->camToWorld.matrix3x4();
            voTimestamp = frame->timestamp;
            voId = frame->id;
        }

        virtual void pushDepthImage(MinimalImageB3* image) override
        {
             depth_image_ptr = image;
        }

        virtual bool needPushDepthImage() override
        {
            if(publish_depth_image){
                return true;
            }
            else{
                return false;
            }
        }


        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override{

            // inverse intrinsic
            float fx  = HCalib->fxl();
            float fy  = HCalib->fyl();
            float cx  = HCalib->cxl();
            float cy  = HCalib->cyl();
            float fxi = 1 / fx;
            float fyi = 1 / fy;
            float cxi = - cx / fx;
            float cyi = - cy / fy;

            if(final == true && publish_point_cloud){
                for(FrameHessian* f : frames){
                    // clear point cloud
                    pointCloud.clear();
                    // camera frame to world frame
                    const Eigen::Matrix<double,3,4> cam2world =  f->shell->camToWorld.matrix3x4();

                    // get all actie points in this frame
    //                    std::vector<PointHessian*>* activePoints =  &(f->pointHessians);
                    std::vector<PointHessian*>* activePoints =  &(f->pointHessiansMarginalized);

                    for(PointHessian* p : *activePoints){
                        // pixel frame to camera frame
                        float depth = 1.0f / p->idepth;
                        const double x = (p->u * fxi + cxi) * depth;
                        const double y = (p->v * fyi + cyi) * depth;
                        const double z = depth;

                        Eigen::Vector4d pixel2cam   = Eigen::Vector4d(x, y, z, 1.0f); // homogenous
                        Eigen::Vector3d pixel2world = cam2world * pixel2cam ;
                        // use sensor message directly ?
                        pointCloud.push_back(pixel2world);
                    }
                }
            }

         }


        void setPublishDepthImage(bool publish){
            publish_depth_image = publish;
        }

        void setPublishPointCloud(bool publish){
            publish_point_cloud = publish;
        }


public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // camera pose
        Eigen::Matrix<double,3,4> voTrackingPose;
        double voTimestamp;
        int voId;

        // depth map
        MinimalImageB3* depth_image_ptr;
        bool publish_depth_image = false;

        // point cloud
        std::vector<Eigen::Vector3d> pointCloud;
        bool publish_point_cloud = false;
};



}



}

























#endif // MYOUTPUTWRAPPER_H
