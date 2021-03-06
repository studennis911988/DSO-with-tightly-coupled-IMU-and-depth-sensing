/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;


void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;
	adHost = new Mat88[nFrames*nFrames];
	adTarget = new Mat88[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			FrameHessian* host = frames[h]->data;
			FrameHessian* target = frames[t]->data;

			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88::Identity();
			Mat88 AT = Mat88::Identity();

			AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
			AT.topLeftCorner<6,6>() = Mat66::Identity();


			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
			AT(6,6) = -affLL[0];
			AH(6,6) = affLL[0];
			AT(7,7) = -1;
			AH(7,7) = affLL[0];

			AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AH.block<3,8>(3,0) *= SCALE_XI_ROT;
			AH.block<1,8>(6,0) *= SCALE_A;
			AH.block<1,8>(7,0) *= SCALE_B;
			AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AT.block<3,8>(3,0) *= SCALE_XI_ROT;
			AT.block<1,8>(6,0) *= SCALE_A;
			AT.block<1,8>(7,0) *= SCALE_B;

			adHost[h+t*nFrames] = AH;
			adTarget[h+t*nFrames] = AT;
		}
	cPrior = VecC::Constant(setting_initialCalibHessian);


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	adHostF = new Mat88f[nFrames*nFrames];
	adTargetF = new Mat88f[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
			adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
		}

	cPriorF = cPrior.cast<float>();


	EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;


	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS);
	bM = VecX::Zero(CPARS);

  HM_imu = MatXX::Zero(CPARS+7,CPARS+7);
  bM_imu = VecX::Zero(CPARS+7);

  HM_bias = MatXX::Zero(CPARS+7,CPARS+7);
  bM_bias = VecX::Zero(CPARS+7);

	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
	currentLambda=0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames)
	{
		for(EFPoint* p : f->points)
		{
			for(EFResidual* r : p->residualsAll)
			{
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;



	delete accSSE_top_L;
	delete accSSE_top_A;
	delete accSSE_bot;
}




void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			int idx = h+t*nFrames;
			adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					        +frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
		}

	cDeltaF = HCalib->value_minus_value_zero.cast<float>();
	for(EFFrame* f : frames)
	{
		f->delta = f->data->get_state_minus_stateZero().head<8>();
		f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

		for(EFPoint* p : f->points)
			p->deltaF = p->data->idepth-p->data->idepth_zero;
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
				accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,true);
		resInA = accSSE_top_A->nres[0];
	}
	else
	{
		accSSE_top_A->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_A->addPoint<0>(p,this);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,false);
		resInA = accSSE_top_A->nres[0];
	}
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
				accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
		resInL = accSSE_top_L->nres[0];
	}
	else
	{
		accSSE_top_L->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_L->addPoint<1>(p,this);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
		resInL = accSSE_top_L->nres[0];
	}
}





void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
				accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_bot->stitchDoubleMT(red,H,b,this,true);
	}
	else
	{
		accSSE_bot->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_bot->addPoint(p, true);
		accSSE_bot->stitchDoubleMT(red, H, b,this,false);
	}
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>();

	Mat18f* xAd = new Mat18f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();
	for(EFFrame* h : frames)
	{
		h->data->step.head<8>() = - x.segment<8>(CPARS+8*h->idx);
		h->data->step.tail<2>().setZero();

		for(EFFrame* t : frames)
			xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS+8*h->idx).transpose() *   adHostF[h->idx+nFrames*t->idx]
			            + xF.segment<8>(CPARS+8*t->idx).transpose() * adTargetF[h->idx+nFrames*t->idx];
	}

	if(MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
						this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0)
		{
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
		b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
		}

		p->data->step = - b*p->HdiF;
    assert(std::isfinite(p->data->step));
	}
}


double EnergyFunctional::calcMEnergyF()
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2*bM + HM*delta);
}


void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{

	Accumulator11 E;
	E.initialize();
	VecCf dc = cDeltaF;

	for(int i=min;i<max;i++)
	{
		EFPoint* p = allPoints[i];
		float dd = p->deltaF;

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isLinearized || !r->isActive()) continue;

			Mat18f dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
			RawResidualJacobian* rJ = r->J;



			// compute Jp*delta
			float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())
						   +rJ->Jpdc[0].dot(dc)
						   +rJ->Jpdd[0]*dd;

			float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())
						   +rJ->Jpdc[1].dot(dc)
						   +rJ->Jpdd[1]*dd;

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
			__m128 delta_a = _mm_set1_ps((float)(dp[6]));
			__m128 delta_b = _mm_set1_ps((float)(dp[7]));

			for(int i=0;i+3<patternNum;i+=4)
			{
				// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
				__m128 Jdelta =            _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i),Jp_delta_x);
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i),Jp_delta_y));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i),delta_a));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i),delta_b));

				__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF)+i);
				r0 = _mm_add_ps(r0,r0);
				r0 = _mm_add_ps(r0,Jdelta);
				Jdelta = _mm_mul_ps(Jdelta,r0);
				E.updateSSENoShift(Jdelta);
			}
			for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			{
				float Jdelta = rJ->JIdx[0][i]*Jp_delta_x_1 + rJ->JIdx[1][i]*Jp_delta_y_1 +
								rJ->JabF[0][i]*dp[6] + rJ->JabF[1][i]*dp[7];
				E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2*r->res_toZeroF[i])));
			}
		}
		E.updateSingle(p->deltaF*p->deltaF*p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}




double EnergyFunctional::calcLEnergyF_MT()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;
	for(EFFrame* f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

	E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
			this, _1, _2, _3, _4), 0, allPoints.size(), 50);

	return E+red->stats[0];
}



EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

	nResiduals++;
	r->efResidual = efr;
	return efr;
}
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size();
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;

	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8*nFrames+CPARS);
	HM.conservativeResize(8*nFrames+CPARS,8*nFrames+CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();

  bM_imu.conservativeResize(17*nFrames+CPARS+7);
  HM_imu.conservativeResize(17*nFrames+CPARS+7,17*nFrames+CPARS+7);
  bM_imu.tail<17>().setZero();
  HM_imu.rightCols<17>().setZero();
  HM_imu.bottomRows<17>().setZero();

  bM_bias.conservativeResize(17*nFrames+CPARS+7);
  HM_bias.conservativeResize(17*nFrames+CPARS+7,17*nFrames+CPARS+7);
  bM_bias.tail<17>().setZero();
  HM_bias.rightCols<17>().setZero();
  HM_bias.bottomRows<17>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib);
	makeIDX();


	for(EFFrame* fh2 : frames)
	{
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
    /// if EFPoint is from PointHessian with depth from depth camera
    if(ph->hasDepthFromDepthCam){
        efp->hasDepthFromDepthCam = true;
    }
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}


void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();


	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;


    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}



void EnergyFunctional::connectIMUfactor(int idx_marg, int idx_connect2)
{
  FrameShell* fs_connect0 = frames[idx_marg-1]->data->shell;
  FrameShell* fs_marg     = frames[idx_marg]->data->shell;
  FrameShell* fs_connect2 = frames[idx_connect2]->data->shell;
  /// 1. connect to new KF (idx = idx-1)
  // last idx of frame to marg
  std::cout << "1. connect to new KF"<< "\n";

  int id_KF_connect0 = fs_connect0->id;
  // change ref KF
  std::cout << "fh original ref : " << fs_connect2->preintegration_shell.first << "\t";
  fs_connect2->preintegration_shell.first = id_KF_connect0;
  std::cout << "fh : " << fs_connect2->id << "ref change to : " << id_KF_connect0 << "\n";

  /// 2. Integrate factor
  // IMU factor of frame to marg
  std::cout << "2.Integrate factor"<< "\n";

  IMU_PreintegrationShell& factor = fs_marg->preintegration_shell.second;
  IMUPreintegrator temp_imu_preintegrator = IMUPreintegrator(factor);
  // measurement from idx_marg to idx+1
  for(int i=0; i<fs_connect2->dt.size(); i++)
  {
    /** TODO : MT**/
    temp_imu_preintegrator.propagate(fs_connect2->dt[i], fs_connect2->gyro[i]-fs_connect0->bias_g, fs_connect2->acc[i]-fs_connect0->bias_a);
  }
  // update factor
  fs_connect2->preintegration_shell.second = temp_imu_preintegrator.getFactor();

  // 3. update measurement
  std::cout << "3.update measurement"<< "\n";
  std::cout << "original measurement size: "<< fs_connect2->dt.size() << "\n";

  fs_marg->dt.reserve(fs_marg->dt.size() + fs_connect2->dt.size());
  fs_marg->gyro.reserve(fs_marg->gyro.size() + fs_connect2->gyro.size());
  fs_marg->acc.reserve(fs_marg->acc.size() + fs_connect2->acc.size());

  fs_marg->dt.insert(fs_marg->dt.end(), fs_connect2->dt.begin(), fs_connect2->dt.end());
  fs_marg->gyro.insert(fs_marg->gyro.end(), fs_connect2->gyro.begin(), fs_connect2->gyro.end());
  fs_marg->acc.insert(fs_marg->acc.end(), fs_connect2->acc.begin(), fs_connect2->acc.end());

  fs_connect2->dt = fs_marg->dt;
  fs_connect2->gyro = fs_marg->gyro;
  fs_connect2->acc = fs_marg->acc;

  std::cout << "new measurement size: "<< fs_connect2->dt.size() << "\n";
}



void EnergyFunctional::marginalizeFrame_imu(EFFrame* fh){

  int ndim = nFrames*17+CPARS+7-17;// new dimension
  int odim = nFrames*17+CPARS+7;// old dimension

  if(nFrames >= setting_maxFrames){
     imu_track_ready = true;
     std::cout << "*imu_track ready!" << "\n";
  }

  MatXX HM_change = MatXX::Zero(CPARS+7+nFrames*17, CPARS+7+nFrames*17);
  VecX bM_change = VecX::Zero(CPARS+7+nFrames*17);


  for(int i=fh->idx-1;i<fh->idx+1;i++){
      if(i<0 /*|| frames[i+1]->data->shell->noIMUfactor*/)  continue;


      MatXX J_all = MatXX::Zero(9, CPARS+7+nFrames*17);
      VecX r_all = VecX::Zero(9);

      IMU_PreintegrationShell& imu_preCal = frames[i+1]->data->shell->preintegration_shell.second;
      double dt = imu_preCal.delta_t;
      if(dt>0.5)continue;

      FrameHessian* Framei = frames[i]->data;
      FrameHessian* Framej = frames[i+1]->data;

      SE3 worldToCam_i = Framei->get_worldToCam_evalPT();
      SE3 worldToCam_j = Framej->get_worldToCam_evalPT();
      SE3 worldToCam_i2 = Framei->PRE_worldToCam;
      SE3 worldToCam_j2 = Framej->PRE_worldToCam;

      Vec3 g_w;
      g_w << 0,0,-G_norm;

      Mat44 M_WB2 = T_WD.matrix()*worldToCam_i2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
      SE3 T_WB2(M_WB2);
      Mat33 R_WB2 = T_WB2.rotationMatrix();
      Vec3 t_WB2 = T_WB2.translation();

      Mat44 M_WBj2 = T_WD.matrix()*worldToCam_j2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
      SE3 T_WBj2(M_WBj2);
      Mat33 R_WBj2 = T_WBj2.rotationMatrix();
      Vec3 t_WBj2 = T_WBj2.translation();


      Mat33 R_temp = SO3::exp(imu_preCal.J_R_Biasg*Framei->delta_bias_g).matrix();
      Mat33 res_R2 = (imu_preCal.delta_R*R_temp).transpose()*R_WB2.transpose()*R_WBj2;

      Vec3 res_phi2 = SO3(res_R2).log();
      Vec3 res_v2 = R_WB2.transpose()*(Framej->velocity-Framei->velocity-g_w*dt)-
         (imu_preCal.delta_V+imu_preCal.J_V_Biasa*Framei->delta_bias_a+imu_preCal.J_V_Biasg*Framei->delta_bias_g);
      Vec3 res_p2 = R_WB2.transpose()*(t_WBj2-t_WB2-Framei->velocity*dt-0.5*g_w*dt*dt)-
         (imu_preCal.delta_P+imu_preCal.J_P_Biasa*Framei->delta_bias_a+imu_preCal.J_P_Biasg*Framei->delta_bias_g);

      Mat99 Cov = imu_preCal.cov_P_V_Phi;

      Mat33 J_resPhi_phi_i = -IMUPreintegrator::JacobianRInv(res_phi2)*R_WBj2.transpose()*R_WB2;
      Mat33 J_resPhi_phi_j = IMUPreintegrator::JacobianRInv(res_phi2);
      Mat33 J_resPhi_bg = -IMUPreintegrator::JacobianRInv(res_phi2)*SO3::exp(-res_phi2).matrix()*
         IMUPreintegrator::JacobianR(imu_preCal.J_R_Biasg*Framei->delta_bias_g)*imu_preCal.J_R_Biasg;

      Mat33 J_resV_phi_i = SO3::hat(R_WB2.transpose()*(Framej->velocity - Framei->velocity - g_w*dt));
      Mat33 J_resV_v_i = -R_WB2.transpose();
      Mat33 J_resV_v_j = R_WB2.transpose();
      Mat33 J_resV_ba = -imu_preCal.J_V_Biasa;
      Mat33 J_resV_bg = -imu_preCal.J_V_Biasg;

      Mat33 J_resP_p_i = -Mat33::Identity();
      Mat33 J_resP_p_j = R_WB2.transpose()*R_WBj2;
      Mat33 J_resP_bg = -imu_preCal.J_P_Biasg;
      Mat33 J_resP_ba = -imu_preCal.J_P_Biasa;
      Mat33 J_resP_v_i = -R_WB2.transpose()*dt;
      Mat33 J_resP_phi_i = SO3::hat(R_WB2.transpose()*(t_WBj2 - t_WB2 - Framei->velocity*dt - 0.5*g_w*dt*dt));

      Mat915 J_imui = Mat915::Zero();//rho,phi,v,bias_g,bias_a;
      J_imui.block(0,0,3,3) = J_resP_p_i;
      J_imui.block(0,3,3,3) = J_resP_phi_i;
      J_imui.block(0,6,3,3) = J_resP_v_i;
      J_imui.block(0,9,3,3) = J_resP_bg;
      J_imui.block(0,12,3,3) = J_resP_ba;

      J_imui.block(3,3,3,3) = J_resPhi_phi_i;
      J_imui.block(3,9,3,3) = J_resPhi_bg;

      J_imui.block(6,3,3,3) = J_resV_phi_i;
      J_imui.block(6,6,3,3) = J_resV_v_i;
      J_imui.block(6,9,3,3) = J_resV_bg;
      J_imui.block(6,12,3,3) = J_resV_ba;

      Mat915 J_imuj = Mat915::Zero();
      J_imuj.block(0,0,3,3) = J_resP_p_j;
      J_imuj.block(3,3,3,3) = J_resPhi_phi_j;
      J_imuj.block(6,6,3,3)  = J_resV_v_j;

      Mat99 Weight = Mat99::Zero();
      Weight.block(0,0,3,3) = Cov.block(0,0,3,3);
      Weight.block(3,3,3,3) = Cov.block(6,6,3,3);
      Weight.block(6,6,3,3) = Cov.block(3,3,3,3);
      Mat99 Weight2 = Mat99::Zero();
      for(int i=0;i<9;++i){
        Weight2(i,i) = Weight(i,i);
      }
      Weight = Weight2;

      Weight = imu_weight*imu_weight*Weight.inverse();
      Vec9 b_1 = Vec9::Zero();
      b_1.segment<3>(0) = res_p2;
      b_1.segment<3>(3) = res_phi2;
      b_1.segment<3>(6) = res_v2;


      Mat44 T_tempj = T_BC.matrix()*T_WD.matrix()*worldToCam_j.matrix();
      Mat1515 J_relj = Mat1515::Identity();
      J_relj.block(0,0,6,6) = (-1*Sim3(T_tempj).Adj()).block(0,0,6,6);
      Mat44 T_tempi = T_BC.matrix()*T_WD.matrix()*worldToCam_i.matrix();
      Mat1515 J_reli = Mat1515::Identity();
      J_reli.block(0,0,6,6) = (-1*Sim3(T_tempi).Adj()).block(0,0,6,6);


      Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
      Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
      Mat1515 J_r_l_i = Mat1515::Identity();
      Mat1515 J_r_l_j = Mat1515::Identity();
      J_r_l_i.block(0,0,6,6) = J_xi_r_l_i;
      J_r_l_j.block(0,0,6,6) = J_xi_r_l_j;


      J_all.block(0,CPARS+7+i*17,9,6) += J_imui.block(0,0,9,6)*J_reli.block(0,0,6,6)*J_xi_r_l_i;
      J_all.block(0,CPARS+7+(i+1)*17,9,6) += J_imuj.block(0,0,9,6)*J_relj.block(0,0,6,6)*J_xi_r_l_j;
      J_all.block(0,CPARS+7+i*17+8,9,9) += J_imui.block(0,6,9,9);
      J_all.block(0,CPARS+7+(i+1)*17+8,9,9) += J_imuj.block(0,6,9,9);


      r_all.segment<9>(0) += b_1;

      HM_change += (J_all.transpose()*Weight*J_all);
      bM_change += (J_all.transpose()*Weight*r_all);


      MatXX J_all2 = MatXX::Zero(6, CPARS+7+nFrames*17);
      VecX r_all2 = VecX::Zero(6);
      r_all2.segment<3>(0) = Framej->bias_g+Framej->delta_bias_g - (Framei->bias_g+Framei->delta_bias_g);
      r_all2.segment<3>(3) = Framej->bias_a+Framej->delta_bias_a - (Framei->bias_a+Framei->delta_bias_a);

      J_all2.block(0,CPARS+7+i*17+8+3,3,3) = -Mat33::Identity();
      J_all2.block(0,CPARS+7+(i+1)*17+8+3,3,3) = Mat33::Identity();
      J_all2.block(3,CPARS+7+i*17+8+6,3,3) = -Mat33::Identity();
      J_all2.block(3,CPARS+7+(i+1)*17+8+6,3,3) = Mat33::Identity();
      Mat66 Cov_bias = Mat66::Zero();
      Cov_bias.block(0,0,3,3) = GyrRandomWalkNoise*dt;
      Cov_bias.block(3,3,3,3) = AccRandomWalkNoise*dt;
//      std::cout << "Cov_bias" <<"\n" << Cov_bias << "\n";
//      std::cout << "GyrRandomWalkNoise :"  << GyrRandomWalkNoise << "\n";
//      std::cout << "AccRandomWalkNoise :"  << AccRandomWalkNoise << "\n";

      Mat66 weight_bias = Mat66::Identity()*imu_weight*imu_weight*Cov_bias.inverse();
      HM_bias += (J_all2.transpose()*weight_bias*J_all2*setting_margWeightFac_imu);
      bM_bias += (J_all2.transpose()*weight_bias*r_all2*setting_margWeightFac_imu);
  }

  HM_change = HM_change*setting_margWeightFac_imu;
  bM_change = bM_change*setting_margWeightFac_imu;

  for(int i=fh->idx-1;i<fh->idx+1;i++){
      if(i<0)continue;

      double dt = frames[i+1]->data->shell->preintegration_shell.second.delta_t;

      if(dt>0.5)continue;
      if(i==fh->idx-1){
        frames[i]->m_flag = true;
      }
      if(i==fh->idx){
        frames[i+1]->m_flag = true;
      }
  }



  //marginalize bias
  {
  if((int)fh->idx != (int)frames.size()-1)
  {
    int io = fh->idx*17+CPARS+7;	// index of frame to move to end
    int ntail = 17*(nFrames-fh->idx-1);
    assert((io+17+ntail) == nFrames*17+CPARS+7);

    Vec17 bTmp = bM_bias.segment<17>(io);
    VecX tailTMP = bM_bias.tail(ntail);
    bM_bias.segment(io,ntail) = tailTMP;
    bM_bias.tail<17>() = bTmp;

    MatXX HtmpCol = HM_bias.block(0,io,odim,17);
    MatXX rightColsTmp = HM_bias.rightCols(ntail);
    HM_bias.block(0,io,odim,ntail) = rightColsTmp;
    HM_bias.rightCols(17) = HtmpCol;

    MatXX HtmpRow = HM_bias.block(io,0,17,odim);
    MatXX botRowsTmp = HM_bias.bottomRows(ntail);
    HM_bias.block(io,0,ntail,odim) = botRowsTmp;
    HM_bias.bottomRows(17) = HtmpRow;
  }
  VecX SVec = (HM_bias.diagonal().cwiseAbs()+VecX::Constant(HM_bias.cols(), 10)).cwiseSqrt();
  VecX SVecI = SVec.cwiseInverse();

  MatXX HMScaled = SVecI.asDiagonal() * HM_bias * SVecI.asDiagonal();
  VecX bMScaled =  SVecI.asDiagonal() * bM_bias;

  Mat1717 hpi = HMScaled.bottomRightCorner<17,17>();
  hpi = 0.5f*(hpi+hpi);
  hpi = hpi.inverse();
  hpi = 0.5f*(hpi+hpi);
  if(std::isfinite(hpi(0,0))==false){
      hpi = Mat1717::Zero();
  }

  MatXX bli = HMScaled.bottomLeftCorner(17,ndim).transpose() * hpi;
  HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17,ndim);
  bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<17>();

  //unscale!
  HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
  bMScaled = SVec.asDiagonal() * bMScaled;

  // set.
  HM_bias = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
  bM_bias = bMScaled.head(ndim);
  }

  {
    HM_imu += HM_change;
    bM_imu += bM_change;

    if((int)fh->idx != (int)frames.size()-1)
    {
      int io = fh->idx*17+CPARS+7;	// index of frame to move to end
      int ntail = 17*(nFrames-fh->idx-1);
      assert((io+17+ntail) == nFrames*17+CPARS+7);

      Vec17 bTmp = bM_imu.segment<17>(io);
      VecX tailTMP = bM_imu.tail(ntail);
      bM_imu.segment(io,ntail) = tailTMP;
      bM_imu.tail<17>() = bTmp;

      MatXX HtmpCol = HM_imu.block(0,io,odim,17);
      MatXX rightColsTmp = HM_imu.rightCols(ntail);
      HM_imu.block(0,io,odim,ntail) = rightColsTmp;
      HM_imu.rightCols(17) = HtmpCol;

      MatXX HtmpRow = HM_imu.block(io,0,17,odim);
      MatXX botRowsTmp = HM_imu.bottomRows(ntail);
      HM_imu.block(io,0,ntail,odim) = botRowsTmp;
      HM_imu.bottomRows(17) = HtmpRow;
    }
    VecX SVec = (HM_imu.diagonal().cwiseAbs()+VecX::Constant(HM_imu.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();

    MatXX HMScaled = SVecI.asDiagonal() * HM_imu * SVecI.asDiagonal();
    VecX bMScaled =  SVecI.asDiagonal() * bM_imu;

    Mat1717 hpi = HMScaled.bottomRightCorner<17,17>();
    hpi = 0.5f*(hpi+hpi);
    hpi = hpi.inverse();
    hpi = 0.5f*(hpi+hpi);
    if(std::isfinite(hpi(0,0))==false){
      hpi = Mat1717::Zero();
    }

    MatXX bli = HMScaled.bottomLeftCorner(17,ndim).transpose() * hpi;
    HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17,ndim);
    bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<17>();

    //unscale!
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    // set.
    HM_imu = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
    bM_imu = bMScaled.head(ndim);
  }
}

void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);

  if(imu_use_flag)
  {
    marginalizeFrame_imu(fh);
    /// connect IMU factor frames[fh->idx-1] to frames[fh->idx+1]
    ///   idx-1 <------> idx <------> idx+1
    if(fh->idx != 0)
      connectIMUfactor(fh->idx, fh->idx+1);
  }

	int ndim = nFrames*8+CPARS-8;// new dimension
	int odim = nFrames*8+CPARS;// old dimension


//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//



	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*8+CPARS;	// index of frame to move to end
		int ntail = 8*(nFrames-fh->idx-1);
		assert((io+8+ntail) == nFrames*8+CPARS);

		Vec8 bTmp = bM.segment<8>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io,ntail) = tailTMP;
		bM.tail<8>() = bTmp;

		MatXX HtmpCol = HM.block(0,io,odim,8);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		MatXX HtmpRow = HM.block(io,0,8,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8,8>().diagonal() += fh->prior;
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);



//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";


	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();


//	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
//	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

	// scale!
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);

  if(std::isfinite(hpi(0,0))==false){
      hpi = Mat88::Zero();
  }

	// schur-complement!
	MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);

	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++)
	{
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);




//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}




void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);


	allPointsToMarg.clear();
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE)
			{
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}

	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);
	for(EFPoint* p : allPointsToMarg)
	{
		accSSE_top_A->addPoint<2>(p,this);
		accSSE_bot->addPoint(p,false);
		removePoint(p);
	}
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accSSE_top_A->stitchDouble(M,Mb,this,false,false);
	accSSE_bot->stitchDouble(Msc,Mbsc,this);

	resInM+= accSSE_top_A->nres[0];

	MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;

		if(!haveFirstFrame)
			orthogonalize(&b, &H);

	}

	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bM, &HM);

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{


	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP)
			{
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";


	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());





	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();



	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++)
	{
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;


//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

}


void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc;
	VecX  bL_top, bA_top, bM_top, b_sc;

	accumulateAF_MT(HA_top, bA_top,multiThreading);

	accumulateLF_MT(HL_top, bL_top,multiThreading);

	accumulateSCF_MT(H_sc, b_sc,multiThreading);

  VecX delta_update = getStitchedDeltaF();

  bM_top = (bM+ HM * delta_update);

  VecX delta_update_imu = VecX::Zero(CPARS+7+nFrames*17);
  for(int i=0;i<nFrames;++i){
      if(frames[i]->m_flag){
        delta_update_imu.segment(CPARS+7+17*i,6) = delta_update.segment(CPARS+8*i,6);
      }
  }

  VecX bM_top_imu = (bM_imu + HM_imu*delta_update_imu);

  MatXX H_imu;
  VecX  b_imu;
  calculateIMUHessian(H_imu, b_imu);
  //std::cout << "Himu" <<"\n" << H_imu << "\n";
  //std::cout << "bimu" <<"\n" << b_imu.transpose() << "\n";


	MatXX HFinal_top;
	VecX bFinal_top;
	if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;
		MatXX HT_act =  HL_top + HA_top - H_sc;
		VecX bT_act =   bL_top + bA_top - b_sc;

		if(!haveFirstFrame)
			orthogonalize(&bT_act, &HT_act);

		HFinal_top = HT_act + HM;
		bFinal_top = bT_act + bM_top;

		lastHS = HFinal_top;
		lastbS = bFinal_top;

    for(int i=0;i<8*nFrames+CPARS;i++)
       HFinal_top(i,i) *= (1+lambda);
	}
	else
	{
		HFinal_top = HL_top + HM + HA_top;
		bFinal_top = bL_top + bM_top + bA_top - b_sc;

		lastHS = HFinal_top - H_sc;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);
		HFinal_top -= H_sc * (1.0f/(1+lambda));
	}


  for(int i=0;i<7+15*nFrames;i++)H_imu(i,i)*= (1+lambda);

  //imu_term
  MatXX HFinal_top2 =  MatXX::Zero(CPARS+7+17*nFrames,CPARS+7+17*nFrames);//Cam,Twd,pose,a,b,v,bg,ba
  VecX bFinal_top2 = VecX::Zero(CPARS+7+17*nFrames);
  HFinal_top2.block(0,0,CPARS,CPARS) = HFinal_top.block(0,0,CPARS,CPARS);
  HFinal_top2.block(CPARS,CPARS,7,7) = H_imu.block(0,0,7,7);
  bFinal_top2.segment(0,CPARS) = bFinal_top.segment(0,CPARS);
  bFinal_top2.segment(CPARS,7) = b_imu.segment(0,7);


  for(int i=0;i<nFrames;++i){
      //cam
      HFinal_top2.block(0,CPARS+7+i*17,CPARS,8) += HFinal_top.block(0,CPARS+i*8,CPARS,8);
      HFinal_top2.block(CPARS+7+i*17,0,8,CPARS) += HFinal_top.block(CPARS+i*8,0,8,CPARS);
      //Twd
      HFinal_top2.block(CPARS,CPARS+7+i*17,7,6) += H_imu.block(0,7+i*15,7,6);
      HFinal_top2.block(CPARS+7+i*17,CPARS,6,7) += H_imu.block(7+i*15,0,6,7);
      HFinal_top2.block(CPARS,CPARS+7+i*17+8,7,9) += H_imu.block(0,7+i*15+6,7,9);
      HFinal_top2.block(CPARS+7+i*17+8,CPARS,9,7) += H_imu.block(7+i*15+6,0,9,7);
      //pose a b
      HFinal_top2.block(CPARS+7+i*17,CPARS+7+i*17,8,8) += HFinal_top.block(CPARS+i*8,CPARS+i*8,8,8);
      //pose
      HFinal_top2.block(CPARS+7+i*17,CPARS+7+i*17,6,6) += H_imu.block(7+i*15,7+i*15,6,6);
      //v bg ba
      HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+i*17+8,9,9) += H_imu.block(7+i*15+6,7+i*15+6,9,9);
      //v bg ba,pose
      HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+i*17,9,6) += H_imu.block(7+i*15+6,7+i*15,9,6);
      //pose,v bg ba
      HFinal_top2.block(CPARS+7+i*17,CPARS+7+i*17+8,6,9) += H_imu.block(7+i*15,7+i*15+6,6,9);

  for(int j=i+1;j<nFrames;++j){
    //pose a b
    HFinal_top2.block(CPARS+7+i*17,CPARS+7+j*17,8,8) += HFinal_top.block(CPARS+i*8,CPARS+j*8,8,8);
    HFinal_top2.block(CPARS+7+j*17,CPARS+7+i*17,8,8) += HFinal_top.block(CPARS+j*8,CPARS+i*8,8,8);
    //pose
    HFinal_top2.block(CPARS+7+i*17,CPARS+7+j*17,6,6) += H_imu.block(7+i*15,7+j*15,6,6);
    HFinal_top2.block(CPARS+7+j*17,CPARS+7+i*17,6,6) += H_imu.block(7+j*15,7+i*15,6,6);
    //v bg ba
    HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+j*17+8,9,9) += H_imu.block(7+i*15+6,7+j*15+6,9,9);
    HFinal_top2.block(CPARS+7+j*17+8,CPARS+7+i*17+8,9,9) += H_imu.block(7+j*15+6,7+i*15+6,9,9);
    //v bg ba,pose
    HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+j*17,9,6) += H_imu.block(7+i*15+6,7+j*15,9,6);
    HFinal_top2.block(CPARS+7+j*17,CPARS+7+i*17+8,6,9) += H_imu.block(7+j*15,7+i*15+6,6,9);
    //pose,v bg ba
    HFinal_top2.block(CPARS+7+i*17,CPARS+7+j*17+8,6,9) += H_imu.block(7+i*15,7+j*15+6,6,9);
    HFinal_top2.block(CPARS+7+j*17+8,CPARS+7+i*17,9,6) += H_imu.block(7+j*15+6,7+i*15,9,6);
   }
      bFinal_top2.segment(CPARS+7+17*i,8) += bFinal_top.segment(CPARS+8*i,8);
      bFinal_top2.segment(CPARS+7+17*i,6) += b_imu.segment(7+15*i,6);
      bFinal_top2.segment(CPARS+7+17*i+8,9) += b_imu.segment(7+15*i+6,9);
  }
  HFinal_top2 += (HM_imu + HM_bias);
  bFinal_top2 += (bM_top_imu + bM_bias);



  bool HFinal_top2_finite = HFinal_top2.allFinite();
  bool bFinal_top2_finite = bFinal_top2.allFinite();
  bool HFinal_top_finite = HFinal_top.allFinite();
  bool bFinal_top_finite = bFinal_top.allFinite();
  bool HM_finite = HM.allFinite();
  bool bM_finite = bM.allFinite();
  bool H_imu_finite = H_imu.allFinite();
  bool b_imu_finite = b_imu.allFinite();
  bool HM_imu_finite = HM_imu.allFinite();
  bool bM_imu_finite = bM_imu.allFinite();
  bool HM_bias_finite = HM_bias.allFinite();
  bool bM_bias_finite = bM_bias.allFinite();
  bool bM_top_finite = bM_top.allFinite();
  bool bM_top_imu_finite = bM_top_imu.allFinite();


//  std::cout << "HFinal_top2_finite? : " << HFinal_top2_finite << "\n";
//  std::cout << "bFinal_top2_finite? : " << bFinal_top2_finite << "\n";
//  std::cout << "HFinal_top_finite? : " << HFinal_top_finite << "\n";
//  std::cout << "bFinal_top_finite? : " << bFinal_top_finite << "\n";
//  std::cout << "HM_finite? : " << HM_finite << "\n";
//  std::cout << "bM_finite? : " << bM_finite << "\n";
//  std::cout << "H_imu_finite? : " << H_imu_finite << "\n";
//  std::cout << "b_imu_finite? : " << b_imu_finite << "\n";
//  std::cout << "HM_imu_finite? : " << HM_imu_finite << "\n";
//  std::cout << "bM_imu_finite? : " << bM_imu_finite << "\n";
//  std::cout << "HM_bias_finite? : " << HM_bias_finite << "\n";
//  std::cout << "bM_bias_finite? : " << bM_bias_finite << "\n";
//  std::cout << "bM_top_finite? : " << bM_top_finite << "\n";
//  std::cout << "bM_top_imu_finite? : " << bM_top_imu_finite << "\n";
//  if(H_hasNAN)
//  {
//    std::cout << "**** H_FINALTOP:" << "\n" << HFinal_top2 << "\n"
//              << "**** H_imu" << "\n" << H_imu << "\n"
//              << "**** HM_imu" << "\n" << HM_imu << "\n"
//              << "**** HM_bias" << "\n" << HM_bias << "\n";
//  }
//  if(b_hasNAN)
//  {
//    std::cout << "**** b_FINALTOP:" << "\n" << bFinal_top2.transpose() << "\n"
//              << "**** b_imu" << "\n" << b_imu.transpose() << "\n"
//              << "**** bM_imu" << "\n" << bM_imu.transpose() << "\n"
//              << "**** bM_bias" << "\n" << bM_bias.transpose() << "\n";
//  }


//  std::cout << "HM_bias" <<"\n" << HM_bias << "\n";
  //std::cout << "bM_top_imu" <<"\n" << bM_top_imu.transpose() << "\n";
  //std::cout << "bM_bias" <<"\n" << bM_bias.transpose() << "\n";



  VecX x = VecX::Zero(CPARS+8*nFrames);
  VecX x2= VecX::Zero(CPARS+7+17*nFrames);

	if(setting_solverMode & SOLVER_SVD)
	{
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }

			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;

	}
	else
	{
    if(!imu_use_flag){
        VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
    }
    else{
        VecX SVecI = (HFinal_top2.diagonal()+VecX::Constant(HFinal_top2.cols(), 10)).cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top2 * SVecI.asDiagonal();
        x2 = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top2);//  SVec.asDiagonal() * svd.matrixV() * Ub;
        std::cout << "step x2" <<"\n" << x2.transpose() << "\n";

        x.block(0,0,CPARS,1) = x2.block(0,0,CPARS,1);
        for(int i=0;i<nFrames;++i)
        {
          x.block(CPARS+i*8,0,8,1) = x2.block(CPARS+7+17*i,0,8,1);
          frames[i]->data->step_imu = -x2.block(CPARS+7+17*i+8,0,9,1);
          std::cout << "step imu fh: " << frames[i]->data->shell->id
                    <<"\t" << frames[i]->data->step_imu.transpose() << "\n";

        }
    }
	}

	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
	{
		VecX xOld = x;
		orthogonalize(&x, 0);
	}

	lastX = x;

	//resubstituteF(x, HCalib);
	currentLambda= lambda;
	resubstituteF_MT(x, HCalib,multiThreading);
	currentLambda=0;
}

void EnergyFunctional::calculateIMUHessian(MatXX &H, VecX &b){
  H = MatXX::Zero(7+nFrames*15, 7+nFrames*15);
  b = VecX::Zero(7+nFrames*15);

  if(nFrames==1)  return;

  int count_imu_res = 0;
  Energy_imu = 0;

  for(int i=0;i<frames.size()-1;++i)
  {
//    if(frames[i+1]->data->shell->noIMUfactor) continue;

    MatXX J_all = MatXX::Zero(9, 7+nFrames*15);
    VecX r_all = VecX::Zero(9);

    //preintegrate

    IMU_PreintegrationShell& imu_preCal = frames[i+1]->data->shell->preintegration_shell.second;

    double dt = imu_preCal.delta_t;


    count_imu_res++;

    FrameHessian* Framei = frames[i]->data;
    FrameHessian* Framej = frames[i+1]->data;

    //bias model
    MatXX J_all2 = MatXX::Zero(6, 7+nFrames*15);
    VecX r_all2 = VecX::Zero(6);

    r_all2.block(0,0,3,1) = Framej->bias_g+Framej->delta_bias_g - (Framei->bias_g+Framei->delta_bias_g);
    r_all2.block(3,0,3,1) = Framej->bias_a+Framej->delta_bias_a - (Framei->bias_a+Framei->delta_bias_a);

    J_all2.block(0,7+i*15+9,3,3) = -Mat33::Identity();
    J_all2.block(0,7+(i+1)*15+9,3,3) = Mat33::Identity();
    J_all2.block(3,7+i*15+12,3,3) = -Mat33::Identity();
    J_all2.block(3,7+(i+1)*15+12,3,3) = Mat33::Identity();
    Mat66 Cov_bias = Mat66::Zero();
    Cov_bias.block(0,0,3,3) = GyrRandomWalkNoise*dt;
    Cov_bias.block(3,3,3,3) = AccRandomWalkNoise*dt;

//    std::cout << "cov:" << Cov_bias << "\n";
    Mat66 weight_bias = Mat66::Identity()*imu_weight*imu_weight*Cov_bias.inverse();

    H += J_all2.transpose()*weight_bias*J_all2;
    b += J_all2.transpose()*weight_bias*r_all2;

    if(dt>0.5)continue;    // throw pose and velocity constrain if the keyframe time gap is larger
                            // than 0.5 sec, but keep the bias constrain

    SE3 worldToCam_i = Framei->get_worldToCam_evalPT();
    SE3 worldToCam_j = Framej->get_worldToCam_evalPT();
    SE3 worldToCam_i2 = Framei->PRE_worldToCam;
    SE3 worldToCam_j2 = Framej->PRE_worldToCam;


    Vec3 g_w;
    g_w << 0,0,-G_norm;

    Mat44 M_WB = T_WD.matrix()*worldToCam_i.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
    SE3 T_WB(M_WB);
    Mat33 R_WB = T_WB.rotationMatrix();
    Vec3 t_WB = T_WB.translation();

    Mat44 M_WB2 = T_WD.matrix()*worldToCam_i2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
    SE3 T_WB2(M_WB2);
    Mat33 R_WB2 = T_WB2.rotationMatrix();
    Vec3 t_WB2 = T_WB2.translation();

    Mat44 M_WBj = T_WD.matrix()*worldToCam_j.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
     SE3 T_WBj(M_WBj);
    Mat33 R_WBj = T_WBj.rotationMatrix();
    Vec3 t_WBj = T_WBj.translation();

    Mat44 M_WBj2 = T_WD.matrix()*worldToCam_j2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
    SE3 T_WBj2(M_WBj2);
    Mat33 R_WBj2 = T_WBj2.rotationMatrix();
    Vec3 t_WBj2 = T_WBj2.translation();


    Mat33 R_temp = SO3::exp(imu_preCal.J_R_Biasg*Framei->delta_bias_g).matrix();
    Mat33 res_R2 = (imu_preCal.delta_R*R_temp).transpose()*R_WB2.transpose()*R_WBj2;

    Vec3 res_phi2 = SO3(res_R2).log();
    Vec3 res_v2 = R_WB2.transpose()*(Framej->velocity-Framei->velocity-g_w*dt)-
     (imu_preCal.delta_V+imu_preCal.J_V_Biasa*Framei->delta_bias_a+imu_preCal.J_V_Biasg*Framei->delta_bias_g);
    Vec3 res_p2 = R_WB2.transpose()*(t_WBj2-t_WB2-Framei->velocity*dt-0.5*g_w*dt*dt)-
     (imu_preCal.delta_P+imu_preCal.J_P_Biasa*Framei->delta_bias_a+imu_preCal.J_P_Biasg*Framei->delta_bias_g);

    Mat99 Cov = imu_preCal.cov_P_V_Phi;

    Mat33 J_resPhi_phi_i = -IMUPreintegrator::JacobianRInv(res_phi2)*R_WBj2.transpose()*R_WB2;
    Mat33 J_resPhi_phi_j = IMUPreintegrator::JacobianRInv(res_phi2);
    Mat33 J_resPhi_bg = -IMUPreintegrator::JacobianRInv(res_phi2)*SO3::exp(-res_phi2).matrix()*
     IMUPreintegrator::JacobianR(imu_preCal.J_R_Biasg*Framei->delta_bias_g)*imu_preCal.J_R_Biasg;

    Mat33 J_resV_phi_i = SO3::hat(R_WB2.transpose()*(Framej->velocity - Framei->velocity - g_w*dt));
    Mat33 J_resV_v_i = -R_WB2.transpose();
    Mat33 J_resV_v_j = R_WB2.transpose();
    Mat33 J_resV_ba = -imu_preCal.J_V_Biasa;
    Mat33 J_resV_bg = -imu_preCal.J_V_Biasg;

    Mat33 J_resP_p_i = -Mat33::Identity();
    Mat33 J_resP_p_j = R_WB2.transpose()*R_WBj2;
    Mat33 J_resP_bg = -imu_preCal.J_P_Biasg;
    Mat33 J_resP_ba = -imu_preCal.J_P_Biasa;
    Mat33 J_resP_v_i = -R_WB2.transpose()*dt;
    Mat33 J_resP_phi_i = SO3::hat(R_WB2.transpose()*(t_WBj2 - t_WB2 - Framei->velocity*dt - 0.5*g_w*dt*dt));



    Mat915 J_imui = Mat915::Zero();//rho,phi,v,bias_g,bias_a;
    J_imui.block(0,0,3,3) = J_resP_p_i;
    J_imui.block(0,3,3,3) = J_resP_phi_i;
    J_imui.block(0,6,3,3) = J_resP_v_i;
    J_imui.block(0,9,3,3) = J_resP_bg;
    J_imui.block(0,12,3,3) = J_resP_ba;

    J_imui.block(3,3,3,3) = J_resPhi_phi_i;
    J_imui.block(3,9,3,3) = J_resPhi_bg;

    J_imui.block(6,3,3,3) = J_resV_phi_i;
    J_imui.block(6,6,3,3) = J_resV_v_i;
    J_imui.block(6,9,3,3) = J_resV_bg;
    J_imui.block(6,12,3,3) = J_resV_ba;


    Mat915 J_imuj = Mat915::Zero();
    J_imuj.block(0,0,3,3) = J_resP_p_j;
    J_imuj.block(3,3,3,3) = J_resPhi_phi_j;
    J_imuj.block(6,6,3,3)  = J_resV_v_j;

    Mat99 Weight = Mat99::Zero();
    Weight.block(0,0,3,3) = Cov.block(0,0,3,3);
    Weight.block(3,3,3,3) = Cov.block(6,6,3,3);
    Weight.block(6,6,3,3) = Cov.block(3,3,3,3);
//    std::cout << "Weight:" << Weight << "\n";

    Mat99 Weight2 = Mat99::Zero();
    for(int i=0;i<9;++i){
        Weight2(i,i) = Weight(i,i);
    }
    Weight = Weight2;
    Weight = imu_weight*imu_weight*Weight.inverse();

    Vec9 b_1 = Vec9::Zero();
    b_1.block(0,0,3,1) = res_p2;
    b_1.block(3,0,3,1) = res_phi2;
    b_1.block(6,0,3,1) = res_v2;

    Mat44 T_tempj = T_BC.matrix()*T_WD.matrix()*worldToCam_j.matrix();  // worldToCam_j :FEJ
    Mat1515 J_relj = Mat1515::Identity();
    J_relj.block(0,0,6,6) = (-1*Sim3(T_tempj).Adj()).block(0,0,6,6);
    Mat44 T_tempi = T_BC.matrix()*T_WD.matrix()*worldToCam_i.matrix();
    Mat1515 J_reli = Mat1515::Identity();
    J_reli.block(0,0,6,6) = (-1*Sim3(T_tempi).Adj()).block(0,0,6,6);


    Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
    Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
    Mat1515 J_r_l_i = Mat1515::Identity();
    Mat1515 J_r_l_j = Mat1515::Identity();
    J_r_l_i.block(0,0,6,6) = J_xi_r_l_i;
    J_r_l_j.block(0,0,6,6) = J_xi_r_l_j;


    J_all.block(0,7+i*15,9,15) += J_imui*J_reli*J_r_l_i;
    J_all.block(0,7+(i+1)*15,9,15) += J_imuj*J_relj*J_r_l_j;  // compare to front-end : no need J_xi_tw_th

    r_all.block(0,0,9,1) += b_1;

    H += (J_all.transpose()*Weight*J_all);
    b += (J_all.transpose()*Weight*r_all);

    Energy_imu =  Energy_imu + (r_all.transpose()*Weight*r_all)+(r_all2.transpose()*weight_bias*r_all2);
  }


  for(int i=0;i<nFrames;i++)
  {
    H.block(0,7+i*15,7+nFrames*15,3) *= SCALE_XI_TRANS;
    H.block(7+i*15,0,3,7+nFrames*15) *= SCALE_XI_TRANS;
    b.block(7+i*15,0,3,1) *= SCALE_XI_TRANS;

    H.block(0,7+i*15+3,7+nFrames*15,3) *= SCALE_XI_ROT;
    H.block(7+i*15+3,0,3,7+nFrames*15) *= SCALE_XI_ROT;
    b.block(7+i*15+3,0,3,1) *= SCALE_XI_ROT;
   }
}


void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points)
		{
			allPoints.push_back(p);
			for(EFResidual* r : p->residualsAll)
			{
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
			}
		}


	EFIndicesValid=true;
}


VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*8); d.head<CPARS>() = cDeltaF.cast<double>();
	for(int h=0;h<nFrames;h++) d.segment<8>(CPARS+8*h) = frames[h]->delta;
	return d;
}



}
