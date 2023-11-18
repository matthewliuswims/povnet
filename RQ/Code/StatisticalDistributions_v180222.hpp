//g++ -O2 -I/Users/eita/boost_1_63_0 -I/Users/eita/Dropbox/Research/Tool/All/ some.cpp -o some
#ifndef StatisticalDistributions_HPP
#define StatisticalDistributions_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<cassert>
#include<algorithm>
#include<boost/math/distributions.hpp> 
#include<boost/random.hpp>
#include"BasicCalculation_v170122.hpp"

using namespace std;

class GaussianModel{
public:
	double mean;
	double stdev;
	boost::mt19937 gen;
	boost::random::normal_distribution<> dist;

	GaussianModel(double mean_,double stdev_,int seed_=0){
		SetSeed(seed_);
		SetParam(mean_,stdev_);
	}//end GaussianModel
	~GaussianModel(){
	}//end ~GaussianModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void SetParam(double mean_,double stdev_){
		mean=mean_;
		stdev=stdev_;
		dist=boost::random::normal_distribution<>(mean,stdev);
	}//end SetSeed

	double Sample(){
		return dist(gen);
	}//end Sample

	double GetLP(double x){
		return -0.5*log(2*M_PI*stdev*stdev)-0.5*pow((x-mean)/stdev,2.);
	}//endfor x

};//endclass GaussianModel

class UniformModel{
public:
	double min;
	double max;
	boost::mt19937 gen;
	boost::random::uniform_01<> dist;

	UniformModel(double min_,double max_,int seed_=0){
		SetSeed(seed_);
		SetParam(min_,max_);
	}//end UniformModel
	~UniformModel(){
	}//end ~UniformModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void SetParam(double min_,double max_){
		min=min_;
		max=max_;
	}//end SetSeed

	double Sample(){
		return (max-min)*dist(gen)+min;
	}//end Sample

};//endclass UniformModel

class CauchyModel{
public:
	double median;
	double sigma;
	boost::mt19937 gen;
	boost::random::cauchy_distribution<> dist;

//	CauchyModel(double median_,double sigma_,int seed_=0):unifDistr(0,1,seed_){
	CauchyModel(double median_,double sigma_,int seed_=0){
		SetSeed(seed_);
		SetParam(median_,sigma_);
	}//end CauchyModel
	~CauchyModel(){
	}//end ~CauchyModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void SetParam(double median_,double sigma_){
		median=median_;
		sigma=sigma_;
		dist=boost::random::cauchy_distribution<>(median,sigma);
	}//end SetParam

	double Sample(){
		return dist(gen);
	}//end Sample

};//endclass CauchyModel



class LogNormalModel{
public:
	double m;//m>0
	double s;//s>0
	double mean;
	double stdev;
	boost::mt19937 gen;
	boost::random::lognormal_distribution<> dist;

	LogNormalModel(double m_,double s_,int seed_=0){
		SetSeed(seed_);
		SetParam(m_,s_);
	}//end LogNormalModel
	~LogNormalModel(){
	}//end ~LogNormalModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void MSToMeanStdev(){
		mean=m*exp(s*s/2.);
		stdev=mean*pow(exp(s*s)-1,0.5);
	}//end MSToMeanStdev

	void MeanStdevToMS(){
		s=pow(log(pow(stdev/mean,2.)+1),0.5);
		m=mean*pow(pow(stdev/mean,2.)+1,-0.5);
	}//end MeanStdevToMS

	void SetParam(double m_,double s_){
		m=m_;
		s=s_;
		MSToMeanStdev();
		dist=boost::random::lognormal_distribution<>(log(m),s);
	}//end SetParam

	void SetParamFromMeanStdev(double mean_,double stdev_){
		mean=mean_;
		stdev=stdev_;
		MeanStdevToMS();
		dist=boost::random::lognormal_distribution<>(log(m),s);
	}//end SetParamFromMeanStdev

	double Sample(){
		return dist(gen);
	}//end Sample

	double GetProb(double x){//1/sqrt(2*M_PI)=0.3989422804
		return 0.3989422804/(s*x)*exp(-0.5*pow((log(x)-log(m))/s,2.));
	}//end GetProb

	double GetLogProb(double x){//ln(1/sqrt(2*M_PI))=-0.9189385332
		return -0.9189385332-log(s*x)-0.5*pow((log(x)-log(m))/s,2.);
	}//end GetLogProb


};//endclass LogNormalModel

class GammaModel{
public:
	double a;//a>0
	double b;//b>0
	double mean;
	double stdev;
	boost::mt19937 gen;
	boost::random::gamma_distribution<> dist;

	GammaModel(double a_,double b_,int seed_=0){
		SetSeed(seed_);
		SetParam(a_,b_);
	}//end GammaModel
	~GammaModel(){
	}//end ~GammaModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void ABToMeanStdev(){
		mean=a*b;
		stdev=b*pow(a,0.5);
	}//end ABToMeanStdev

	void MeanStdevToAB(){
		a=pow(mean/stdev,2.);
		b=mean/a;
	}//end MeanStdevToAB

	void SetParam(double a_,double b_){
		a=a_;
		b=b_;
		ABToMeanStdev();
		dist=boost::random::gamma_distribution<>(a,b);
	}//end SetParam

	void SetParamFromMeanStdev(double mean_,double stdev_){
		mean=mean_;
		stdev=stdev_;
		MeanStdevToAB();
		dist=boost::random::gamma_distribution<>(a,b);
	}//end SetParamFromMeanStdev

	double Sample(){
		return dist(gen);
	}//end Sample

	double GetProb(double x){
//		return pow(b,-a)/boost::math::tgamma(a)*pow(x,1-a)*exp(-x/b);
		return exp(GetLogProb(x));
	}//end GetProb

	double GetLogProb(double x){
		if(x<1E-100){return -1000;}
		return -a*log(b)-LogGamma(a)+(a-1)*log(x)-x/b;
	}//end GetLogProb

	double LogGamma(double x){
		if(x<30){
			return log(boost::math::tgamma(x));
		}else{
			return x*log(x)-x+0.5*log(2*M_PI/x)+1./(12*x)-1./(360*pow(x,3.));
		}//endif
	}//end LogGamma

};//endclass GammaModel


class BetaPrimeModel{
public:
	double a;//a>0
	double b;//b>0
	double mean;
	double stdev;
	boost::mt19937 gen;
	boost::random::gamma_distribution<> dist1;
	boost::random::gamma_distribution<> dist2;

	BetaPrimeModel(double a_,double b_,int seed_=0){
		SetSeed(seed_);
		SetParam(a_,b_);
	}//end BetaPrimeModel
	~BetaPrimeModel(){
	}//end ~BetaPrimeModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void ABToMeanStdev(){
		mean=a/(b-1);
		stdev=pow(a*(a+b-1)/(b-2),0.5)/(b-1);
	}//end ABToMeanStdev

	void MeanStdevToAB(){
		a=mean+(mean+1)*pow(mean/stdev,2.);
		b=2+mean*(mean+1)/(stdev*stdev);
	}//end MeanStdevToAB

	void SetParam(double a_,double b_){
		a=a_;
		b=b_;
		ABToMeanStdev();
		dist1=boost::random::gamma_distribution<>(a,1);
		dist2=boost::random::gamma_distribution<>(b,1);
	}//end SetParam

	void SetParamFromMeanStdev(double mean_,double stdev_){
		mean=mean_;
		stdev=stdev_;
		MeanStdevToAB();
		dist1=boost::random::gamma_distribution<>(a,1);
		dist2=boost::random::gamma_distribution<>(b,1);
	}//end SetParamFromMeanStdev

	double Sample(){
		return dist1(gen)/dist2(gen);
	}//end Sample

};//endclass BetaPrimeModel



class BetaModel{
public:
	double a;
	double b;
	double mean;
	double stdev;
	boost::mt19937 gen;
	boost::random::beta_distribution<> dist;

	BetaModel(double a_,double b_,int seed_=0){
		SetSeed(seed_);
		SetParam(a_,b_);
	}//end BetaModel
	~BetaModel(){
	}//end ~BetaModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void SetParam(double a_,double b_){
		a=a_;
		b=b_;
		ABToMeanStdev();
		dist=boost::random::beta_distribution<>(a,b);
	}//end SetParam

	void SetParamFromMeanStdev(double mean_,double stdev_){
		mean=mean_;
		stdev=stdev_;
		MeanStdevToAB();
		dist=boost::random::beta_distribution<>(a,b);
	}//end SetParamFromMeanStdev

	void ABToMeanStdev(){
		mean=a/(a+b);
		stdev=pow(a*b/(a+b+1),0.5)/(a+b);
	}//end ABToMeanStdev

	void MeanStdevToAB(){
		double r=1./mean-1;
		a=1./(1+r)*( r*pow((1+r)*stdev,-2)-1  );
		b=r*a;
	}//end MeanStdevToAB

	double Sample(){
		return dist(gen);
	}//end Sample

	double GetProb(double x){
//		return pow(x,a-1)*pow(1-x,b-1)/boost::math::beta(a,b);
		return exp(GetLogProb(x));
	}//end GetProb

	double GetLogProb(double x){
		if(x<1E-100){return -1000;}
		return (a-1)*log(x)+(b-1)*log(1-x)-LogGamma(a)-LogGamma(b)+LogGamma(a+b);
	}//end GetLogProb

	double LogGamma(double x){
		if(x<30){
			return log(boost::math::tgamma(x));
		}else{
			return x*log(x)-x+0.5*log(2*M_PI/x)+1./(12*x)-1./(360*pow(x,3.));
		}//endif
	}//end LogGamma

	double Factor(){
		return exp(-LogGamma(a)-LogGamma(b)+LogGamma(a+b));//1/B(a,b)
	}//end Factor

};//endclass BetaModel


class CategoricalModel{
public:
	Prob<int> prob;
	boost::mt19937 gen;
	boost::random::discrete_distribution<> dist;

	CategoricalModel(Prob<int> &prob_,int seed_=3){
		SetSeed(seed_);
		SetProb(prob_);
	}//end CategoricalModel
	~CategoricalModel(){
	}//end ~CategoricalModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void SetProb(Prob<int> &prob_){
		prob=prob_;
		dist=boost::random::discrete_distribution<>(prob.P);
	}//end SetSeed

	int Sample(){
		return dist(gen);
	}//end Sample

};//endclass CategoricalModel

class DirichletModel{
public:
	int nDim;
	vector<double> dirParam;
	Prob<int> prob;
	boost::mt19937 gen;
	boost::random::discrete_distribution<> dist;
	vector<boost::gamma_distribution<> > gamma_dsts;

	DirichletModel(vector<double> &dirParam_,int seed_=0){
		SetSeed(seed_);
		SetDirParam(dirParam_);
	}//end DirichletModel
	DirichletModel(double concentration_,Prob<int> &prob_,int seed_=0){
		SetSeed(seed_);
		SetDirParamFromProb(concentration_,prob_);
	}//end DirichletModel
	~DirichletModel(){
	}//end ~DirichletModel

	void SetSeed(int seed_){
		gen.seed(seed_);
	}//end SetSeed

	void SetDirParam(vector<double> dirParam_){
		dirParam=dirParam_;
		nDim=dirParam.size();
		gamma_dsts.resize(nDim);
		for(int k=0;k<nDim;k+=1){
			gamma_dsts[k]=boost::gamma_distribution<>( dirParam[k], 1. );
		}//endfor k
//		dist=boost::random::discrete_distribution<>(prob.P);
	}//end SetDirParam

	void SetDirParamFromProb(double concentration_,Prob<int> prob_){
		vector<double> dirParam_;
		for(int k=0;k<prob_.P.size();k+=1){
			dirParam_.push_back(concentration_*prob_.P[k]);
		}//endfor k
		SetDirParam(dirParam_);
	}//end SetDirParamFromProb

	Prob<int> SampleProb(){
		Prob<int> prob;
		prob.P=dirParam;
		for(int k=0;k<nDim;k+=1){
			prob.P[k]=gamma_dsts[k](gen);
		}//endfor k
/*
		for(int i=0;i<nDim;i+=1){
			boost::gamma_distribution<> gamma_dst( prob.P[i], 1. );
			prob.P[i]=gamma_dst(gen);
		}//endfor i
*/
		prob.Normalize();
		return prob;
	}//end SampleProb


};//endclass DirichletModel


#endif // StatisticalDistributions_HPP
