#ifndef NVR_MRF_HPP
#define NVR_MRF_HPP

#define printOn false

#include<fstream>
#include<iostream>
#include<cmath>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include"stdio.h"
#include"stdlib.h"
#include "Trx_v170203.hpp"
#include "ContextTree_v170130.hpp"
#include "PianoRoll_v170503.hpp"
#include "QuantizedPianoRoll_v190822.hpp"

using namespace std;


class NVR_MRF{
public:

	int TPQN;
	double a1,b1,p1,a2,b2,p2,lnw1,lnw2,lnz1,lnz2;
	double a3,b3,p3,lnz3;

	vector<vector<vector<double> > > InterDepIONVLP;//For each nbhRange=0,...,15

	double beta1,beta2,beta31,beta32;
	int rangeNBH;//pitch interval to define neighbourhood

	ContextTree tree;

	Trx trx;
	QuantizedPianoRoll qpr;
	PianoRoll pr;

	NVR_MRF(string InterDepIONVProbFile){
		Init(InterDepIONVProbFile);
	}//end NVR_MRF
	~NVR_MRF(){
	}//end ~NVR_MRF

	void Init(string InterDepIONVProbFile){

		beta1=0.965;
		beta2=0.03;
		beta31=0.5;
		beta32=0.005;

		rangeNBH=12;

		TPQN=24;

		a1=2.24;
		b1=0.24;
		p1=0.69;
		lnz1=1.44932;
		a2=13.8;
		b2=15.2;
		p2=-1.22;
		lnz2=29.7682;

		lnw1=log(0.814);
		lnw2=log(0.152519);

		a3=0.94;
		b3=0.51;
		p3=0.80;
		lnz3=0.76477;

		InterDepIONVLP.resize(16);
		for(int i=0;i<InterDepIONVLP.size();i+=1){
			InterDepIONVLP[i].resize(11);
			for(int j=0;j<11;j+=1){
				InterDepIONVLP[i][j].assign(11,0);
			}//endfor j
		}//endfor i
{
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		ifstream ifs(InterDepIONVProbFile.c_str());//"InterDepIONVProb.txt"
		for(int i=0;i<16;i+=1){
			getline(ifs,s[99]);
			for(int j=0;j<11;j+=1){
				for(int j_=0;j_<11;j_+=1){
					ifs>>InterDepIONVLP[i][j][j_];
					InterDepIONVLP[i][j][j_]=log(InterDepIONVLP[i][j][j_]);
				}//endfor j_
				getline(ifs,s[99]);
			}//endfor j
		}//endfor i
		ifs.close();
}//

	}//end Init

	void SetParameters(double beta1_,double beta2_,double beta31_,double beta32_,int rangeNBH_){
		beta1=beta1_;
		beta2=beta2_;
		beta31=beta31_;
		beta32=beta32_;
		rangeNBH=rangeNBH_;
	}//end SetParameters

	void SetTrData(Trx trx_){
		trx=trx_;
	}//end SetTrData

	void ConstructTrData(QuantizedPianoRoll qpr_,PianoRoll pr_){
		assert(qpr.evts.size()==pr.evts.size());
		qpr=qpr_;
		pr=pr_;
		trx.Clear();
		TPQN=qpr.TPQN;
		trx.TPQN=qpr.TPQN;
		for(int n=0;n<qpr.evts.size();n+=1){
			TrxEvt trxEvt(pr.evts[n]);
			trxEvt.onstime=qpr.evts[n].onstime;
			trxEvt.offstime=qpr.evts[n].offstime;
			trxEvt.voice=0;
			trxEvt.secPerQN=qpr.spqnEvts[0].value;
			for(int i=0;i<qpr.spqnEvts.size();i+=1){
				if(trxEvt.onstime<qpr.spqnEvts[i].stime){break;}
				trxEvt.secPerQN=qpr.spqnEvts[i].value;
			}//endfor i
			trx.evts.push_back(trxEvt);
		}//endfor n
	}//end ConstructTrData

	void SetContextTree(ContextTree tree_){
		tree=tree_;
	}//end SetTrData

	double OutputLogProb(double x){
		assert(x>0);
		return LogAdd(lnw1+lnz1+(p1-1)*log(x)-(a1*x+b1/x),lnw2+lnz2+(p2-1)*log(x)-(a2*x+b2/x));
	}//end OutputLogProb

	double OutputPedalLogProb(double x){
		assert(x>0);
		return lnz3+(p3-1)*log(x)-(a3*x+b3/x);
	}//end OutputPedalLogProb

	void EstimateNVs(){

		vector<vector<int> > clusterIDs;
{
		vector<int> cluster;
		cluster.push_back(0);
		for(int n=1;n<trx.evts.size();n+=1){
			if(trx.evts[n].onstime!=trx.evts[n-1].onstime){
				clusterIDs.push_back(cluster);
				cluster.clear();
			}//endif
			cluster.push_back(n);
		}//endfor n
		clusterIDs.push_back(cluster);
}//

		/// Set contexts
		vector<vector<int> > contexts(trx.evts.size());
{
		int curPitch;
		vector<int> context;
		for(int nn=0;nn<clusterIDs.size();nn+=1){
			for(int n=0;n<clusterIDs[nn].size();n+=1){
				curPitch=SitchToPitch(trx.evts[clusterIDs[nn][n]].sitch);
				context.assign(10,0);

				for(int k=0;k<10;k+=1){
					int nn_=nn+k+1;
					if(nn_>=clusterIDs.size()){continue;}
					int minInt=88;
					for(int i_=0;i_<clusterIDs[nn_].size();i_+=1){
						int intvl=SitchToPitch(trx.evts[clusterIDs[nn_][i_]].sitch)-curPitch;
						if(intvl<0){intvl*=-1;}
						if(intvl<minInt){
							minInt=intvl;
						}//endif
					}//endfor i_
					if(minInt==88){minInt=0;}
					context[k]=minInt;
				}//endfor k

				contexts[clusterIDs[nn][n]]=context;

			}//endfor n
		}//endfor nn
}//

		for(int nn=0;nn<clusterIDs.size()-1;nn+=1){
			vector<int> genIONV;
			for(int i=nn+1;i<=nn+10;i+=1){
				if(i>=clusterIDs.size()){continue;}
				if(clusterIDs[nn].size()==7 && i>nn+7){continue;}
				if(clusterIDs[nn].size()==8 && i>nn+6){continue;}
				if(clusterIDs[nn].size()==9 && i>nn+5){continue;}
				if(clusterIDs[nn].size()==10 && i>nn+4){continue;}
				if(clusterIDs[nn].size()>=11 && i>nn+2){continue;}
				if(clusterIDs[nn].size()>=13 && i>nn+1){continue;}
				genIONV.push_back( trx.evts[clusterIDs[i][0]].onstime-trx.evts[clusterIDs[nn][0]].onstime );
			}//endfor i

			vector<vector<int> > genIONVIDs;//Solution space clusterIDs[nn].size() x genIONV.size()
			int size=1;
			for(int k=0;k<clusterIDs[nn].size();k+=1){
				size*=genIONV.size();
			}//endfor k
{
			vector<int> vi(clusterIDs[nn].size());
			int i_;
			for(int i=0;i<size;i+=1){
				i_=i;
				for(int k=0;k<clusterIDs[nn].size();k+=1){
					vi[k]=i_%genIONV.size();
					i_=i_/genIONV.size();
				}//endfor k
				genIONVIDs.push_back(vi);
			}//endfor i
}//

			vector<vector<bool> > nearPitchPair;//nearPitchPair[i][j]=true if sitches of i and j are near
			vector<double> dur(clusterIDs[nn].size());
			vector<double> durPedal(clusterIDs[nn].size());

			for(int k=0;k<clusterIDs[nn].size();k+=1){
				dur[k]=trx.evts[clusterIDs[nn][k]].offtime-trx.evts[clusterIDs[nn][k]].ontime;
				durPedal[k]=trx.evts[clusterIDs[nn][k]].endtime-trx.evts[clusterIDs[nn][k]].ontime;
			}//endfor k
			double secPerTick=trx.evts[clusterIDs[nn][0]].secPerQN/double(TPQN);

			nearPitchPair.resize(clusterIDs[nn].size());
			for(int k=0;k<clusterIDs[nn].size();k+=1){
				nearPitchPair[k].assign(clusterIDs[nn].size(),false);
				for(int l=0;l<clusterIDs[nn].size();l+=1){
					if(SitchToPitch(trx.evts[clusterIDs[nn][l]].sitch)>=SitchToPitch(trx.evts[clusterIDs[nn][k]].sitch)-rangeNBH
					   && SitchToPitch(trx.evts[clusterIDs[nn][l]].sitch)<=SitchToPitch(trx.evts[clusterIDs[nn][k]].sitch)+rangeNBH){
						nearPitchPair[k][l]=true;
					}//endif
				}//endfor l
			}//endfor k

			double max;
			double logP;
			int amax;
			for(int i=0;i<size;i+=1){
				logP=0;
				for(int k=0;k<clusterIDs[nn].size();k+=1){

					/// H1
					logP+=beta1*tree.nodes[tree.FindLeafID(contexts[clusterIDs[nn][k]])].prob.LP[genIONVIDs[i][k]+1];

					/// H2
					for(int l=k+1;l<clusterIDs[nn].size();l+=1){
						if(nearPitchPair[k][l]){
							logP+=2*beta2*InterDepIONVLP[rangeNBH][genIONVIDs[i][k]+1][genIONVIDs[i][l]+1];
						}//endif
					}//endfor l

					/// H3
					logP+=beta31*OutputLogProb(dur[k]/(double(genIONV[genIONVIDs[i][k]])*secPerTick));
					logP+=beta32*OutputPedalLogProb(durPedal[k]/(double(genIONV[genIONVIDs[i][k]])*secPerTick));
				}//endfor k
				if(i==0 || logP>max){
					max=logP;
					amax=i;
				}//endif
			}//endfor i

			for(int k=0;k<clusterIDs[nn].size();k+=1){

				trx.evts[clusterIDs[nn][k]].offstime=trx.evts[clusterIDs[nn][k]].onstime+genIONV[genIONVIDs[amax][k]];
			}//endfor k

		}//endfor nn

		for(int k=0;k<clusterIDs[clusterIDs.size()-1].size();k+=1){
			trx.evts[clusterIDs[clusterIDs.size()-1][k]].offstime=trx.evts[clusterIDs[clusterIDs.size()-1][k]].onstime+TPQN;
		}//endfor k

	}//end EstimateNVs

	void WriteFile(string filename){
		trx.WriteFile(filename);
	}//end WriteFile

	void WriteQprFile(string filename){
		assert(trx.evts.size()==qpr.evts.size());
		for(int n=0;n<qpr.evts.size();n+=1){
			qpr.evts[n].offstime=trx.evts[n].offstime;
		}//endfor n
		qpr.WriteFile(filename,0);
	}//end WriteQprFile

};//endclass NVR_MRF



#endif // NVR_MRF_HPP
