#ifndef PianoFingeringDeterminationEngine_HPP
#define PianoFingeringDeterminationEngine_HPP

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
#include"PianoFingering_v170101_2.hpp"
#include"HandSeparationData_MergedOututHMM_v161230.hpp"
#include"KeyPos_v161230.hpp"
#include"FingeringHMMParameters_v161230.hpp"

using namespace std;

class PianoFingeringDeterminationEngine{
public:

	PianoFingering fingering;

	PianoFingeringDeterminationEngine(){}//end PianoFingeringDeterminationEngine
	~PianoFingeringDeterminationEngine(){}//end ~PianoFingeringDeterminationEngine

	void SeparateHands(){//chan= 0:Right Hand, 1:Left Hand

		HandSeparationData_MergedOutputHMM data;
		vector<vector<double> > Lprob(data.Lprob);
		vector<vector<double> > uniLprob(data.uniLprob);
		vector<double> LRLprob(data.LRLprob);
		vector<int> v(10);

		int length=fingering.evts.size();
		int dp_c=15;
		int handPartPreference[length][2];//HandPartPreference[m][0]=1 if m-th note is likely to be in the right-hand-part
		vector<int> pitch;
		for(int n=0;n<length;n+=1){
			PianoFingeringEvt evt=fingering.evts[n];
			handPartPreference[n][0]=0;
			handPartPreference[n][1]=0;
			int p_cur=SitchToPitch(evt.sitch);
			int p_max=p_cur;
			int p_min=p_cur;
			pitch.push_back(p_cur);
			for(int m=0;m<length;m+=1){
				if(fingering.evts[m].offtime < evt.ontime){continue;}
				if(fingering.evts[m].ontime  > evt.offtime){break;}
				int p=SitchToPitch(fingering.evts[m].sitch);
				if(p>p_max){p_max=p;}
				if(p<p_min){p_min=p;}
			}//endfor m
			if(p_cur>p_min+dp_c){handPartPreference[n][0]=1;}//likely to be in the right-hand-part
			if(p_cur<p_max-dp_c){handPartPreference[n][1]=1;}//likely to be in the left-hand-part
		}//endfor n

		int Nh=50;
		vector<double> LP;//k=2*h+sig
		LP.assign(2*Nh,-1000);
		vector<vector<int> > argmaxHist;
		LP[0]=uniLprob[0][pitch[0]];
		LP[1]=uniLprob[1][pitch[1]];
		for(int n=1;n<length;n+=1){
			double max,logP;
			vector<double> preLP(LP);
			vector<int> argmax(2*Nh);
			for(int i=0;i<2*Nh;i+=1){
				max=preLP[i]-10000;
				argmax[i]=i;
				for(int j=0;j<2*Nh;j+=1){
					if(j%2==i%2 && j/2==i/2-1){
						logP=preLP[j]+LRLprob[i%2]+Lprob[i%2][pitch[n]-pitch[n-1]+128];
						if(logP>max){max=logP; argmax[i]=j;}
					}//endif
					if(j%2!=i%2 && i/2==0){
						if(n-2-j/2>=0){
							logP=preLP[j]+LRLprob[i%2]+Lprob[i%2][pitch[n]-pitch[n-2-j/2]+128];
						}else{
							logP=preLP[j]+LRLprob[i%2]+Lprob[i%2][pitch[n]-((j%2==0)? 53:71)+128];
						}//endif
						if(logP>max){max=logP; argmax[i]=j;}
					}//endif
				}//endfor j
				if(i%2==0){
					v[1]=53;
					if(n-1-i/2>=0){v[1]=pitch[n-1-i/2];}
					LP[i]=max+((v[1]<pitch[n])? 0:-4.605)+((handPartPreference[n][0]>0)? -0.0202027:-0.693147)+((handPartPreference[n][1]>0)? -3.912023:-0.693147);
				}else{
					v[0]=71;
					if(n-1-i/2>=0){v[0]=pitch[n-1-i/2];}
					LP[i]=max+((v[0]>pitch[n])? 0:-4.605)+((handPartPreference[n][0]>0)? -3.912023:-0.693147)+((handPartPreference[n][1]>0)? -0.0202027:-0.693147);
				}//endif
			}//endfor i
			argmaxHist.push_back(argmax);
		}//endfor n

		vector<int> estStates(length);
		double max=LP[0];
		int amax=0;
		for(int i=0;i<LP.size();i+=1){if(LP[i]>max){max=LP[i]; amax=i;}}
		estStates[length-1]=amax;
		for(int n=0;n<length-1;n+=1){
			amax=argmaxHist[length-2-n][amax];
			estStates[length-2-n]=amax;
		}//endfor n
		for(int n=0;n<length;n+=1){
			fingering.evts[n].channel=estStates[n]%2;
			fingering.evts[n].fingerNum=((estStates[n]%2==0)? "1":"-1");
		}//endfor n

	}//end SeparateHands


	void DetermineFingering_OneHand(int hand){//hand= 0:Right, 1:Left
		if(hand!=0 && hand!=1){
			cout<<"Error in DetermineFingering_OneHand: Hand must be 0:Right or 1:Left. Selected : "<<hand<<endl;
			return;
		}//endif

		vector<int> pitchSeq;
		vector<int> originalPos;

		for(int n=0;n<fingering.evts.size();n+=1){
			if(fingering.evts[n].channel!=hand){continue;}
			pitchSeq.push_back(SitchToPitch(fingering.evts[n].sitch));
			originalPos.push_back(n);
		}//endfor n

		if(hand==1){//if left hand, convert pitches (reflection symmetry)
			for(int n=0;n<pitchSeq.size();n+=1){
				pitchSeq[n]=(62-pitchSeq[n])+62;
			}//endfor n
		}//endif

		FingeringHMMParameters parameters;
		vector<vector<double> > trLP(parameters.trLP);
		vector<vector<vector<vector<double> > > > outLP(parameters.outLP);

		vector<double> LP;
		vector<vector<int> > amaxHist;
		LP.assign(6,0);//Remark: LP[0] is not used.
		LP[0]=-1000;
{//Viterbi
		vector<int> amax(6);
		vector<double> prevLP;
		for(int m=1;m<pitchSeq.size();m+=1){
			KeyPos keyInt=SubtrKeyPos(PitchToKeyPos(pitchSeq[m]),PitchToKeyPos(pitchSeq[m-1]));
			prevLP=LP;
			for(int j=1;j<=5;j+=1){
				LP[j]=prevLP[1]+trLP[1][j]+outLP[1][j][keyInt.x+50][keyInt.y+1];
				double logP;
				amax[j]=1;
				for(int i=1;i<=5;i+=1){
					logP=prevLP[i]+trLP[i][j]+outLP[i][j][keyInt.x+50][keyInt.y+1];
					if(logP>LP[j]){LP[j]=logP; amax[j]=i;}
				}//endfor i
			}//endfor j
			amaxHist.push_back(amax);
		}//endfor m
}//

		vector<int> maxSeq;
{//Backtrace
		int size1=amaxHist.size();
		double max=LP[1];
		int amax=1;
		for(int i=1;i<=5;i+=1){
			if(LP[i]>max){max=LP[i]; amax=i;}
		}//endfor i
		maxSeq.push_back(amax);
		for(int m=1;m<size1+1;m+=1){
			maxSeq.push_back(amaxHist[size1-m][maxSeq[maxSeq.size()-1]]);
		}//endfor m
		assert(originalPos.size()==maxSeq.size());
		int size2=maxSeq.size();
		stringstream ss;
		for(int n=0;n<size2;n+=1){
			ss.str("");
			ss<<((hand==0)? "":"-")<<maxSeq[size2-1-n];
			fingering.evts[originalPos[n]].fingerNum=ss.str();
//cout<<time[m]<<"\t"<<pitchSeq[m]<<"\t"<<1<<"\t"<<maxSeq[size2-1-m]<<endl;
		}//endfor n
}//

	}//end DetermineFingering_OneHand


	void DetermineFingering_BothHands(){
		SeparateHands();
		DetermineFingering_OneHand(0);
		DetermineFingering_OneHand(1);
	}//end DetermineFingering_BothHands


};//end class PianoFingeringDeterminationEngine

#endif // PianoFingeringDeterminationEngine_HPP

