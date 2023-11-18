#ifndef QprAnalyzer_HPP
#define QprAnalyzer_HPP

#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cfloat>
#include<cmath>
#include<cassert>
#include<algorithm>
#include"BasicCalculation_v170122.hpp"
#include"QuantizedPianoRoll_v191003.hpp"
#include"PianoRoll_v170503.hpp"
#include"PianoFingeringDeterminationEngine_v170101_2.hpp"

using namespace std;

class PitchRank{
public:
	int pitch;
	int rank;

	PitchRank(){
	}//end PitchRank
	PitchRank(int pitch_){
		pitch=pitch_;
		rank=-1;
	}//end PitchRank
	~PitchRank(){
	}//end ~PitchRank

};//endclass PitchRank

void SetRank(vector<PitchRank> &pitchRankSet){
	vector<Pair> pairs;
	for(int i=0;i<pitchRankSet.size();i+=1){
		pairs.push_back(Pair(i,pitchRankSet[i].pitch));
	}//endfor i
	sort(pairs.begin(), pairs.end(), LessPair());
	for(int i=0;i<pairs.size();i+=1){
		pitchRankSet[pairs[i].ID].rank=i;
	}//endfor i
}//end SetRank

double BPSetDist(vector<vector<int> > &bpSet,vector<vector<int> > &bpSet_){
	if(bpSet.size()==0 || bpSet_.size()==0){return 0;}
	double Dp=0;
	for(int l=0;l<bpSet.size();l+=1){
		for(int l_=0;l_<bpSet_.size();l_+=1){
			if(bpSet[l]==bpSet_[l_]){Dp+=1;}
		}//endfor l_
	}//endfor l
	Dp/=(0.5*double(bpSet.size()+bpSet_.size()));
	return Dp;
}//end BPSetDist

double SSMContrast(double x){//0<=x<=1
	return pow(x-0.5,2.)-0.025;
//	if(x==0){return 0;}
//	return pow(x,4);
//	return pow(x-0.5,6);
// 	return pow(x-0.5,4);
}//end SSMContrast


class QprAnalyzer{
public:
	QuantizedPianoRoll qpr,qprRH,qprLH;
	vector<vector<double> > SSM,SSMRH,SSMLH;

	int nBeat;//for metrical analysis
	Prob<int> uniMetProb,uniMetRHProb,uniMetLHProb;//nBeat
	vector<Prob<int> > trMetProb,trMetRHProb,trMetLHProb;//nBeat x nBeat
	vector<Prob<int> > chordProb,chordRHProb,chordLHProb;//nBeat x 2 (0,1=transit,stay)
	vector<Prob<int> > nvProb,nvRHProb,nvLHProb;//nBeat x nBeat (r=0 <-> nv=1, nv>=nBeat <-> r=nBeat-1
	Prob<int> rpcUniProb;//12
	vector<Prob<int> > rpcProb,rpcRHProb,rpcLHProb;//nBeat x 12
	int pitchRankWidth;
//	vector<Prob<int> > pitchRankProb;//2(0:offBeat/1:onBeat) x pitchRankWidth
	vector<Prob<int> > pitchRankProb;//nBeat x pitchRankWidth

	double metLP,metLPRH,metLPLH;
	double nvLP,nvLPRH,nvLPLH;
	double tieRate,tieRateRH,tieRateLH;
	double rpcLP,rpcLPRH,rpcLPLH;
	double ssmContrast,ssmContrastRH,ssmContrastLH;
	double pitchRankLP;

	QprAnalyzer(){
		nBeat=48;
		pitchRankWidth=10;
	}//end QprAnalyzer
	QprAnalyzer(string name){
		ReadFile(name);
	}//end QprAnalyzer
	~QprAnalyzer(){
	}//end ~QprAnalyzer

	void ReadFile(string name){
		qpr.ReadFile(name,0);
		qpr.Sort();
		qpr.ChangeTPQN(12);
		RemoveRests();
		qpr.SplitBars(48);
		SeparateHands();
	}//end ReadFile

	void LoadQpr(QuantizedPianoRoll qpr_,int beatShift){
		qpr=qpr_;
		qpr.Sort();
		qpr.ChangeTPQN(12);
		if(beatShift!=0){
			for(int l=0;l<qpr.evts.size();l+=1){
				qpr.evts[l].onstime+=beatShift*6;
				qpr.evts[l].offstime+=beatShift*6;
			}//endfor l
			qpr.meterEvts[qpr.meterEvts.size()-1].stime+=beatShift*6;
		}//endif
		RemoveRests();
		qpr.SplitBars(48);
		SeparateHands();
	}//end LoadQpr

	void LoadQprScale(QuantizedPianoRoll qpr_,int rescale){//rescale=0,-1,1
		qpr=qpr_;
		qpr.Sort();
		if(rescale==1){//Make tempo double
			for(int l=0;l<qpr.evts.size();l+=1){
				qpr.evts[l].onstime*=2;
				qpr.evts[l].offstime*=2;
			}//endfor l
			for(int i=0;i<qpr.meterEvts.size();i+=1){
				qpr.meterEvts[i].stime*=2;
			}//endfor i
			for(int i=0;i<qpr.spqnEvts.size();i+=1){
				qpr.spqnEvts[i].stime*=2;
				qpr.spqnEvts[i].value/=2.;
				qpr.spqnEvts[i].bpm*=2;
			}//endfor i
		}else if(rescale==-1){//Make tempo half
			for(int l=0;l<qpr.evts.size();l+=1){
				qpr.evts[l].onstime/=2;
				qpr.evts[l].offstime/=2;
			}//endfor l
			for(int i=0;i<qpr.meterEvts.size();i+=1){
				qpr.meterEvts[i].stime/=2;
			}//endfor i
			for(int i=0;i<qpr.spqnEvts.size();i+=1){
				qpr.spqnEvts[i].stime/=2;
				qpr.spqnEvts[i].value*=2.;
				qpr.spqnEvts[i].bpm/=2;
			}//endfor i
		}//endif
		qpr.ChangeTPQN(12);
		RemoveRests();
		qpr.SplitBars(48);
		SeparateHands();
	}//end LoadQprScale

	void AssignZeros(){
		uniMetProb.Assign(nBeat,1E-1);
		uniMetRHProb.Assign(nBeat,1E-1);
		uniMetLHProb.Assign(nBeat,1E-1);
		trMetProb.clear(); trMetRHProb.clear(); trMetLHProb.clear();
		chordProb.clear(); chordRHProb.clear(); chordLHProb.clear();
		nvProb.clear(); nvRHProb.clear(); nvLHProb.clear();
		pitchRankProb.clear();
		trMetProb.resize(nBeat); trMetRHProb.resize(nBeat); trMetLHProb.resize(nBeat);
		chordProb.resize(nBeat); chordRHProb.resize(nBeat); chordLHProb.resize(nBeat);
		nvProb.resize(nBeat); nvRHProb.resize(nBeat); nvLHProb.resize(nBeat);
		pitchRankProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			trMetProb[b].Assign(nBeat,1E-1);
			trMetRHProb[b].Assign(nBeat,1E-1);
			trMetLHProb[b].Assign(nBeat,1E-1);
			chordProb[b].Assign(2,1E-1);
			chordRHProb[b].Assign(2,1E-1);
			chordLHProb[b].Assign(2,1E-1);
			nvProb[b].Assign(nBeat,1E-1);
			nvRHProb[b].Assign(nBeat,1E-1);
			nvLHProb[b].Assign(nBeat,1E-1);
			pitchRankProb[b].Assign(pitchRankWidth,1E-1);
		}//endfor b

		rpcUniProb.Assign(12,1E-1);
		rpcProb.clear(); rpcRHProb.clear(); rpcLHProb.clear();
		rpcProb.resize(nBeat); rpcRHProb.resize(nBeat); rpcLHProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			rpcProb[b].Assign(12,1E-1);
			rpcRHProb[b].Assign(12,1E-1);
			rpcLHProb[b].Assign(12,1E-1);
		}//endfor b
	}//end AssignZeros


	void ReadParamFile(string filename){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		ifstream ifs(filename.c_str());

		ifs>>s[1]>>nBeat;
		getline(ifs,s[99]);
		ifs>>s[1]>>pitchRankWidth;
		getline(ifs,s[99]);

		AssignZeros();

		getline(ifs,s[99]);//### uniMetProb
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniMetProb.P[b];
		}//endfor b
		uniMetProb.Normalize();
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### uniMetRHProb
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniMetRHProb.P[b];
		}//endfor b
		uniMetRHProb.Normalize();
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### uniMetLHProb
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniMetLHProb.P[b];
		}//endfor b
		uniMetLHProb.Normalize();
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### trMetProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>trMetProb[b].P[bp];
			}//endfor bp
			trMetProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### trMetRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>trMetRHProb[b].P[bp];
			}//endfor bp
			trMetRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### trMetLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>trMetLHProb[b].P[bp];
			}//endfor bp
			trMetLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### chordProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ifs>>chordProb[b].P[bp];
			}//endfor bp
			chordProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### chordRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ifs>>chordRHProb[b].P[bp];
			}//endfor bp
			chordRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### chordLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ifs>>chordLHProb[b].P[bp];
			}//endfor bp
			chordLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### nvProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>nvProb[b].P[bp];
			}//endfor bp
			nvProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### nvRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>nvRHProb[b].P[bp];
			}//endfor bp
			nvRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### nvLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>nvLHProb[b].P[bp];
			}//endfor bp
			nvLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);


		getline(ifs,s[99]);//### rpcUniProb
		for(int b=0;b<12;b+=1){
			ifs>>rpcUniProb.P[b];
		}//endfor b
		rpcUniProb.Normalize();
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### rpcProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<12;bp+=1){
				ifs>>rpcProb[b].P[bp];
			}//endfor bp
			rpcProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### rpcRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<12;bp+=1){
				ifs>>rpcRHProb[b].P[bp];
			}//endfor bp
			rpcRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### rpcLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<12;bp+=1){
				ifs>>rpcLHProb[b].P[bp];
			}//endfor bp
			rpcLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### pitchRankProb
		for(int b=0;b<nBeat;b+=1){
			for(int k=0;k<pitchRankWidth;k+=1){
				ifs>>pitchRankProb[b].P[k];
			}//endfor k
			pitchRankProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		ifs.close();
	}//end ReadParamFile

// 	void ReadPitchRankProbParam(string filename){
// 		pitchRankProb.clear();
// 		pitchRankProb.resize(2);
// 		pitchRankProb[0].Assign(pitchRankWidth,1);
// 		pitchRankProb[1].Assign(pitchRankWidth,1);
// 		vector<int> v(100);
// 		vector<double> d(100);
// 		vector<string> s(100);
// 		stringstream ss;
// 		ifstream ifs(filename.c_str());
// 		getline(ifs,s[99]);
// 		for(int i=0;i<pitchRankWidth;i+=1){
// 			ifs>>v[0]>>pitchRankProb[0].P[i]>>pitchRankProb[1].P[i];
// 		}//endfor i
// 		ifs.close();
// 		pitchRankProb[0].Normalize();
// 		pitchRankProb[1].Normalize();
// 	}//end ReadPitchRankProbParam

	void RemoveRests(){
		for(int l=qpr.evts.size()-1;l>=0;l-=1){
			if(qpr.evts[l].pitch<0){qpr.evts.erase(qpr.evts.begin()+l);}
		}//endfor l
	}//end RemoveRests

	void SeparateHands(){

		PianoRoll pr;
		pr=qpr.ToPianoRoll(0);
		PianoFingering fingering(pr);

		PianoFingeringDeterminationEngine engine;
		engine.fingering=fingering;
		engine.DetermineFingering_BothHands();
		fingering=engine.fingering;

		//Correct results
		for(int n=0;n<fingering.evts.size();n+=1){
			fingering.evts[n].ext1=n;
		}//endfor n

		vector<vector<PianoFingeringEvt> > onsetClusters;
{
		vector<PianoFingeringEvt> cluster;
		for(int n=0;n<fingering.evts.size();n+=1){
			if(n==0){cluster.push_back(fingering.evts[n]);continue;}
			if(fingering.evts[n].ontime-fingering.evts[n-1].ontime>0.05){
				onsetClusters.push_back(cluster);
				cluster.clear();
			}//endif
			cluster.push_back(fingering.evts[n]);
		}//endfor n
		onsetClusters.push_back(cluster);
}//

		for(int i=0;i<onsetClusters.size();i+=1){
			sort(onsetClusters[i].begin(),onsetClusters[i].end(),LessPitchPianoFingeringEvt());
			if(onsetClusters[i].size()<=2){continue;}
			if(onsetClusters[i][onsetClusters[i].size()-1].pitch-onsetClusters[i][0].pitch<14){continue;}
			for(int j=0;j<onsetClusters[i].size();j+=1){
				if(onsetClusters[i][onsetClusters[i].size()-1].pitch-onsetClusters[i][j].pitch>14){onsetClusters[i][j].channel=1;}
				if(onsetClusters[i][j].pitch-onsetClusters[i][0].pitch>14){onsetClusters[i][j].channel=0;}
			}//endfor j
			for(int j=0;j<onsetClusters[i].size();j+=1){
				fingering.evts[onsetClusters[i][j].ext1].channel=onsetClusters[i][j].channel;
			}//endfor j
		}//endfor i

		assert(qpr.evts.size()==fingering.evts.size());

		qprLH=qpr;
		qprRH=qpr;
		qprRH.evts.clear();
		qprLH.evts.clear();
		for(int l=0;l<qpr.evts.size();l+=1){
			if(fingering.evts[l].channel==0){//RH
				qprRH.evts.push_back(qpr.evts[l]);
			}else{//LH
				qprLH.evts.push_back(qpr.evts[l]);
			}//endif
		}//endfor n

		qprRH.SplitBars(48);
		qprLH.SplitBars(48);

		assert(qprRH.bars.size()==qpr.bars.size());
		assert(qprLH.bars.size()==qpr.bars.size());

	}//end SeparateHands

	void SetTieRate(){
		double numNote=0;
		double numTie=0;
		for(int m=0;m<qpr.bars.size();m+=1){
			for(int l=0;l<qpr.bars[m].notes.size();l+=1){
				numNote+=1;
				if(qpr.bars[m].notes[l].offstime>qpr.bars[m].info.stime+qpr.bars[m].info.barlen){numTie+=1;}
			}//endfor l
		}//endfor m
		tieRate=-numTie/numNote;

		numNote=0;
		numTie=0;
		for(int m=0;m<qprRH.bars.size();m+=1){
			for(int l=0;l<qprRH.bars[m].notes.size();l+=1){
				numNote+=1;
				if(qprRH.bars[m].notes[l].offstime>qprRH.bars[m].info.stime+qprRH.bars[m].info.barlen){numTie+=1;}
			}//endfor l
		}//endfor m
		tieRateRH=-numTie/numNote;

		numNote=0;
		numTie=0;
		for(int m=0;m<qprLH.bars.size();m+=1){
			for(int l=0;l<qprLH.bars[m].notes.size();l+=1){
				numNote+=1;
				if(qprLH.bars[m].notes[l].offstime>qprLH.bars[m].info.stime+qprLH.bars[m].info.barlen){numTie+=1;}
			}//endfor l
		}//endfor m
		tieRateLH=-numTie/numNote;

	}//end SetTieRate

	void SetSSM(){
{
		vector<vector<vector<int> > > bpSetList;//[m][][0,1]=b,p, no double count
		vector<vector<vector<int> > > brSetList;//[m][][0,1]=b,r, no double count
		bpSetList.resize(qpr.bars.size());
		brSetList.resize(qpr.bars.size());
		vector<int> bp(2),br(2);

		for(int m=0;m<qpr.bars.size();m+=1){
			for(int l=0;l<qpr.bars[m].notes.size();l+=1){
				br[0]=qpr.bars[m].notes[l].onMetpos;
				br[1]=qpr.bars[m].notes[l].offstime-qpr.bars[m].notes[l].onstime;
				if(find(brSetList[m].begin(),brSetList[m].end(),br)==brSetList[m].end()){brSetList[m].push_back(br);}
			}//endfor l
		}//endfor m

		QuantizedPianoRoll qprSplit=qpr;
		qprSplit.SplitNotesAcrossBars();
		assert(qprSplit.bars.size()==qpr.bars.size());
		for(int m=0;m<qprSplit.bars.size();m+=1){
			for(int l=0;l<qprSplit.bars[m].notes.size();l+=1){
				int nv=qprSplit.bars[m].notes[l].offstime-qprSplit.bars[m].notes[l].onstime;
				for(int b=qprSplit.bars[m].notes[l].onMetpos;b<qprSplit.bars[m].notes[l].onMetpos+nv;b+=1){
					if(b>=qprSplit.bars[m].info.barlen){continue;}
					bp[0]=b;
					bp[1]=qprSplit.bars[m].notes[l].pitch;
					if(find(bpSetList[m].begin(),bpSetList[m].end(),bp)==bpSetList[m].end()){bpSetList[m].push_back(bp);}
				}//endfor b
			}//endfor l
		}//endfor m

		SSM.clear();
		SSM.resize(qpr.bars.size());
		for(int m=0;m<qpr.bars.size();m+=1){
			for(int m_=0;m_<qpr.bars.size();m_+=1){
				SSM[m].push_back( 0.5*(BPSetDist(bpSetList[m],bpSetList[m_])+BPSetDist(brSetList[m],brSetList[m_])) );
			}//endfor i_
		}//endfor i
}{
		vector<vector<vector<int> > > bpSetList;//[m][][0,1]=b,p, no double count
		vector<vector<vector<int> > > brSetList;//[m][][0,1]=b,r, no double count
		bpSetList.resize(qprRH.bars.size());
		brSetList.resize(qprRH.bars.size());
		vector<int> bp(2),br(2);

		for(int m=0;m<qprRH.bars.size();m+=1){
			for(int l=0;l<qprRH.bars[m].notes.size();l+=1){
				br[0]=qprRH.bars[m].notes[l].onMetpos;
				br[1]=qprRH.bars[m].notes[l].offstime-qprRH.bars[m].notes[l].onstime;
				if(find(brSetList[m].begin(),brSetList[m].end(),br)==brSetList[m].end()){brSetList[m].push_back(br);}
			}//endfor l
		}//endfor m

		QuantizedPianoRoll qprSplit=qprRH;
		qprSplit.SplitNotesAcrossBars();
		assert(qprSplit.bars.size()==qprRH.bars.size());
		for(int m=0;m<qprSplit.bars.size();m+=1){
			for(int l=0;l<qprSplit.bars[m].notes.size();l+=1){
				int nv=qprSplit.bars[m].notes[l].offstime-qprSplit.bars[m].notes[l].onstime;
				for(int b=qprSplit.bars[m].notes[l].onMetpos;b<qprSplit.bars[m].notes[l].onMetpos+nv;b+=1){
					if(b>=qprSplit.bars[m].info.barlen){continue;}
					bp[0]=b;
					bp[1]=qprSplit.bars[m].notes[l].pitch;
					if(find(bpSetList[m].begin(),bpSetList[m].end(),bp)==bpSetList[m].end()){bpSetList[m].push_back(bp);}
				}//endfor b
			}//endfor l
		}//endfor m

		SSMRH.clear();
		SSMRH.resize(qprRH.bars.size());
		for(int m=0;m<qprRH.bars.size();m+=1){
			for(int m_=0;m_<qprRH.bars.size();m_+=1){
				SSMRH[m].push_back( 0.5*(BPSetDist(bpSetList[m],bpSetList[m_])+BPSetDist(brSetList[m],brSetList[m_])) );
			}//endfor i_
		}//endfor i
}{
		vector<vector<vector<int> > > bpSetList;//[m][][0,1]=b,p, no double count
		vector<vector<vector<int> > > brSetList;//[m][][0,1]=b,r, no double count
		bpSetList.resize(qprLH.bars.size());
		brSetList.resize(qprLH.bars.size());
		vector<int> bp(2),br(2);

		for(int m=0;m<qprLH.bars.size();m+=1){
			for(int l=0;l<qprLH.bars[m].notes.size();l+=1){
				br[0]=qprLH.bars[m].notes[l].onMetpos;
				br[1]=qprLH.bars[m].notes[l].offstime-qprLH.bars[m].notes[l].onstime;
				if(find(brSetList[m].begin(),brSetList[m].end(),br)==brSetList[m].end()){brSetList[m].push_back(br);}
			}//endfor l
		}//endfor m

		QuantizedPianoRoll qprSplit=qprLH;
		qprSplit.SplitNotesAcrossBars();
		assert(qprSplit.bars.size()==qprLH.bars.size());
		for(int m=0;m<qprSplit.bars.size();m+=1){
			for(int l=0;l<qprSplit.bars[m].notes.size();l+=1){
				int nv=qprSplit.bars[m].notes[l].offstime-qprSplit.bars[m].notes[l].onstime;
				for(int b=qprSplit.bars[m].notes[l].onMetpos;b<qprSplit.bars[m].notes[l].onMetpos+nv;b+=1){
					if(b>=qprSplit.bars[m].info.barlen){continue;}
					bp[0]=b;
					bp[1]=qprSplit.bars[m].notes[l].pitch;
					if(find(bpSetList[m].begin(),bpSetList[m].end(),bp)==bpSetList[m].end()){bpSetList[m].push_back(bp);}
				}//endfor b
			}//endfor l
		}//endfor m

		SSMLH.clear();
		SSMLH.resize(qprLH.bars.size());
		for(int m=0;m<qprLH.bars.size();m+=1){
			for(int m_=0;m_<qprLH.bars.size();m_+=1){
				SSMLH[m].push_back( 0.5*(BPSetDist(bpSetList[m],bpSetList[m_])+BPSetDist(brSetList[m],brSetList[m_])) );
			}//endfor i_
		}//endfor i
}//
	}//end SetSSM

	void SetSSMContrast(){
		ssmContrast=0;
		double numElem=0;
		for(int m=0;m<SSM.size()-1;m+=1){
				ssmContrast+=SSMContrast(SSM[m][m+1]);
				numElem+=1;
				if(m<SSM.size()-2){
					ssmContrast+=SSMContrast(SSM[m][m+2]);
					numElem+=1;
				}//endif
		}//endfor m
		ssmContrast/=numElem;

		ssmContrastRH=0;
		numElem=0;
		for(int m=0;m<SSMRH.size()-1;m+=1){
				ssmContrastRH+=SSMContrast(SSMRH[m][m+1]);
				numElem+=1;
				if(m<SSMRH.size()-2){
					ssmContrastRH+=SSMContrast(SSMRH[m][m+2]);
					numElem+=1;
				}//endif
		}//endfor m
		ssmContrastRH/=numElem;

		ssmContrastLH=0;
		numElem=0;
		for(int m=0;m<SSMLH.size()-1;m+=1){
				ssmContrastLH+=SSMContrast(SSMLH[m][m+1]);
				numElem+=1;
				if(m<SSMLH.size()-2){
					ssmContrastLH+=SSMContrast(SSMLH[m][m+2]);
					numElem+=1;
				}//endif
		}//endfor m
		ssmContrastLH/=numElem;

	}//end SetSSMContrast


	void SetMetLP(){

		nvLP=0; nvLPRH=0; nvLPLH=0;
		metLP=0;metLPRH=0;metLPLH=0;
		double nNote=0,nNoteRH=0,nNoteLH=0;

		int preMetpos=0,preRHMetpos=0,preLHMetpos=0;
		for(int m=0;m<qpr.bars.size();m+=1){
			if(qpr.bars[m].info.barlen!=nBeat){continue;}

			for(int l=0;l<qpr.bars[m].notes.size();l+=1){
				nNote+=1;
				int r=qpr.bars[m].notes[l].offstime-qpr.bars[m].notes[l].onstime-1;
				if(r>=nBeat){r=nBeat-1;}
				nvLP+=nvProb[qpr.bars[m].notes[l].onMetpos].LP[r];
				if(l==0){
					if(m==0){
						metLP+=uniMetProb.LP[qpr.bars[m].notes[l].onMetpos];
					}else{
						metLP+=trMetProb[preMetpos].LP[qpr.bars[m].notes[l].onMetpos];
						metLP+=chordProb[preMetpos].LP[0];
					}//endif
				}else if(qpr.bars[m].notes[l].onMetpos!=qpr.bars[m].notes[l-1].onMetpos){
					metLP+=trMetProb[qpr.bars[m].notes[l-1].onMetpos].LP[qpr.bars[m].notes[l].onMetpos];
					metLP+=chordProb[qpr.bars[m].notes[l-1].onMetpos].LP[0];
				}else{
					metLP+=chordProb[qpr.bars[m].notes[l-1].onMetpos].LP[1];
				}//endif
				preMetpos=qpr.bars[m].notes[l].onMetpos;
			}//endfor l

			for(int l=0;l<qprRH.bars[m].notes.size();l+=1){
				nNoteRH+=1;
				int r=qprRH.bars[m].notes[l].offstime-qprRH.bars[m].notes[l].onstime-1;
				if(r>=nBeat){r=nBeat-1;}
				nvLPRH+=nvRHProb[qprRH.bars[m].notes[l].onMetpos].LP[r];
				if(l==0){
					if(m==0){
						metLPRH+=uniMetRHProb.LP[qprRH.bars[m].notes[l].onMetpos];
					}else{
						metLPRH+=trMetRHProb[preRHMetpos].LP[qprRH.bars[m].notes[l].onMetpos];
						metLPRH+=chordRHProb[preRHMetpos].LP[0];
					}//endif
				}else if(qprRH.bars[m].notes[l].onMetpos!=qprRH.bars[m].notes[l-1].onMetpos){
					metLPRH+=trMetRHProb[qprRH.bars[m].notes[l-1].onMetpos].LP[qprRH.bars[m].notes[l].onMetpos];
					metLPRH+=chordRHProb[qprRH.bars[m].notes[l-1].onMetpos].LP[0];
				}else{
					metLPRH+=chordRHProb[qprRH.bars[m].notes[l-1].onMetpos].LP[1];
				}//endif
				preRHMetpos=qprRH.bars[m].notes[l].onMetpos;
			}//endfor l

			for(int l=0;l<qprLH.bars[m].notes.size();l+=1){
				nNoteLH+=1;
				int r=qprLH.bars[m].notes[l].offstime-qprLH.bars[m].notes[l].onstime-1;
				if(r>=nBeat){r=nBeat-1;}
				nvLPLH+=nvLHProb[qprLH.bars[m].notes[l].onMetpos].LP[r];
				if(l==0){
					if(m==0){
						metLPLH+=uniMetLHProb.LP[qprLH.bars[m].notes[l].onMetpos];
					}else{
						metLPLH+=trMetLHProb[preLHMetpos].LP[qprLH.bars[m].notes[l].onMetpos];
						metLPLH+=chordLHProb[preLHMetpos].LP[0];
					}//endif
				}else if(qprLH.bars[m].notes[l].onMetpos!=qprLH.bars[m].notes[l-1].onMetpos){
					metLPLH+=trMetLHProb[qprLH.bars[m].notes[l-1].onMetpos].LP[qprLH.bars[m].notes[l].onMetpos];
					metLPLH+=chordLHProb[qprLH.bars[m].notes[l-1].onMetpos].LP[0];
				}else{
					metLPLH+=chordLHProb[qprLH.bars[m].notes[l-1].onMetpos].LP[1];
				}//endif
				preLHMetpos=qprLH.bars[m].notes[l].onMetpos;
			}//endfor l

		}//endfor m

		nvLP/=nNote;
		nvLPRH/=nNoteRH;
		nvLPLH/=nNoteLH;
		metLP/=nNote;
		metLPRH/=nNoteRH;
		metLPLH/=nNoteLH;

	}//end SetMetLP


	void SetRpcLP(){

		//Estimate tonic
		vector<int> tonic(qpr.bars.size());
		vector<double> LP(12);
		vector<vector<int> > amax(qpr.bars.size());
		for(int m=0;m<qpr.bars.size();m+=1){
			amax[m].assign(12,0);
			if(m==0){
				LP.assign(12,0);
				if(qpr.bars[m].notes.size()==0){continue;}
				for(int q=0;q<12;q+=1){
					double logP=0;
					for(int l=0;l<qpr.bars[m].notes.size();l+=1){
						logP+=rpcUniProb.LP[(qpr.bars[m].notes[l].pitch-q+120)%12];
					}//endfor l
					logP/=double(qpr.bars[m].notes.size());
					LP[q]+=logP;
				}//endfor q
				continue;
			}//endif

			if(qpr.bars[m].notes.size()==0){
				for(int q=0;q<12;q+=1){
					amax[m][q]=q;
				}//endfor q
				continue;
			}//endif

			vector<double> preLP(LP);
			for(int q=0;q<12;q+=1){
				double logP;
				LP[q]=preLP[0]+((0==q)? -0.01106094735:-6.90775527898);//ln(0.001)=-6.90775527898 log(1-0.011)=-0.01106094735
				for(int qp=0;qp<12;qp+=1){
					logP=preLP[qp]+((qp==q)? -0.01106094735:-6.90775527898);
					if(logP>LP[q]){
						LP[q]=logP;
						amax[m][q]=qp;
					}//endif
				}//endfor qp

				logP=0;
				for(int l=0;l<qpr.bars[m].notes.size();l+=1){
					logP+=rpcUniProb.LP[(qpr.bars[m].notes[l].pitch-q+120)%12];
				}//endfor l
				logP/=double(qpr.bars[m].notes.size());
				LP[q]+=logP;
			}//endfor q

		}//endfor m

		//Backtrack
		tonic[qpr.bars.size()-1]=0;
		for(int q=0;q<12;q+=1){
			if(LP[q]>LP[tonic[qpr.bars.size()-1]]){tonic[qpr.bars.size()-1]=q;}
		}//endfor q
		for(int m=qpr.bars.size()-2;m>=0;m-=1){
			tonic[m]=amax[m+1][tonic[m+1]];
		}//endfor m

		QuantizedPianoRoll qprSplit=qpr;
		qprSplit.SplitNotesAcrossBars();
		QuantizedPianoRoll qprSplitRH=qprRH;
		qprSplitRH.SplitNotesAcrossBars();
		QuantizedPianoRoll qprSplitLH=qprLH;
		qprSplitLH.SplitNotesAcrossBars();

		assert(qprSplit.bars.size()==qpr.bars.size());
		assert(qprSplitRH.bars.size()==qpr.bars.size());
		assert(qprSplitLH.bars.size()==qpr.bars.size());

		rpcLP=0; rpcLPRH=0; rpcLPLH=0;
		double nBin=0,nBinRH=0,nBinLH=0;

		for(int m=0;m<qprSplit.bars.size();m+=1){

			for(int l=0;l<qprSplit.bars[m].notes.size();l+=1){
				for(int b=qprSplit.bars[m].notes[l].onMetpos;b<nBeat;b+=1){
					if(b>=qprSplit.bars[m].notes[l].onMetpos+qprSplit.bars[m].notes[l].offstime-qprSplit.bars[m].notes[l].onstime){break;}
					rpcLP+=rpcProb[b].LP[(qprSplit.bars[m].notes[l].pitch-tonic[m]+120)%12];
					nBin+=1;
				}//endfor b
			}//enbdfor l

			for(int l=0;l<qprSplitRH.bars[m].notes.size();l+=1){
				for(int b=qprSplitRH.bars[m].notes[l].onMetpos;b<nBeat;b+=1){
					if(b>=qprSplitRH.bars[m].notes[l].onMetpos+qprSplitRH.bars[m].notes[l].offstime-qprSplitRH.bars[m].notes[l].onstime){break;}
					rpcLPRH+=rpcRHProb[b].LP[(qprSplitRH.bars[m].notes[l].pitch-tonic[m]+120)%12];
					nBinRH+=1;
				}//endfor b
			}//enbdfor l

			for(int l=0;l<qprSplitLH.bars[m].notes.size();l+=1){
				for(int b=qprSplitLH.bars[m].notes[l].onMetpos;b<nBeat;b+=1){
					if(b>=qprSplitLH.bars[m].notes[l].onMetpos+qprSplitLH.bars[m].notes[l].offstime-qprSplitLH.bars[m].notes[l].onstime){break;}
					rpcLPLH+=rpcLHProb[b].LP[(qprSplitLH.bars[m].notes[l].pitch-tonic[m]+120)%12];
					nBinLH+=1;
				}//endfor b
			}//enbdfor l

		}//endfor m

		if(nBin>0){rpcLP/=nBin;}//endif
		if(nBinRH>0){rpcLPRH/=nBinRH;}//endif
		if(nBinLH>0){rpcLPLH/=nBinLH;}//endif

	}//end SetRpcLP

	void SetPitchRankLP(){
		pitchRankLP=0;

		double nNote=0;
		for(int l=0;l<qpr.evts.size();l+=1){
			vector<PitchRank> pitchRankSet;
			pitchRankSet.clear();
			for(int lp=l;lp<qpr.evts.size();lp+=1){
				if(pitchRankSet.size()>=pitchRankWidth){break;}
				pitchRankSet.push_back(PitchRank(qpr.evts[lp].pitch));
			}//endfor lp
			SetRank(pitchRankSet);

			pitchRankLP+=pitchRankProb[qpr.evts[l].onMetpos].LP[pitchRankSet[0].rank];
			nNote+=1;

		}//endfor l

		pitchRankLP/=nNote;

	}//end SetPitchRankLP


	void AnalyzeAll(){
		SetTieRate();
		SetSSM();
		SetSSMContrast();
		SetMetLP();
		SetRpcLP();
		SetPitchRankLP();
	}//end AnalyzeAll

};//endclass QprAnalyzer

class QprAnalyzerTrainer{
public:

	vector<QprAnalyzer> trainData;

	int nBeat;//for metrical analysis
	Prob<int> uniMetProb,uniMetRHProb,uniMetLHProb;//nBeat
	vector<Prob<int> > trMetProb,trMetRHProb,trMetLHProb;//nBeat x nBeat
	vector<Prob<int> > chordProb,chordRHProb,chordLHProb;//nBeat x 2 (0,1=transit,stay)
	vector<Prob<int> > nvProb,nvRHProb,nvLHProb;//nBeat x nBeat (r=0 <-> nv=1, nv>=nBeat <-> r=nBeat-1
	Prob<int> rpcUniProb;//12
	vector<Prob<int> > rpcProb,rpcRHProb,rpcLHProb;//nBeat x 12
	int pitchRankWidth;
	vector<Prob<int> > pitchRankProb;//nBeat x pitchRankWidth

	QprAnalyzerTrainer(){
		nBeat=48;
		pitchRankWidth=10;
	}//end QprAnalyzerTrainer
	~QprAnalyzerTrainer(){
	}//end ~QprAnalyzerTrainer

	void WriteParamFile(string filename){
		ofstream ofs(filename.c_str());

		ofs<<"#nBeat:\t"<<nBeat<<"\n";
		ofs<<"#pitchRankWidth:\t"<<pitchRankWidth<<"\n";

		ofs<<"### uniMetProb\n";
		for(int b=0;b<nBeat;b+=1){
			ofs<<uniMetProb.P[b]<<"\t";
		}//endfor b
		ofs<<"\n";
		ofs<<"### uniMetRHProb\n";
		for(int b=0;b<nBeat;b+=1){
			ofs<<uniMetRHProb.P[b]<<"\t";
		}//endfor b
		ofs<<"\n";
		ofs<<"### uniMetLHProb\n";
		for(int b=0;b<nBeat;b+=1){
			ofs<<uniMetLHProb.P[b]<<"\t";
		}//endfor b
		ofs<<"\n";

		ofs<<"### trMetProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ofs<<trMetProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b
		ofs<<"### trMetRHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ofs<<trMetRHProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b
		ofs<<"### trMetLHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ofs<<trMetLHProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b

		ofs<<"### chordProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ofs<<chordProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b
		ofs<<"### chordRHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ofs<<chordRHProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b
		ofs<<"### chordLHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ofs<<chordLHProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b

		ofs<<"### nvProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ofs<<nvProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b
		ofs<<"### nvRHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ofs<<nvRHProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b
		ofs<<"### nvLHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ofs<<nvLHProb[b].P[bp]<<"\t";
			}//endfor bp
			ofs<<"\n";
		}//endfor b

		ofs<<"### rpcUniProb\n";
		for(int q=0;q<12;q+=1){
			ofs<<rpcUniProb.P[q]<<"\t";
		}//endfor q
		ofs<<"\n";

		ofs<<"### rpcProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int q=0;q<12;q+=1){
				ofs<<rpcProb[b].P[q]<<"\t";
			}//endfor q
			ofs<<"\n";
		}//endfor b
		ofs<<"### rpcRHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int q=0;q<12;q+=1){
				ofs<<rpcRHProb[b].P[q]<<"\t";
			}//endfor q
			ofs<<"\n";
		}//endfor b
		ofs<<"### rpcLHProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int q=0;q<12;q+=1){
				ofs<<rpcLHProb[b].P[q]<<"\t";
			}//endfor q
			ofs<<"\n";
		}//endfor b

		ofs<<"### pitchRankProb\n";
		for(int b=0;b<nBeat;b+=1){
			for(int k=0;k<pitchRankWidth;k+=1){
				ofs<<pitchRankProb[b].P[k]<<"\t";
			}//endfor k
			ofs<<"\n";
		}//endfor b

		ofs.close();
	}//end WriteParamFile


	void ReadParamFile(string filename){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		ifstream ifs(filename.c_str());

		ifs>>s[1]>>nBeat;
		getline(ifs,s[99]);
		ifs>>s[1]>>pitchRankWidth;
		getline(ifs,s[99]);

		AssignZeros();

		getline(ifs,s[99]);//### uniMetProb
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniMetProb.P[b];
		}//endfor b
		uniMetProb.Normalize();
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### uniMetRHProb
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniMetRHProb.P[b];
		}//endfor b
		uniMetRHProb.Normalize();
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### uniMetLHProb
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniMetLHProb.P[b];
		}//endfor b
		uniMetLHProb.Normalize();
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### trMetProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>trMetProb[b].P[bp];
			}//endfor bp
			trMetProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### trMetRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>trMetRHProb[b].P[bp];
			}//endfor bp
			trMetRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### trMetLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>trMetLHProb[b].P[bp];
			}//endfor bp
			trMetLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### chordProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ifs>>chordProb[b].P[bp];
			}//endfor bp
			chordProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### chordRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ifs>>chordRHProb[b].P[bp];
			}//endfor bp
			chordRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### chordLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<2;bp+=1){
				ifs>>chordLHProb[b].P[bp];
			}//endfor bp
			chordLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### nvProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>nvProb[b].P[bp];
			}//endfor bp
			nvProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### nvRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>nvRHProb[b].P[bp];
			}//endfor bp
			nvRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### nvLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<nBeat;bp+=1){
				ifs>>nvLHProb[b].P[bp];
			}//endfor bp
			nvLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);


		getline(ifs,s[99]);//### rpcUniProb
		for(int b=0;b<12;b+=1){
			ifs>>rpcUniProb.P[b];
		}//endfor b
		rpcUniProb.Normalize();
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### rpcProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<12;bp+=1){
				ifs>>rpcProb[b].P[bp];
			}//endfor bp
			rpcProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### rpcRHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<12;bp+=1){
				ifs>>rpcRHProb[b].P[bp];
			}//endfor bp
			rpcRHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);
		getline(ifs,s[99]);//### rpcLHProb
		for(int b=0;b<nBeat;b+=1){
			for(int bp=0;bp<12;bp+=1){
				ifs>>rpcLHProb[b].P[bp];
			}//endfor bp
			rpcLHProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		getline(ifs,s[99]);//### pitchRankProb
		for(int b=0;b<nBeat;b+=1){
			for(int k=0;k<pitchRankWidth;k+=1){
				ifs>>pitchRankProb[b].P[k];
			}//endfor k
			pitchRankProb[b].Normalize();
		}//endfor b
		getline(ifs,s[99]);

		ifs.close();
	}//end ReadParamFile


	void ReadTrainData(string listFile,string folder){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		if(folder[folder.size()-1]!='/'){folder+="/";}
		trainData.clear();

		vector<string> filenames;
{
		ifstream ifs(listFile.c_str());
		while(ifs>>s[1]){
			filenames.push_back(s[1]);
			getline(ifs,s[99]);
		}//endwhile
		ifs.close();
}//

		for(int i=0;i<filenames.size();i+=1){
			ss.str(""); ss<<folder<<filenames[i]<<"_qpr.txt";
			QprAnalyzer analyer(ss.str());
			analyer.nBeat=nBeat;
			trainData.push_back(analyer);

			assert(analyer.qprRH.bars.size()==analyer.qpr.bars.size());
			assert(analyer.qprLH.bars.size()==analyer.qpr.bars.size());

		}//endfor i

	}//end ReadTrainData

	void AssignZeros(){
		uniMetProb.Assign(nBeat,1E-1);
		uniMetRHProb.Assign(nBeat,1E-1);
		uniMetLHProb.Assign(nBeat,1E-1);
		trMetProb.clear(); trMetRHProb.clear(); trMetLHProb.clear();
		chordProb.clear(); chordRHProb.clear(); chordLHProb.clear();
		nvProb.clear(); nvRHProb.clear(); nvLHProb.clear();
		pitchRankProb.clear();
		trMetProb.resize(nBeat); trMetRHProb.resize(nBeat); trMetLHProb.resize(nBeat);
		chordProb.resize(nBeat); chordRHProb.resize(nBeat); chordLHProb.resize(nBeat);
		nvProb.resize(nBeat); nvRHProb.resize(nBeat); nvLHProb.resize(nBeat);
		pitchRankProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			trMetProb[b].Assign(nBeat,1E-1);
			trMetRHProb[b].Assign(nBeat,1E-1);
			trMetLHProb[b].Assign(nBeat,1E-1);
			chordProb[b].Assign(2,1E-1);
			chordRHProb[b].Assign(2,1E-1);
			chordLHProb[b].Assign(2,1E-1);
			nvProb[b].Assign(nBeat,1E-1);
			nvRHProb[b].Assign(nBeat,1E-1);
			nvLHProb[b].Assign(nBeat,1E-1);
			pitchRankProb[b].Assign(pitchRankWidth,1E-1);
		}//endfor b

		rpcUniProb.Assign(12,1E-1);
		rpcProb.clear(); rpcRHProb.clear(); rpcLHProb.clear();
		rpcProb.resize(nBeat); rpcRHProb.resize(nBeat); rpcLHProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			rpcProb[b].Assign(12,1E-1);
			rpcRHProb[b].Assign(12,1E-1);
			rpcLHProb[b].Assign(12,1E-1);
		}//endfor b
	}//end AssignZeros

	void LearnMetProb(){
		//Assume hand separation

		uniMetProb.Assign(nBeat,1E-1);
		uniMetRHProb.Assign(nBeat,1E-1);
		uniMetLHProb.Assign(nBeat,1E-1);
		trMetProb.clear(); trMetRHProb.clear(); trMetLHProb.clear();
		chordProb.clear(); chordRHProb.clear(); chordLHProb.clear();
		nvProb.clear(); nvRHProb.clear(); nvLHProb.clear();
		trMetProb.resize(nBeat); trMetRHProb.resize(nBeat); trMetLHProb.resize(nBeat);
		chordProb.resize(nBeat); chordRHProb.resize(nBeat); chordLHProb.resize(nBeat);
		nvProb.resize(nBeat); nvRHProb.resize(nBeat); nvLHProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			trMetProb[b].Assign(nBeat,1E-1);
			trMetRHProb[b].Assign(nBeat,1E-1);
			trMetLHProb[b].Assign(nBeat,1E-1);
			chordProb[b].Assign(2,1E-1);
			chordRHProb[b].Assign(2,1E-1);
			chordLHProb[b].Assign(2,1E-1);
			nvProb[b].Assign(nBeat,1E-1);
			nvRHProb[b].Assign(nBeat,1E-1);
			nvLHProb[b].Assign(nBeat,1E-1);
		}//endfor b

		for(int n=0;n<trainData.size();n+=1){

			int preMetpos=0;
			for(int m=0;m<trainData[n].qpr.bars.size();m+=1){
				if(trainData[n].qpr.bars[m].info.barlen!=nBeat){continue;}
				for(int l=0;l<trainData[n].qpr.bars[m].notes.size();l+=1){
					uniMetProb.P[trainData[n].qpr.bars[m].notes[l].onMetpos]+=1;
					int r=trainData[n].qpr.bars[m].notes[l].offstime-trainData[n].qpr.bars[m].notes[l].onstime-1;
					if(r>=nBeat){r=nBeat-1;}
					nvProb[trainData[n].qpr.bars[m].notes[l].onMetpos].P[r]+=1;
					if(l==0){
						if(m>0){
							trMetProb[preMetpos].P[trainData[n].qpr.bars[m].notes[l].onMetpos]+=1;
							chordProb[preMetpos].P[0]+=1;
						}//endif
					}else{//l>0
						if(trainData[n].qpr.bars[m].notes[l].onMetpos!=trainData[n].qpr.bars[m].notes[l-1].onMetpos){
							trMetProb[trainData[n].qpr.bars[m].notes[l-1].onMetpos].P[trainData[n].qpr.bars[m].notes[l].onMetpos]+=1;
							chordProb[trainData[n].qpr.bars[m].notes[l-1].onMetpos].P[0]+=1;
						}else{
							chordProb[trainData[n].qpr.bars[m].notes[l-1].onMetpos].P[1]+=1;
						}//endif
					}//endif
					preMetpos=trainData[n].qpr.bars[m].notes[l].onMetpos;
				}//enbdfor l
			}//endfor m

			preMetpos=0;
			for(int m=0;m<trainData[n].qprRH.bars.size();m+=1){
				if(trainData[n].qprRH.bars[m].info.barlen!=nBeat){continue;}
				for(int l=0;l<trainData[n].qprRH.bars[m].notes.size();l+=1){
					uniMetRHProb.P[trainData[n].qprRH.bars[m].notes[l].onMetpos]+=1;
					int r=trainData[n].qprRH.bars[m].notes[l].offstime-trainData[n].qprRH.bars[m].notes[l].onstime-1;
					if(r>=nBeat){r=nBeat-1;}
					nvRHProb[trainData[n].qprRH.bars[m].notes[l].onMetpos].P[r]+=1;
					if(l==0){
						if(m>0){
							trMetRHProb[preMetpos].P[trainData[n].qprRH.bars[m].notes[l].onMetpos]+=1;
							chordRHProb[preMetpos].P[0]+=1;
						}//endif
					}else{//l>0
						if(trainData[n].qprRH.bars[m].notes[l].onMetpos!=trainData[n].qprRH.bars[m].notes[l-1].onMetpos){
							trMetRHProb[trainData[n].qprRH.bars[m].notes[l-1].onMetpos].P[trainData[n].qprRH.bars[m].notes[l].onMetpos]+=1;
							chordRHProb[trainData[n].qprRH.bars[m].notes[l-1].onMetpos].P[0]+=1;
						}else{
							chordRHProb[trainData[n].qprRH.bars[m].notes[l-1].onMetpos].P[1]+=1;
						}//endif
					}//endif
					preMetpos=trainData[n].qprRH.bars[m].notes[l].onMetpos;
				}//enbdfor l
			}//endfor m

			preMetpos=0;
			for(int m=0;m<trainData[n].qprLH.bars.size();m+=1){
				if(trainData[n].qprLH.bars[m].info.barlen!=nBeat){continue;}
				for(int l=0;l<trainData[n].qprLH.bars[m].notes.size();l+=1){
					uniMetLHProb.P[trainData[n].qprLH.bars[m].notes[l].onMetpos]+=1;
					int r=trainData[n].qprLH.bars[m].notes[l].offstime-trainData[n].qprLH.bars[m].notes[l].onstime-1;
					if(r>=nBeat){r=nBeat-1;}
					nvLHProb[trainData[n].qprLH.bars[m].notes[l].onMetpos].P[r]+=1;
					if(l==0){
						if(m>0){
							trMetLHProb[preMetpos].P[trainData[n].qprLH.bars[m].notes[l].onMetpos]+=1;
							chordLHProb[preMetpos].P[0]+=1;
						}//endif
					}else{//l>0
						if(trainData[n].qprLH.bars[m].notes[l].onMetpos!=trainData[n].qprLH.bars[m].notes[l-1].onMetpos){
							trMetLHProb[trainData[n].qprLH.bars[m].notes[l-1].onMetpos].P[trainData[n].qprLH.bars[m].notes[l].onMetpos]+=1;
							chordLHProb[trainData[n].qprLH.bars[m].notes[l-1].onMetpos].P[0]+=1;
						}else{
							chordLHProb[trainData[n].qprLH.bars[m].notes[l-1].onMetpos].P[1]+=1;
						}//endif
					}//endif
					preMetpos=trainData[n].qprLH.bars[m].notes[l].onMetpos;
				}//enbdfor l
			}//endfor m

		}//endfor n

		uniMetProb.Normalize();
		uniMetRHProb.Normalize();
		uniMetLHProb.Normalize();
		for(int b=0;b<nBeat;b+=1){
			trMetProb[b].Normalize();
			trMetRHProb[b].Normalize();
			trMetLHProb[b].Normalize();
			chordProb[b].Normalize();
			chordRHProb[b].Normalize();
			chordLHProb[b].Normalize();
			nvProb[b].Normalize();
			nvRHProb[b].Normalize();
			nvLHProb[b].Normalize();
		}//endfor b

	}//end LearnMetProb


	void LearnRpcProb(){
		//Assume hand separation

		rpcUniProb.Assign(12,1E-1);
		rpcProb.clear(); rpcRHProb.clear(); rpcLHProb.clear();
		rpcProb.resize(nBeat); rpcRHProb.resize(nBeat); rpcLHProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			rpcProb[b].Assign(12,1E-1);
			rpcRHProb[b].Assign(12,1E-1);
			rpcLHProb[b].Assign(12,1E-1);
		}//endfor b

		for(int n=0;n<trainData.size();n+=1){
			QuantizedPianoRoll qprSplit=trainData[n].qpr;
			qprSplit.SplitNotesAcrossBars();
			QuantizedPianoRoll qprSplitRH=trainData[n].qprRH;
			qprSplitRH.SplitNotesAcrossBars();
			QuantizedPianoRoll qprSplitLH=trainData[n].qprLH;
			qprSplitLH.SplitNotesAcrossBars();

			assert(qprSplit.bars.size()==trainData[n].qpr.bars.size());
			assert(qprSplitRH.bars.size()==trainData[n].qpr.bars.size());
			assert(qprSplitLH.bars.size()==trainData[n].qpr.bars.size());

			int curKeyPos=0;
			for(int m=0;m<qprSplit.bars.size();m+=1){

				if(curKeyPos<qprSplit.keyEvts.size()-1){
					if(qprSplit.bars[m].info.stime>=qprSplit.keyEvts[curKeyPos+1].stime){curKeyPos+=1;}
				}//endif
				if(qprSplit.bars[m].info.barlen!=nBeat){continue;}
				int tonic=(qprSplit.keyEvts[curKeyPos].keyfifth*7+1200)%12;

				for(int l=0;l<qprSplit.bars[m].notes.size();l+=1){
					rpcUniProb.P[(qprSplit.bars[m].notes[l].pitch-tonic+120)%12]+=1;
					for(int b=qprSplit.bars[m].notes[l].onMetpos;b<nBeat;b+=1){
						if(b>=qprSplit.bars[m].notes[l].onMetpos+qprSplit.bars[m].notes[l].offstime-qprSplit.bars[m].notes[l].onstime){break;}
						rpcProb[b].P[(qprSplit.bars[m].notes[l].pitch-tonic+120)%12]+=1;
					}//endfor b
				}//enbdfor l

				for(int l=0;l<qprSplitRH.bars[m].notes.size();l+=1){
					for(int b=qprSplitRH.bars[m].notes[l].onMetpos;b<nBeat;b+=1){
						if(b>=qprSplitRH.bars[m].notes[l].onMetpos+qprSplitRH.bars[m].notes[l].offstime-qprSplitRH.bars[m].notes[l].onstime){break;}
						rpcRHProb[b].P[(qprSplitRH.bars[m].notes[l].pitch-tonic+120)%12]+=1;
					}//endfor b
				}//enbdfor l

				for(int l=0;l<qprSplitLH.bars[m].notes.size();l+=1){
					for(int b=qprSplitLH.bars[m].notes[l].onMetpos;b<nBeat;b+=1){
						if(b>=qprSplitLH.bars[m].notes[l].onMetpos+qprSplitLH.bars[m].notes[l].offstime-qprSplitLH.bars[m].notes[l].onstime){break;}
						rpcLHProb[b].P[(qprSplitLH.bars[m].notes[l].pitch-tonic+120)%12]+=1;
					}//endfor b
				}//enbdfor l

			}//endfor m

		}//endfor n

		rpcUniProb.Normalize();
		for(int b=0;b<nBeat;b+=1){
			rpcProb[b].Normalize();
			rpcRHProb[b].Normalize();
			rpcLHProb[b].Normalize();
		}//endfor b

	}//end LearnRpcProb

	void LearnPitchRankProb(){

		pitchRankProb.clear();
		pitchRankProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			pitchRankProb[b].Assign(pitchRankWidth,1E-1);
		}//endfor b

		for(int n=0;n<trainData.size();n+=1){

			for(int l=0;l<trainData[n].qpr.evts.size();l+=1){
				vector<PitchRank> pitchRankSet;
				pitchRankSet.clear();
				for(int lp=l;lp<trainData[n].qpr.evts.size();lp+=1){
					if(pitchRankSet.size()>=pitchRankWidth){break;}
					pitchRankSet.push_back(PitchRank(trainData[n].qpr.evts[lp].pitch));
				}//endfor lp
				SetRank(pitchRankSet);
				pitchRankProb[trainData[n].qpr.evts[l].onMetpos].P[pitchRankSet[0].rank]+=1;
			}//endfor l

		}//endfor n

		for(int b=0;b<nBeat;b+=1){
			pitchRankProb[b].Normalize();
		}//endfor b

	}//end LearnPitchRankProb


};//endclass QprAnalyzerTrainer


#endif // QprAnalyzer_HPP
