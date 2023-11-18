#ifndef PolyMetHMM_HPP
#define PolyMetHMM_HPP

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

class PolyMetHMM{
public:

	int TPQN;
	int nBeat;//Beat resolution = bar length
	Prob<int> uniProb;//nBeat
	vector<double> uniDirParam;//nBeat

	Prob<int> iniProb;
	vector<Prob<int> > trProb;//(nBeat x nBeat)
	vector<double> iniDirParam;//nBeat
	vector<vector<double> > trDirParam;//nBeat x nBeat
	vector<Prob<int> > chordProb;//(nBeat x 2) 0,1=transit,stay

	int pitchRankWidth;
	vector<Prob<int> > pitchRankProb;//2(0:offBeat/1:onBeat) x pitchRankWidth
	double lamPitchRank;

	double bpm;//[QN/min] (ref:144)
	double sig_t;//[sec] (ref=0.02)
	double secPerTick;//=secPerQN/TPQN
	double SigmaV;
	double lambda;//chordalIOIのscale parameter
	double fac_t;
	double fac_lam;
	vector<double> secPerQN;
	vector<Prob<int> > tempoTrProb;
	Prob<int> tempoTrIniProb;
	int numTempoState;
	double minSecPerQN,maxSeqPerQN;
	int pruningCutOff;
	int tempoSearchHalfWidth;

	int nStates;//=nBeat*numTempoState*2; z=2*(v*nBeat+b)+g, v=z/nBeat/2, b=(z/2)%nBeat, g=z%2;

	//For training
	vector<QuantizedPianoRoll> trainData;
	vector<vector<vector<int> > > data;//piece,bar,note -> metpos
	//For transcription
	vector<PianoRoll> testData;
	vector<QuantizedPianoRoll> estimatedData;//corresponding to testData
	vector<double> maxLP;//max log probability after Viterbi 
	vector<vector<int> > estTempoVar;//corresponding to testData

	int nPiece;
	int nBar;
	int nNote;

	double EPS;
	bool PRINTON;

	PolyMetHMM(){
		EPS=0.1;
		PRINTON=false;
		secPerTick=0.1;
		pitchRankWidth=10;
		lamPitchRank=0;
		Init();
	}//end PolyMetHMM
	PolyMetHMM(string paramfile){
		EPS=0.1;
		PRINTON=false;
		secPerTick=0.1;
		pitchRankWidth=10;
		lamPitchRank=0;
		Init();
		ReadFile(paramfile);
	}//end PolyMetHMM
	~PolyMetHMM(){
	}//end ~PolyMetHMM

	void Init(double sig_t_=0.03,double SigmaV_=0.1){

		minSecPerQN=0.3;//BPM=200
		maxSeqPerQN=1.5;//BPM=40
		numTempoState=50;
// 		sig_t=0.02;//TASLP1 0.014
// 		SigmaV=1;//3.32 × 10^−2
		sig_t=sig_t_;//0.03
		SigmaV=SigmaV_;//0.1
// 		minSecPerQN=0.3125;//BPM=192
// 		maxSeqPerQN=1.;//BPM=60
// 		numTempoState=36;
		nStates=numTempoState*nBeat*2;
		pruningCutOff=nStates;
		lambda=0.0101;//0.0101
		fac_t=-0.5*log(2*M_PI*sig_t*sig_t);
		fac_lam=-log(lambda);
		tempoSearchHalfWidth=2;

		double eps=log(maxSeqPerQN/minSecPerQN)/numTempoState;
		for(int i=0;i<numTempoState;i+=1){
			secPerQN.push_back(minSecPerQN*exp(i*eps));
		}//endfor i

		tempoTrProb.resize(numTempoState);
		for(int i=0;i<numTempoState;i+=1){
			tempoTrProb[i].P.resize(numTempoState);
			for(int j=0;j<numTempoState;j+=1){
				tempoTrProb[i].P[j]=exp(-0.5*pow((i-j)/SigmaV,2.));
			}//endfor j
			tempoTrProb[i].Normalize();
		}//endfor i

		tempoTrIniProb.P.resize(numTempoState);
////////////////// Original
		//mean=89.4, stdev = 
		for(int i=0;i<numTempoState;i+=1){
			tempoTrIniProb.P[i]=exp(-0.5*pow((i-numTempoState/2)/(3*SigmaV),2.))+1E-30;
		}//endfor i
////////////////// Uniform
// 		tempoTrIniProb.P.assign(numTempoState,1);
////////////////// MuseScore Data
		//mean = 112.586 (60/112.586=0.532926), stdev = 35.44 (60/(112.586+35.44) - 0.532926 = -0.12759)
		//geo mean = 10**(2.03098) = 107.4, geo stdev = 10**(2.03098+0.133682)-107.4 = 38.703964304
// 		for(int i=0;i<numTempoState;i+=1){
// 			tempoTrIniProb.P[i]=exp( -0.5*pow((secPerQN[i]-0.532926)/0.12759,2.) )+1E-30;
// 		}//endfor i
//////////////////
		tempoTrIniProb.Normalize();

		pitchRankProb.resize(2);
		pitchRankProb[0].Assign(pitchRankWidth,1);
		pitchRankProb[1].Assign(pitchRankWidth,1);
		pitchRankProb[0].Normalize();
		pitchRankProb[1].Normalize();

	}//end Init


	void RandomInit(int nBeat_){
		nBeat=nBeat_;
		Init();
		uniProb.Resize(nBeat);
		uniProb.Randomize();
		uniDirParam.assign(nBeat,1);
		iniProb.Resize(nBeat);
		iniProb.Randomize();
		iniDirParam.assign(nBeat,1);
		trProb.resize(nBeat);
		trDirParam.resize(nBeat);
		chordProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
			trProb[b].Resize(nBeat);
			trProb[b].Randomize();
			trDirParam[b].assign(nBeat,1);
			chordProb[b].Resize(2);
			chordProb[b].Randomize();
		}//endfor b
	}//end RandomInit

	void WriteFile(string filename){
		ofstream ofs(filename.c_str());
		ofs<<"#TPQN: "<<TPQN<<"\n";
		ofs<<"#nBeat: "<<nBeat<<"\n";

		ofs<<"### Unigram Prob\n";
		for(int b=0;b<nBeat;b+=1){
ofs<<uniProb.P[b]<<"\t";
		}//endfor b
ofs<<"\n";

		ofs<<"### Init Prob\n";
		for(int b=0;b<nBeat;b+=1){
ofs<<iniProb.P[b]<<"\t";
		}//endfor b
ofs<<"\n";

		ofs<<"### Transition Prob\n";
		for(int b=0;b<nBeat;b+=1){
			for(int b_=0;b_<nBeat;b_+=1){
ofs<<trProb[b].P[b_]<<"\t";
			}//endfor b_
ofs<<"\n";
		}//endfor b

		ofs<<"### Chord Transition Prob\n";
		for(int b=0;b<nBeat;b+=1){
			for(int g=0;g<2;g+=1){
ofs<<chordProb[b].P[g]<<"\t";
			}//endfor g
ofs<<"\n";
		}//endfor b

		ofs.close();
	}//end WriteFile

	void ReadFile(string filename){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		ifstream ifs(filename.c_str());
		ifs>>s[1]>>TPQN;
		getline(ifs,s[99]);
		ifs>>s[1]>>nBeat;
		getline(ifs,s[99]);

		RandomInit(nBeat);

		getline(ifs,s[99]);//### Unigram Prob
		for(int b=0;b<nBeat;b+=1){
			ifs>>uniProb.P[b];
		}//endfor b
		getline(ifs,s[99]);
		uniProb.Normalize();

		getline(ifs,s[99]);//### Init Prob
		for(int b=0;b<nBeat;b+=1){
			ifs>>iniProb.P[b];
		}//endfor b
		getline(ifs,s[99]);
		iniProb.Normalize();

		getline(ifs,s[99]);//### Transition Prob
		for(int b=0;b<nBeat;b+=1){
			for(int b_=0;b_<nBeat;b_+=1){
				ifs>>trProb[b].P[b_];
			}//endfor b_
			getline(ifs,s[99]);
			trProb[b].Normalize();
		}//endfor b

		getline(ifs,s[99]);//### Chord Transition Prob
		for(int b=0;b<nBeat;b+=1){
			for(int g=0;g<2;g+=1){
				ifs>>chordProb[b].P[g];
			}//endfor g
			getline(ifs,s[99]);
			chordProb[b].Normalize();
		}//endfor b

		ifs.close();
	}//end ReadFile

	void ReadPitchRankProb(string filename){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;
		ifstream ifs(filename.c_str());
		getline(ifs,s[99]);
		for(int i=0;i<pitchRankWidth;i+=1){
			ifs>>v[0]>>pitchRankProb[0].P[i]>>pitchRankProb[1].P[i];
		}//endfor i
		ifs.close();
		pitchRankProb[0].Normalize();
		pitchRankProb[1].Normalize();
	}//end ReadPitchRankProb

	void ReadData(string listFile,string folder){
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
			QuantizedPianoRoll qpr(ss.str());
			qpr.ChangeTPQN(TPQN);
			trainData.push_back(qpr);
		}//endfor i

	}//end ReadData
	
	void SetUpData(){

		data.clear();
		data.resize(trainData.size());

		for(int i=0;i<trainData.size();i+=1){
			trainData[i].SplitBars(48);
			vector<int> notes;
			vector<vector<int> > piece;
			for(int m=0;m<trainData[i].bars.size();m+=1){
				if(trainData[i].bars[m].info.barlen!=nBeat){continue;}
				notes.clear();
				for(int n=0;n<trainData[i].bars[m].notes.size();n+=1){
					if(trainData[i].bars[m].notes[n].pitch<0){continue;}
					notes.push_back(trainData[i].bars[m].notes[n].onMetpos);
				}//endfor n
				if(notes.size()==0){continue;}
//				sort(notes.begin(),notes.end());
				piece.push_back(notes);
			}//endfor m
			data[i]=piece;
		}//endfor i

		nPiece=data.size();
		nBar=0;
		nNote=0;
		for(int l=0;l<nPiece;l+=1){
			nBar+=data[l].size();
			for(int m=0;m<data[l].size();m+=1){
				nNote+=data[l][m].size();
			}//endfor m
		}//endfor l
if(PRINTON){cout<<"#nPiece,nBar,nNote:\t"<<nPiece<<"\t"<<nBar<<"\t"<<nNote<<endl;}

	}//end SetUpData

	void LearnAdditiveSmoothing(double fac=0.1){
		uniProb.P.assign(nBeat,fac);
		for(int l=0;l<data.size();l+=1){
			for(int m=0;m<data[l].size();m+=1){
				for(int n=0;n<data[l][m].size();n+=1){
					uniProb.P[data[l][m][n]]+=1;
				}//endfor n
			}//endfor m
		}//endfor l
		uniProb.Normalize();
		iniProb.P.assign(nBeat,fac);
		trProb.resize(nBeat);
		chordProb.resize(nBeat);
		for(int b=0;b<nBeat;b+=1){
//			trProb[b].P.assign(nBeat,fac);
			trProb[b].P=uniProb.P;
			chordProb[b].P.assign(2,fac);
			chordProb[b].P[0]=1;
		}//endfor b

		for(int l=0;l<data.size();l+=1){
			if(data[l].size()==0){continue;}
			iniProb.P[data[l][0][0]]+=1;
			for(int m=0;m<data[l].size();m+=1){
				for(int n=1;n<data[l][m].size();n+=1){
					if(data[l][m][n-1]==data[l][m][n]){//stay
						chordProb[data[l][m][n-1]].P[1]+=1;
					}else{//transit
						chordProb[data[l][m][n-1]].P[0]+=1;
						trProb[data[l][m][n-1]].P[data[l][m][n]]+=1;
					}//endif
				}//endfor n
				if(m>0){
					trProb[data[l][m-1][data[l][m-1].size()-1]].P[data[l][m][0]]+=1;
					chordProb[data[l][m-1][data[l][m-1].size()-1]].P[0]+=1;//transit
				}//endif
			}//endfor m
		}//endfor l

		iniProb.Normalize();
		for(int b=0;b<nBeat;b+=1){
			trProb[b].Normalize();
			chordProb[b].Normalize();
		}//endfor b

	}//end LearnAdditiveSmoothing

	void SetDirParam(double alpha){
		for(int b=0;b<nBeat;b+=1){
			iniDirParam[b]=alpha*iniProb.P[b];
			for(int bp=0;bp<nBeat;bp+=1){
				trDirParam[b][bp]=alpha*trProb[b].P[bp];
			}//endfor bp
		}//endfor b
	}//end SetDirParam

	double GetLP(){
		double LP=0;

		for(int l=0;l<data.size();l+=1){
			if(data[l].size()==0){continue;}
			double partLP=0;
			partLP+=iniProb.LP[data[l][0][0]];
			for(int m=0;m<data[l].size();m+=1){
				for(int n=1;n<data[l][m].size();n+=1){
					if(data[l][m][n-1]==data[l][m][n]){//stay
						partLP+=chordProb[data[l][m][n-1]].LP[1];
					}else{//transit
						partLP+=chordProb[data[l][m][n-1]].LP[0]+trProb[data[l][m][n-1]].LP[data[l][m][n]];
					}//endif
				}//endfor n
				if(m>0){
					partLP+=chordProb[data[l][m-1][data[l][m-1].size()-1]].LP[0]+trProb[data[l][m-1][data[l][m-1].size()-1]].LP[data[l][m][0]];
				}//endif
			}//endfor m
//cout<<l<<"\t"<<partLP<<endl;
			LP+=partLP;
		}//endfor l

		return LP;
	}//end GetLP


	double OutLP(int bp,int b,double secPerQN_,double dur){
		int nv=b-bp;
		if(nv<=0){nv+=nBeat;}
		if(dur>nv*secPerQN_/double(TPQN)+10*sig_t){
			double min=dur-nv*secPerQN_/double(TPQN);
			int kmin=0;
			for(int k=1;k<100;k+=1){
				if( abs(dur-(nv+k*nBeat)*secPerQN_/double(TPQN)) < min ){
					min=abs(dur-(nv+k*nBeat)*secPerQN_/double(TPQN));
					continue;
				}else{
					kmin=k-1;
					break;
				}//endif
			}//endfor k
			if(kmin<=1){
				return fac_t-0.5*pow((dur-(nv+kmin*nBeat)*secPerQN_/double(TPQN))/((kmin+1)*sig_t),2.)-1;
			}else{
				return fac_t-0.5*pow((dur-(nv+kmin*nBeat)*secPerQN_/double(TPQN))/((kmin+1)*sig_t),2.)-7;//-7?
			}//endif
		}else{
			return fac_t-0.5*pow((dur-nv*secPerQN_/double(TPQN))/sig_t,2.);
		}//endif
	}//end OutLP


	void Transcribe(){

		double betaOutput=1;

		estimatedData.clear();
		maxLP.clear();
		estTempoVar.clear();

		for(int l=0;l<testData.size();l+=1){
			QuantizedPianoRoll qpr;
			qpr.TPQN=TPQN;
			qpr.instrEvts[0].programChange=0;
			qpr.meterEvts[0].num=nBeat/TPQN;
			qpr.meterEvts[0].den=4;
			qpr.meterEvts[0].barlen=nBeat;

			QPREvt evt;
			for(int n=0;n<testData[l].evts.size();n+=1){
				evt.ID=testData[l].evts[n].ID;
				evt.onstime=0;
				evt.offstime=0;
				evt.sitch=testData[l].evts[n].sitch;
				evt.pitch=testData[l].evts[n].pitch;
				qpr.evts.push_back(evt);
			}//endfor n

			estimatedData.push_back(qpr);
		}//endfor l

		maxLP.resize(testData.size());
		estTempoVar.resize(testData.size());

		for(int l=0;l<testData.size();l+=1){
			if(testData[l].evts.size()<2){continue;}
//cout<<"Transcribe -- "<<(l+1)<<"/"<<ontimes.size()<<endl;

			///Viterbi
			int len=testData[l].evts.size()+1;
			vector<int> optPath(len);

			vector<double> LP;
			LP.assign(nStates,-DBL_MAX);
			vector<vector<int> > amax;
			amax.resize(len);
			int b,v,g,bp,vp,gp;

			vector<PitchRank> pitchRankSet;
			pitchRankSet.clear();
			for(int np=0;np<testData[l].evts.size();np+=1){
				if(pitchRankSet.size()>=pitchRankWidth){break;}
				pitchRankSet.push_back(PitchRank(testData[l].evts[np].pitch));
			}//endfor np
			SetRank(pitchRankSet);

//	int nStates;//=nBeat*numTempoState*2; z=2*(v*nBeat+b)+g, v=z/nBeat/2, b=(z/2)%nBeat, g=z%2;

			/// ///Initialization
			for(b=0;b<nBeat;b+=1){
				for(v=0;v<numTempoState;v+=1){
					LP[2*(v*nBeat+b)]=uniProb.LP[b]+tempoTrIniProb.LP[v];
					LP[2*(v*nBeat+b)]+=lamPitchRank*((b==0)? pitchRankProb[0].LP[pitchRankSet[0].rank]:pitchRankProb[1].LP[pitchRankSet[0].rank]);
				}//endfor v
			}//endfor b

			/// ///Update
			double logP;
			double dur;

			for(int n=1;n<len;n+=1){

				amax[n].resize(nStates);
				vector<double> preLP(LP);
				if(n==len-1){
					dur=testData[l].evts[n-1].offtime-testData[l].evts[n-1].ontime;
				}else{
					dur=testData[l].evts[n].ontime-testData[l].evts[n-1].ontime;
				}//endif

				pitchRankSet.clear();
				for(int np=n;np<testData[l].evts.size();np+=1){
					if(pitchRankSet.size()>=pitchRankWidth){break;}
					pitchRankSet.push_back(PitchRank(testData[l].evts[np].pitch));
				}//endfor np
				SetRank(pitchRankSet);

				for(int z=0;z<nStates;z+=1){
					LP[z]=-DBL_MAX;
					b=(z/2)%nBeat;
					v=(z/2)/nBeat;
					g=z%2;
					if(b%TPQN==1 || b%TPQN==5 || b%TPQN==7 || b%TPQN==11){continue;}

					if(g==1){//stay -> vp=v, bp=b

//						LP[z]=preLP[2*(v*nBeat+b)]+chordProb[b].LP[g]+tempoTrProb[v].LP[v]+fac_lam-dur/lambda;
						LP[z]=preLP[2*(v*nBeat+b)]+chordProb[b].LP[g]+betaOutput*(fac_lam-dur/lambda);
						amax[n][z]=2*(v*nBeat+b);

//						logP=preLP[2*(v*nBeat+b)+1]+chordProb[b].LP[g]+tempoTrProb[v].LP[v]+fac_lam-dur/lambda;
						logP=preLP[2*(v*nBeat+b)+1]+chordProb[b].LP[g]+betaOutput*(fac_lam-dur/lambda);
						if(logP>LP[z]){
							LP[z]=logP;
							amax[n][z]=2*(v*nBeat+b)+1;
						}//endif

					}else{//g=0 transit

						amax[n][z]=0;

						for(vp=v-tempoSearchHalfWidth;vp<=v+tempoSearchHalfWidth;vp+=1){
							if(vp<0 || vp>=numTempoState){continue;}
							for(bp=0;bp<nBeat;bp+=1){
								int nv_mod=(b-bp+nBeat)%TPQN;
								if(nv_mod==1 || nv_mod==5 || nv_mod==7 || nv_mod==11){continue;}
								for(gp=0;gp<2;gp+=1){
									int zp=2*(vp*nBeat+bp)+gp;
									logP=preLP[zp]+chordProb[bp].LP[g]+trProb[bp].LP[b]+tempoTrProb[vp].LP[v]+betaOutput*OutLP(bp,b,secPerQN[v],dur);

									if(logP>LP[z]){
										LP[z]=logP;
										amax[n][z]=zp;
									}//endif
								}//endfor gp
							}//endfor bp
						}//endfor vp

					}//endif

					LP[z]+=lamPitchRank*((b==0)? pitchRankProb[0].LP[pitchRankSet[0].rank]:pitchRankProb[1].LP[pitchRankSet[0].rank]);

				}//endfor z

			}//endfor n

			/// ///Backtracking and set stimes
			optPath[optPath.size()-1]=0;
			for(int z=0;z<nStates;z+=1){
				if(LP[z]>LP[optPath[optPath.size()-1]]){
					optPath[optPath.size()-1]=z;
				}//endif
			}//endfor z
			maxLP[l]=LP[optPath[optPath.size()-1]];
			for(int n=optPath.size()-2;n>=0;n-=1){
				optPath[n]=amax[n+1][optPath[n+1]];
			}//endfor n

			estTempoVar[l].resize(len);
			for(int n=0;n<len;n+=1){
				estTempoVar[l][n]=optPath[n]/nBeat/2;
			}//endfor n

			//Set onstime
			estimatedData[l].evts[0].onstime=(optPath[0]/2)%nBeat;
			for(int n=1;n<len;n+=1){
				int nv;
				if(optPath[n]%2==1){//stay -> chord transition
					nv=0;
				}else{//transit
					nv=(optPath[n]/2)%nBeat-(optPath[n-1]/2)%nBeat;
					if(n==len-1){
						dur=testData[l].evts[n-1].offtime-testData[l].evts[n-1].ontime;
					}else{
						dur=testData[l].evts[n].ontime-testData[l].evts[n-1].ontime;
					}//endif
					if(nv<=0){nv+=nBeat;}
					for(int k=1;k<100;k+=1){
						if( abs(dur-(nv+k*nBeat)*secPerQN[(optPath[n]/2)/nBeat]/double(TPQN))>abs(dur-(nv+(k-1)*nBeat)*secPerQN[(optPath[n]/2)/nBeat]/double(TPQN)) ){
							nv=nv+(k-1)*nBeat;
							break;
						}//endif
					}//endfor k
				}//endif

				if(n==len-1){
					estimatedData[l].evts[n-1].offstime=estimatedData[l].evts[n-1].onstime+nv;
				}else{
					estimatedData[l].evts[n].onstime=estimatedData[l].evts[n-1].onstime+nv;
				}//endif

			}//endfor n

			//Set offstime
{
			int maxOffstime=0;
			vector<int> chordOnsetPos;
			chordOnsetPos.push_back(0);
			for(int n=1;n<len-1;n+=1){
				if(estimatedData[l].evts[n].onstime!=estimatedData[l].evts[n-1].onstime){
					chordOnsetPos.push_back(n);
				}//endif
			}//endfor n
			for(int i=0;i<chordOnsetPos.size();i+=1){
				if(i==chordOnsetPos.size()-1){
					for(int n=chordOnsetPos[i];n<estimatedData[l].evts.size();n+=1){
						estimatedData[l].evts[n].offstime=estimatedData[l].evts[estimatedData[l].evts.size()-1].offstime;
						if(estimatedData[l].evts[n].offstime > maxOffstime){maxOffstime=estimatedData[l].evts[n].offstime;}
					}//endfor n
					continue;
				}//endif
				for(int n=chordOnsetPos[i];n<chordOnsetPos[i+1];n+=1){
					if(estimatedData[l].evts[chordOnsetPos[i+1]].onstime-estimatedData[l].evts[chordOnsetPos[i]].onstime>2*nBeat){
						estimatedData[l].evts[n].offstime=(estimatedData[l].evts[n].onstime/nBeat+1)*nBeat;
					}else{
						estimatedData[l].evts[n].offstime=estimatedData[l].evts[chordOnsetPos[i+1]].onstime;
					}//endif
					if(estimatedData[l].evts[n].offstime > maxOffstime){maxOffstime=estimatedData[l].evts[n].offstime;}
				}//endfor n
			}//endfor i

			//Add end meter event
			if(maxOffstime%nBeat!=0){
				maxOffstime=maxOffstime-maxOffstime%nBeat+nBeat;
			}//endif
			MeterEvt meterEvt(maxOffstime,0,0,0);
			estimatedData[l].meterEvts.push_back(meterEvt);

}//

			//Set spqn events
			estimatedData[l].spqnEvts[0].stime=0; estimatedData[l].spqnEvts[0].value=secPerQN[(optPath[0]/2)/nBeat];

			SPQNEvt spqnEvt;
			for(int n=1;n<len-1;n+=1){
				if((optPath[n]/2)/nBeat==(optPath[n-1]/2)/nBeat){continue;}
				spqnEvt.stime=estimatedData[l].evts[n].onstime; spqnEvt.value=secPerQN[(optPath[n]/2)/nBeat];
				estimatedData[l].spqnEvts.push_back(spqnEvt);
			}//endfor n

		}//endfor l

	}//end Transcribe


};//endclass PolyMetHMM


#endif // PolyMetHMM_HPP
