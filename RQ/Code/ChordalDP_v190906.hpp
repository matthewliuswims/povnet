#ifndef ChordalDP_HPP
#define ChordalDP_HPP

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
#include"QuantizedPianoRoll_v191003.hpp"
#include"BasicCalculation_v170122.hpp"

using namespace std;

class MorePitchQPREvt{
public:
	bool operator()(const QPREvt& a, const QPREvt& b){
		if(a.pitch > b.pitch){
			return true;
		}else{
			return false;
		}//endif
	}//end operator()
};//endclass MorePitchQPREvt
//stable_sort(evts.begin(), evts.end(), MorePitchQPREvt());

class ChordalDP{
public:

	QuantizedPianoRoll qpr;
	int limitNumVoice;//10

	ChordalDP(){
		limitNumVoice=10;
		Init();
	}//end ChordalDP
	~ChordalDP(){}//end ~ChordalDP

	void Init(){
	}//end Init

	void Clear(){
	}//end Clear

	void DP(int hand,int maxNumVoice){//hand=0(RH)/1(LH)

		if(!(hand==0 || hand==1)){return;}
		assert(maxNumVoice>0 && maxNumVoice<=limitNumVoice);

		vector<int> pos;
		vector<QPREvt> subQpr;//notes in the selected hand
		for(int n=0;n<qpr.evts.size();n+=1){
			if(qpr.evts[n].channel!=hand){continue;}
			pos.push_back(n);
			qpr.evts[n].idxs.push_back(n);
			qpr.evts[n].idxs.push_back(0);
			subQpr.push_back(qpr.evts[n]);
		}//endfor n

		if(subQpr.size()==0){return;}

		vector<vector<QPREvt> > clusters;
{
		vector<QPREvt> cluster;
		subQpr[0].idxs[1]=0;
		cluster.push_back(subQpr[0]);
		for(int n=1;n<subQpr.size();n+=1){
			if(subQpr[n].onstime!=subQpr[n-1].onstime){
				for(int np=n-1;np>=0;np-=1){
					if(subQpr[np].onstime==subQpr[n-1].onstime){continue;}
					if(subQpr[np].offstime>subQpr[n-1].onstime){
						subQpr[np].idxs[1]=1;
						cluster.push_back(subQpr[np]);
					}//endif
				}//endfor np
				clusters.push_back(cluster);
				cluster.clear();
			}//endif
			subQpr[n].idxs[1]=0;
			cluster.push_back(subQpr[n]);
		}//endfor n
		for(int np=subQpr.size()-1;np>=0;np-=1){
			if(subQpr[np].onstime==subQpr[subQpr.size()-1].onstime){continue;}
			if(subQpr[np].offstime>subQpr[subQpr.size()-1].onstime){
				subQpr[np].idxs[1]=1;
				cluster.push_back(subQpr[np]);
			}//endif
		}//endfor np
		clusters.push_back(cluster);
}//


		for(int i=0;i<clusters.size();i+=1){
			stable_sort(clusters[i].begin(), clusters[i].end(), MorePitchQPREvt());
		}//endfor i

		vector<vector<vector<int> > > states(clusters.size());//[i][k]=(voice association)
{
		vector<vector<int> > vvi;
		for(int i=0;i<clusters.size();i+=1){
			int allnum=1;
			for(int j=0;j<clusters[i].size();j+=1){allnum*=maxNumVoice;}
			vvi.clear();
			for(int k=0;k<allnum;k+=1){
				int kp=k;
				vector<int> vi(clusters[i].size());
				for(int j=0;j<clusters[i].size();j+=1){
					vi[j]=kp%maxNumVoice;
					kp/=maxNumVoice;
				}//endfor j
				vvi.push_back(vi);
			}//endfor k
			states[i]=vvi;
		}//endfor i
}//

		vector<double> CC;//cumulative cost
		vector<vector<int> > amin(clusters.size());

		for(int i=0;i<clusters.size();i+=1){
			vector<double> preCC(CC);
			CC.clear();
			CC.resize(states[i].size());
			amin[i].resize(states[i].size());
			double cost;

			if(i==0){
				for(int k=0;k<states[i].size();k+=1){
					CC[k]=HorizontalCost(states[i][k],clusters[i]);
				}//endfor k
				continue;
			}//endif

			for(int k=0;k<states[i].size();k+=1){
				CC[k]=preCC[0]+VerticalCost(states[i-1][0],clusters[i-1],states[i][k],clusters[i]);
				amin[i][k]=0;
				for(int kp=0;kp<states[i-1].size();kp+=1){
					cost=preCC[kp]+VerticalCost(states[i-1][kp],clusters[i-1],states[i][k],clusters[i]);
					if(cost<CC[k]){
						CC[k]=cost;
						amin[i][k]=kp;
					}//endif
				}//endfor kp
				CC[k]+=HorizontalCost(states[i][k],clusters[i]);
			}//endfor k

		}//endfor i

		//Backward
		vector<int> optPath(clusters.size());
		optPath[optPath.size()-1]=0;
		for(int k=0;k<CC.size();k+=1){
			if(CC[k]<CC[optPath[optPath.size()-1]]){optPath[optPath.size()-1]=k;}
		}//endfor k
		for(int i=optPath.size()-2;i>=0;i-=1){
			optPath[i]=amin[i+1][optPath[i+1]];
		}//endfor i

		//Set voice
		for(int i=0;i<clusters.size();i+=1){
			for(int j=0;j<clusters[i].size();j+=1){
				if(clusters[i][j].idxs[1]!=0){continue;}
				qpr.evts[clusters[i][j].idxs[0]].subvoice=states[i][optPath[i]][j];
			}//endfor j
		}//endfor i

		//Set offset stime
		//offset stimes of simultaneous notes in one voice must match and must not be larger than the next onset stime of that voice
		for(int k=0;k<maxNumVoice;k+=1){
			int n=0;
			while(true){
				if(n>qpr.evts.size()-1){break;}
				for(int np=n;np<qpr.evts.size();np+=1){
					if(qpr.evts[np].channel==hand && qpr.evts[np].subvoice==k){
						n=np;
						break;
					}//endif
				}//endfor np
				vector<int> sameOnsetNotePos;
				sameOnsetNotePos.push_back(n);
				int nextOnsetStime=-1;
				int nextOnsetPos=-1;
				for(int np=n;np<qpr.evts.size();np+=1){
					if(qpr.evts[np].channel==hand && qpr.evts[np].subvoice==k && qpr.evts[np].onstime==qpr.evts[n].onstime){
						sameOnsetNotePos.push_back(np);
					}//endif
					if(qpr.evts[np].channel==hand && qpr.evts[np].subvoice==k && qpr.evts[np].onstime>qpr.evts[n].onstime){
						nextOnsetStime=qpr.evts[np].onstime;
						nextOnsetPos=np;
						break;
					}//endif
				}//endfor np
				int latestOffstime=qpr.evts[n].offstime;
				for(int j=0;j<sameOnsetNotePos.size();j+=1){
					if(qpr.evts[sameOnsetNotePos[j]].offstime>latestOffstime){latestOffstime=qpr.evts[sameOnsetNotePos[j]].offstime;}
				}//endfor j
				if(nextOnsetStime<0){
					for(int j=0;j<sameOnsetNotePos.size();j+=1){qpr.evts[sameOnsetNotePos[j]].offstime=latestOffstime;}//endfor j
				}else if(latestOffstime<=nextOnsetStime){
					for(int j=0;j<sameOnsetNotePos.size();j+=1){qpr.evts[sameOnsetNotePos[j]].offstime=latestOffstime;}//endfor j
				}else{
					for(int j=0;j<sameOnsetNotePos.size();j+=1){qpr.evts[sameOnsetNotePos[j]].offstime=nextOnsetStime;}//endfor j
				}//endif
				if(nextOnsetPos<0){break;}
				n=nextOnsetPos;
			}//endwhile
		}//endfor k

// 		int maxOffstime=-1;
// 		for(int n=qpr.evts.size()-1;n>=0;n-=1){
// 			if(qpr.evts[n].offstime>maxOffstime){maxOffstime=qpr.evts[n].offstime;}
// 		}//endfor n
// 		for(int k=0;k<maxNumVoice;k+=1){
// 			QPREvt evt;
// 			evt.ID="-1";
// 			evt.onstime=maxOffstime-maxOffstime%qpr.TPQN+3*4*qpr.TPQN;
// 			evt.offstime=evt.onstime+qpr.TPQN;
// 			evt.channel=hand;
// 			evt.subvoice=k;
// 			evt.label="-";
// 			if(hand==0){
// 				evt.sitch="C6";
// 				evt.pitch=SitchToPitch(evt.sitch);
// 			}else{
// 				evt.sitch="C2";
// 				evt.pitch=SitchToPitch(evt.sitch);
// 			}//endif
// 			qpr.evts.push_back(evt);
// 		}//endfor k

		int maxOffstime=qpr.meterEvts[qpr.meterEvts.size()-1].stime;
		MeterEvt meterEvt(maxOffstime,4,4,qpr.TPQN*4);
		qpr.meterEvts.insert(qpr.meterEvts.begin()+qpr.meterEvts.size()-1,meterEvt);
		qpr.meterEvts[qpr.meterEvts.size()-1].stime=maxOffstime+qpr.TPQN*4;

		for(int k=0;k<maxNumVoice;k+=1){
			QPREvt evt;
			evt.ID="-1";
			evt.onstime=maxOffstime;
			evt.offstime=evt.onstime+qpr.TPQN;
			evt.channel=hand;
			evt.subvoice=k;
			evt.label="-";
			if(hand==0){
				evt.sitch="C6";
				evt.pitch=SitchToPitch(evt.sitch);
			}else{
				evt.sitch="C2";
				evt.pitch=SitchToPitch(evt.sitch);
			}//endif
			qpr.evts.push_back(evt);
		}//endfor k

	}//end DP

	double HorizontalCost(vector<int> &voices,vector<QPREvt> &notes){
		assert(voices.size()==notes.size());
		double cost=0;
		for(int n=0;n<voices.size();n+=1){
			//Note containing cost
			cost+=1.*voices[n];
			for(int np=0;np<voices.size();np+=1){
				if(np<=n){continue;}
				if(voices[n]>voices[np] && notes[n].pitch!=notes[np].pitch){//voice crossing
					cost+=3.;
				}//endif
				if(voices[n]==voices[np] && notes[n].offstime!=notes[np].offstime){//unmatched offset times in one voice
					cost+=1.;
				}//endif
				if(voices[n]==voices[np] && notes[n].idxs[1]!=notes[np].idxs[1]){//overlapped onset and offset times in one voice
					cost+=1.;
				}//endif
			}//endfor np
		}//endfor n
		return cost;
	}//end HorizontalCost

	double VerticalCost(vector<int> &prevoices,vector<QPREvt> &prenotes,vector<int> &voices,vector<QPREvt> &notes){
		assert(prevoices.size()==prenotes.size());
		assert(voices.size()==notes.size());
		double cost=0;

		for(int n=0;n<voices.size();n+=1){
			for(int np=0;np<prevoices.size();np+=1){

				if(prevoices[np]==voices[n]){//Preference for pitch proximity in one voice
					cost+=0.*pow((prenotes[np].pitch-notes[n].pitch)/7.,2.);
				}else{
					cost+=0.*pow((prenotes[np].pitch-notes[n].pitch)/12.,2.);
				}//endif

				if(notes[n].idxs[1]==1 && prenotes[np].idxs[0]==notes[n].idxs[0]){//Sustained note
					if(prevoices[np]!=voices[n]){//Sustained note constraint
						cost+=5.;
					}//endif
					break;
				}//endif

				if(notes[n].idxs[1]==0 && prevoices[np]==voices[n]){
					if(prenotes[np].offstime==notes[n].onstime){//no gap

					}else if(prenotes[np].offstime<notes[n].onstime){//gap
						cost+=0.2;
					}else if(prenotes[np].offstime>notes[n].onstime){//overlap
						cost+=1.;
					}//endif
				}//endif

			}//endfor np
		}//endfor n

		return cost;
	}//end VerticalCost

};//endclass ChordalDP

#endif // ChordalDP_HPP

