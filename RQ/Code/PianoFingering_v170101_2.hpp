#ifndef PAINOFINGERING_HPP
#define PAINOFINGERING_HPP

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
#include"PianoRoll_v170503.hpp"
#include"BasicCalculation_v170122.hpp"

using namespace std;

inline string GetKeyPressFingerNum(string fingerNum){
	if(fingerNum.find("_")==string::npos){
		return fingerNum;
	}else{
		return fingerNum.substr(0,fingerNum.find("_"));
	}//endif
}//end GetKeyPressFingerNum

class PianoFingeringEvt : public PianoRollEvt{
public:

	string fingerNum;//Right hand: 1,...,5. Left hand: -1,...,-5. Finger substitution: 2_1 etc.
	int finger;//1,...,5,-1,...,-5 not main information
	int ext1;

	PianoFingeringEvt(){}//end PianoFingeringEvt
	PianoFingeringEvt(PianoRollEvt prEvt){
		ID=prEvt.ID;
		ontime=prEvt.ontime;
		offtime=prEvt.offtime;
		sitch=prEvt.sitch;
		pitch=prEvt.pitch;
		onvel=prEvt.onvel;
		offvel=prEvt.offvel;
		channel=prEvt.channel;
		fingerNum="1";
	}//end PianoFingeringEvt
	~PianoFingeringEvt(){}//end ~PianoFingeringEvt

};//end class PianoFingeringEvt

class LessPitchPianoFingeringEvt{
public:
	bool operator()(const PianoFingeringEvt& a, const PianoFingeringEvt& b){
		if(a.pitch < b.pitch){
			return true;
		}else{//if
			return false;
		}//endif
	}//end operator()
};//end class LessPitchPianoFingeringEvt
//stable_sort(evts.begin(), evts.end(), LessPitchPianoFingeringEvt());

class PianoFingering{
public:

	vector<string> comments;
	vector<PianoFingeringEvt> evts;
	string pieceName;
	string performanceName;
	string fingeringName;

	PianoFingering(){}//end PianoFingering
	PianoFingering(PianoRoll pr){
		evts.clear();
		for(int n=0;n<pr.evts.size();n+=1){
			evts.push_back(PianoFingeringEvt(pr.evts[n]));
		}//endfor n
	}//end PianoFingering
	~PianoFingering(){}//end ~PianoFingering

	void Clear(){
		comments.clear();
		evts.clear();
		pieceName="";
		performanceName="";
		fingeringName="";
	}//end Clear

	void ReadFile(string filename){//We use spr for piano fingering
		Clear();
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		string version="";
		stringstream ss;
		PianoFingeringEvt evt;
		ifstream ifs(filename.c_str());
		while(ifs>>s[0]){
			if(s[0][0]=='/'){
				if(s[0]=="//Version:"){
					ifs>>version;
					getline(ifs,s[99]);
					continue;
				}else if(s[0]=="//Piece:"){
					ifs>>pieceName;
					getline(ifs,s[99]);
					continue;
				}else if(s[0]=="//Performance:"){
					ifs>>performanceName;
					getline(ifs,s[99]);
					continue;
				}else if(s[0]=="//Fingering:"){
					ifs>>fingeringName;
					getline(ifs,s[99]);
					continue;
				}//endif
				getline(ifs,s[99]);
				ss.str("");
				ss<<s[0]<<s[99];
				comments.push_back(ss.str());
				continue;
			}else if(s[0][0]=='#'){
				getline(ifs,s[99]);
				ss.str("");
				ss<<s[0]<<s[99];
				comments.push_back(ss.str());
				continue;
			}//endif
			evt.ID=s[0];
			ifs>>evt.ontime>>evt.offtime>>evt.sitch>>evt.onvel>>evt.offvel>>evt.channel>>evt.fingerNum;
			evt.pitch=SitchToPitch(evt.sitch);
			evts.push_back(evt);
			getline(ifs,s[99]);
		}//endwhile
		ifs.close();

		if(version!="PianoFingering_v170101"){
//			cout<<"WARNING: File version previous one or unknown: "<<version<<endl;
		}//endif

	}//end ReadFile

	void WriteFile(string filename){
		ofstream ofs(filename.c_str());
		ofs<<"//Version: PianoFingering_v170101\n";
		if(pieceName!=""){ofs<<"//Piece: "<<pieceName<<"\n";}//endif
		if(performanceName!=""){ofs<<"//Performance: "<<performanceName<<"\n";}//endif
		if(fingeringName!=""){ofs<<"//Fingering: "<<fingeringName<<"\n";}//endif
		for(int i=0;i<comments.size();i+=1){
			ofs<<comments[i]<<endl;
		}//endfor i
		ofs.precision(10);
		for(int n=0;n<evts.size();n+=1){
			PianoFingeringEvt evt=evts[n];
			ofs<<evt.ID<<"\t"<<fixed<<setprecision(6)<<evt.ontime<<"\t"<<evt.offtime<<"\t"<<evt.sitch<<"\t"<<evt.onvel<<"\t"<<evt.offvel<<"\t"<<evt.channel<<"\t"<<evt.fingerNum<<"\n";
		}//endfor n
		ofs.close();
	}//end WriteFile

	void SetChannel(int chan){//chan= 0:Right Hand, 1:Left Hand
		for(int n=0;n<evts.size();n+=1){
		evts[n].channel=chan;
		}//endfor n
	}//end SetChannel

	void SetChannelFromFingerNumber(){//0:RH 1:LH
		for(int n=0;n<evts.size();n+=1){
			if(evts[n].fingerNum[0]=='-'){evts[n].channel=1;
			}else{evts[n].channel=0;
			}//endif
		}//endfor n
	}//end

	void SelectHandByChannel(int hand){//0:RH 1:LH
		if(hand==0){
			for(int n=evts.size()-1;n>=0;n-=1){
				evts[n].ext1=n;
				if(evts[n].channel==1){evts.erase(evts.begin()+n);}
			}//endfor n
		}else{
			for(int n=evts.size()-1;n>=0;n-=1){
				evts[n].ext1=n;
				if(evts[n].channel==0){evts.erase(evts.begin()+n);}
			}//endfor n
		}//endif
	}//end SelectHandByChannel

	void SelectHandByFingerNum(int hand){//0:RH 1:LH
		if(hand==0){
			for(int n=evts.size()-1;n>=0;n-=1){
				evts[n].ext1=n;
				if(evts[n].fingerNum[0]=='-'){evts.erase(evts.begin()+n);}
			}//endfor n
		}else{
			for(int n=evts.size()-1;n>=0;n-=1){
				evts[n].ext1=n;
				if(evts[n].fingerNum[0]!='-'){evts.erase(evts.begin()+n);}
			}//endfor n
		}//endif
	}//end SelectHandByFingerNum

	void ConvertFingerNumberToInt(){
		for(int n=0;n<evts.size();n+=1){
			if(evts[n].fingerNum[0]=='-'){
				if(evts[n].fingerNum[1]=='1'){evts[n].finger=-1;
				}else if(evts[n].fingerNum[1]=='2'){evts[n].finger=-2;
				}else if(evts[n].fingerNum[1]=='3'){evts[n].finger=-3;
				}else if(evts[n].fingerNum[1]=='4'){evts[n].finger=-4;
				}else{evts[n].finger=-5;
				}//endif
			}else{
				if(evts[n].fingerNum[0]=='1'){evts[n].finger=1;
				}else if(evts[n].fingerNum[0]=='2'){evts[n].finger=2;
				}else if(evts[n].fingerNum[0]=='3'){evts[n].finger=3;
				}else if(evts[n].fingerNum[0]=='4'){evts[n].finger=4;
				}else{evts[n].finger=5;
				}//endif
			}//endif
		}//endfor n
	}//end ConvertFingerNumberToInt

	void TimeDepPitchOrder(){
		vector<vector<int> > posCluster;
{
		vector<int> vi;
		vi.push_back(0);
		for(int n=1;n<evts.size();n+=1){
			if(abs(evts[n].ontime-evts[n-1].ontime)>=0.03){
				posCluster.push_back(vi);
				vi.clear();
			}//endif
			vi.push_back(n);
		}//endfor n
		posCluster.push_back(vi);
}//

		vector<PianoFingeringEvt> pre_evts;
		for(int ii=0;ii<posCluster.size();ii+=1){
			if(posCluster[ii].size()==0){
				continue;
			}else if(posCluster[ii].size()==1){
				pre_evts.push_back(evts[posCluster[ii][0]]);
				continue;
			}//endif
			vector<Pair> pairs;
			Pair pair;
			for(int i=0;i<posCluster[ii].size();i+=1){
				pair.ID=i;
				pair.value=-evts[posCluster[ii][i]].pitch;
				pairs.push_back(pair);
			}//endfor i
			stable_sort(pairs.begin(),pairs.end(),MorePair());
			for(int i=0;i<posCluster[ii].size();i+=1){
				pre_evts.push_back(evts[posCluster[ii][pairs[i].ID]]);
			}//endfor i			
		}//endfor ii
		assert(pre_evts.size()==evts.size());
		evts=pre_evts;
	}//end TimeDepPitchOrder

	void ExtractKeyPressFingers(){
		for(int n=0;n<evts.size();n+=1){
			evts[n].fingerNum=GetKeyPressFingerNum(evts[n].fingerNum);
		}//endfor n
	}//end ExtractKeyPressFingers

};//endclass PianoFingering

inline vector<double> CompareFingerings(PianoFingering fingering1,PianoFingering fingering2){
	assert(fingering1.evts.size()==fingering2.evts.size());
	vector<double> vd(2);
	vd[0]=0;
	vd[1]=0;
	for(int n=0;n<fingering1.evts.size();n+=1){
		vd[0]+=1;
		if(fingering1.evts[n].fingerNum==fingering2.evts[n].fingerNum){continue;}//endif
		vd[1]+=1;
	}//endfor n
	vd[1]=(vd[0]-vd[1])/vd[0]*100;
	return vd;//nNotes,matchRate
}//end CompareFingerings


#endif // PAINOFINGERING_HPP

