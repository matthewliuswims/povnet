#ifndef QuantizedPianoRoll_HPP
#define QuantizedPianoRoll_HPP

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
#include"BasicPitchCalculation_v170101.hpp"
#include"Fmt1x_v170108_2.hpp"
#include"Fmt3_v191003.hpp"
#include"Midi_v170101.hpp"
#include"PianoRoll_v170503.hpp"
#include"MusicXML_v190213.hpp"
using namespace std;

class QPREvt{
public:
	string ID;
	int onstime;
	int offstime;
	string sitch;//spelled pitch
	int pitch;//integral pitch
	int channel;//corresponding to staff
	int subvoice;//corresponding to voice in staff
	string label;//fmt1ID or -

	int onMetpos;
	int onFractime;//fractional time
//	int offMetpos;
	int epc;

	double ontime;
	double offtime;
	int onvel;
	int offvel;

	int barPos;//for SplitBars();
	int barNotePos;
	int tied;//0:not tied, 1:tied for SplitBarsAndNotes()

	vector<string> labs;
	vector<int> idxs;
	vector<double> vals;

	QPREvt(){
		ID="0";
		channel=0;
		subvoice=0;
		label="-";
		tied=0;
	}//end QPREvt
	~QPREvt(){}//end ~QPREvt

	void Print(){
cout<<ID<<"\t"<<onstime<<"\t"<<offstime<<"\t"<<sitch<<"\t"<<pitch<<"\t"<<channel<<"\t"<<subvoice<<"\t"<<label<<endl;
	}//end Print

};//end class QPREvt

class LessQPREvt{
public:
	bool operator()(const QPREvt& a, const QPREvt& b){
		if(a.onstime < b.onstime){
			return true;
		}else if(a.onstime==b.onstime){
			if(a.channel<b.channel){
				return true;
			}else if(a.channel==b.channel){
				if(a.subvoice<b.subvoice){
					return true;
				}else if(a.subvoice==b.subvoice){
					if(a.pitch<b.pitch){
						return true;
					}else{
						return false;
					}//endif
				}else{
					return false;
				}//endif
			}else{
				return false;
			}//endif
		}else{//if a.ontime > b.ontime
			return false;
		}//endif
	}//end operator()

};//end class LessQPREvt
//stable_sort(evts.begin(), evts.end(), LessQPREvt());

class InstrEvt{//
public:
	int channel;
	int programChange;
	string name;

	InstrEvt(){
		programChange=-1;
		name="\tNA";
	}//end InstrEvt
	~InstrEvt(){
	}//end ~InstrEvt
};//endclass InstrEvt

class KeyEvt{
public:
	KeyEvt(){
		stime=0;
		tonic="C";
		mode="major";
		keyfifth=0;
		tonic_int=0;
	}//end KeyEvt
	~KeyEvt(){
	}//end ~KeyEvt
	int stime;
	string tonic;
	string mode;//major, minor, etc.
	int keyfifth;//0=natural, 1=sharp, -1=flat, 2,-2,...
	int tonic_int;
};//endclass KeyEvt

class MeterEvt{
public:
	MeterEvt(){
		stime=0;
		num=4;
		den=4;
		barlen=48;
	}//end MeterEvt
	MeterEvt(int stime_,int num_,int den_,int barlen_){
		stime=stime_;
		num=num_;
		den=den_;
		barlen=barlen_;
	}//end MeterEvt
	MeterEvt(int tpqn){
		stime=0;
		num=4;
		den=4;
		barlen=4*tpqn;
	}//end MeterEvt
	~MeterEvt(){
	}//end ~MeterEvt
	int stime;
	int num;
	int den;
	int barlen;//bar length

	void SetBarlen(int tpqn){
		barlen=(num*4*tpqn)/den;
	}//end SetBarlen
};//endclass MeterEvt

class SPQNEvt{//sec per QN
public:
	SPQNEvt(){
		stime=0;
		value=0.5;
		bpm=120;
	}//end SPQNEvt
	~SPQNEvt(){
	}//end ~SPQNEvt
	int stime;
	double value;
	double bpm;
};//endclass SPQNEvt

class BarEvt{
public:
	BarEvt(){
	}//end BarEvt
	BarEvt(int stime_){
		stime=stime_;
	}//end BarEvt
	~BarEvt(){
	}//end ~BarEvt
	int stime;
	int barlen;//bar length
};//endclass BarEvt

class ClefEvt{
public:
	ClefEvt(){
	}//end ClefEvt
	ClefEvt(int stime_,int channel_,string clef_){
		stime=stime_;
		channel=channel_;
		clef=clef_;
	}//end ClefEvt
	~ClefEvt(){
	}//end ~ClefEvt
	int stime;
	int channel;
	string clef;//G2,F4,etc.
};//endclass ClefEvt

class QPRBar{
public:
	MeterEvt info;
	KeyEvt key;
	vector<QPREvt> notes;
};//endclass QPRBar

class ChordEvt{
public:
	int stime;
	int channel;
	string symb;//C,Am,G7,etc.

	ChordEvt(){
	}//end ChordEvt
	ChordEvt(int stime_,int channel_,string symb_){
		stime=stime_;
		channel=channel_;
		symb=symb_;
	}//end ChordEvt
	~ChordEvt(){
	}//end ~ChordEvt
};//endclass ChordEvt


class QuantizedPianoRoll{
public:
	int TPQN;
	vector<InstrEvt> instrEvts;
	vector<QPREvt> evts;
	vector<KeyEvt> keyEvts;
	vector<MeterEvt> meterEvts;
	vector<SPQNEvt> spqnEvts;
	vector<ClefEvt> clefEvts;
	vector<BarEvt> barEvts;
	vector<ChordEvt> chordEvts;
	vector<string> comments;//starting with //
	string name;
	string memo;
	int nBar;

	vector<QPRBar> bars;

	QuantizedPianoRoll(int TPQN_=4){
		TPQN=TPQN_;
		Init();
		SetDefault();
	}//end QuantizedPianoRoll
	QuantizedPianoRoll(string filename){
		ReadFile(filename,0);
	}//end QuantizedPianoRoll
	~QuantizedPianoRoll(){}//end ~QuantizedPianoRoll

	void Clear(){
		instrEvts.clear();
		evts.clear();
		keyEvts.clear();
		meterEvts.clear();
		spqnEvts.clear();
		clefEvts.clear();
		barEvts.clear();
		chordEvts.clear();
		comments.clear();
		name="";
		memo="";
		Init();
	}//end Clear

	void Init(){
		InstrEvt instrEvt;
		for(int i=0;i<100;i+=1){
			instrEvt.channel=i;
			instrEvts.push_back(instrEvt);
		}//endfor i
		nBar=-1;
	}//end Init

	void SetDefault(){
		keyEvts.push_back(KeyEvt());
		meterEvts.push_back(MeterEvt(TPQN));
		spqnEvts.push_back(SPQNEvt());
	}//end SetDefault

	void ReadFile(string filename,int filetype=0){//filetype=0(qpr)/1(qipr)
		vector<int> appearedChannel; appearedChannel.assign(100,0);//0(not appeared)/1(appeared);
		assert(filetype==0 || filetype==1);
		Clear();
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;
		QPREvt evt;
		ifstream ifs(filename.c_str());
		while(ifs>>s[0]){
			if(s[0][0]=='/'){
				getline(ifs,s[99]);
				ss.str("");
//				ss<<s[0]<<""<<s[99];
				ss<<s[99];
				comments.push_back(ss.str());
				continue;
			}else if(s[0][0]=='#'){
				ifs>>s[1];
				if(s[1]=="TPQN"){
					ifs>>TPQN;
				}else if(s[1]=="Name"){
					ifs>>name;
					getline(ifs,s[99]);
					name+=s[99];
					continue;
				}else if(s[1]=="Memo"){
					ifs>>memo;
					getline(ifs,s[99]);
					memo+=s[99];
					continue;
				}else if(s[1]=="nBar"){
					ifs>>nBar;
				}else if(s[1]=="Instr"){
					InstrEvt instrEvt;
					ifs>>instrEvt.channel>>instrEvt.programChange;
					getline(ifs,instrEvt.name);
					instrEvts[instrEvt.channel]=instrEvt;
					continue;
				}else if(s[1]=="Key"){
					KeyEvt keyEvt;
					ifs>>keyEvt.stime>>keyEvt.tonic>>keyEvt.mode>>keyEvt.keyfifth;
					keyEvt.tonic_int=SitchClassToPitchClass(keyEvt.tonic);
					keyEvts.push_back(keyEvt);
				}else if(s[1]=="Meter"){
					MeterEvt meterEvt;
					ifs>>meterEvt.stime>>meterEvt.num>>meterEvt.den>>meterEvt.barlen;
					meterEvts.push_back(meterEvt);
				}else if(s[1]=="SPQN"){
					SPQNEvt spqnEvt;
					ifs>>spqnEvt.stime;
					getline(ifs,s[99]);
					istringstream iss(s[99]);
					vector<double> vals;
					while(iss>>d[1]){vals.push_back(d[1]);}
					if(vals.size()==1){
						spqnEvt.value=vals[0];
						spqnEvt.bpm=60/vals[0];
					}else{
						spqnEvt.value=vals[0];
						spqnEvt.bpm=vals[1];
					}//endif
					spqnEvts.push_back(spqnEvt);
					continue;
				}else if(s[1]=="Bar"){
					BarEvt barEvt;
					ifs>>barEvt.stime>>barEvt.barlen;
					barEvts.push_back(barEvt);
				}else if(s[1]=="Clef"){
					ClefEvt clefEvt;
					ifs>>clefEvt.stime>>clefEvt.channel>>clefEvt.clef;
					clefEvts.push_back(clefEvt);
				}else if(s[1]=="Chord"){
					ChordEvt chordEvt;
					ifs>>chordEvt.stime>>chordEvt.channel>>chordEvt.symb;
					chordEvts.push_back(chordEvt);
				}//endif

				getline(ifs,s[99]); continue;
			}//endif
			evt.ID=s[0];
			ifs>>evt.onstime>>evt.offstime;
			if(filetype==0){
				ifs>>evt.sitch;
				evt.pitch=SitchToPitch(evt.sitch);
			}else{
				ifs>>evt.pitch;
				evt.sitch=PitchToSitch(evt.pitch);
			}//endif
			ifs>>evt.channel>>evt.subvoice>>evt.label;
			appearedChannel[evt.channel]=1;
			evts.push_back(evt);
			getline(ifs,s[99]);
		}//endwhile
		ifs.close();
		for(int i=0;i<appearedChannel.size();i+=1){
			if(appearedChannel[i]==0){continue;}
			if(instrEvts[i].programChange<0){
				instrEvts[i].programChange=0;
				instrEvts[i].name="\tDefault";
			}//endif
		}//endif
	}//end ReadFile

	void WriteFile(string filename,int filetype=0){//filetype=0(qpr)/1(qipr)/2(bar-wise)
		assert(filetype>=0 && filetype<=2);
if(filetype<=1){
		ofstream ofs(filename.c_str());
//		ofs<<"#Version: QuantizedPianoRoll_v190822"<<"\n";
		for(int i=0;i<comments.size();i+=1){
			ofs<<"// "<<comments[i]<<"\n";
		}//endfor i
		if(name.size()>0){
			ofs<<"# Name\t"<<name<<"\n";
		}//endif
		if(memo.size()>0){
			ofs<<"# Memo\t"<<memo<<"\n";
		}//endif
		if(nBar>0){
			ofs<<"# nBar\t"<<nBar<<"\n";
		}//endif
		ofs<<"# TPQN\t"<<TPQN<<"\n";
		for(int i=0;i<instrEvts.size();i+=1){
			if(instrEvts[i].programChange<0){continue;}
			ofs<<"# Instr\t"<<instrEvts[i].channel<<"\t"<<instrEvts[i].programChange<<""<<instrEvts[i].name<<"\n";
		}//endfor i
		for(int i=0;i<keyEvts.size();i+=1){
			ofs<<"# Key\t"<<keyEvts[i].stime<<"\t"<<keyEvts[i].tonic<<"\t"<<keyEvts[i].mode<<"\t"<<keyEvts[i].keyfifth<<"\n";
		}//endfor i
		for(int i=0;i<meterEvts.size();i+=1){
			ofs<<"# Meter\t"<<meterEvts[i].stime<<"\t"<<meterEvts[i].num<<"\t"<<meterEvts[i].den<<"\t"<<meterEvts[i].barlen<<"\n";
		}//endfor i
		for(int i=0;i<spqnEvts.size();i+=1){
			ofs<<"# SPQN\t"<<spqnEvts[i].stime<<"\t"<<spqnEvts[i].value<<"\t"<<spqnEvts[i].bpm<<"\n";
		}//endfor i
		for(int i=0;i<barEvts.size();i+=1){
			ofs<<"# Bar\t"<<barEvts[i].stime<<"\t"<<barEvts[i].barlen<<"\n";
		}//endfor i
		for(int i=0;i<clefEvts.size();i+=1){
			ofs<<"# Clef\t"<<clefEvts[i].stime<<"\t"<<clefEvts[i].channel<<"\t"<<clefEvts[i].clef<<"\n";
		}//endfor i
		for(int i=0;i<chordEvts.size();i+=1){
			ofs<<"# Chord\t"<<chordEvts[i].stime<<"\t"<<chordEvts[i].channel<<"\t"<<chordEvts[i].symb<<"\n";
		}//endfor i

		for(int n=0;n<evts.size();n+=1){
			QPREvt evt=evts[n];
			ofs<<evt.ID<<"\t"<<evt.onstime<<"\t"<<evt.offstime<<"\t";
			if(filetype==0){
				ofs<<evt.sitch<<"\t";
			}else{
				ofs<<evt.pitch<<"\t";
			}//endif
			ofs<<evt.channel<<"\t"<<evt.subvoice<<"\t"<<evt.label<<"\n";
		}//endfor n
		ofs.close();
}else if(filetype==2){
		SplitBars(48);
		ofstream ofs(filename.c_str());
		for(int i=0;i<comments.size();i+=1){
			ofs<<comments[i]<<"\n";
		}//endfor i
		ofs<<"# TPQN\t"<<TPQN<<"\n";
		for(int i=0;i<instrEvts.size();i+=1){
			if(instrEvts[i].programChange<0){continue;}
			ofs<<"# Instr\t"<<instrEvts[i].channel<<"\t"<<instrEvts[i].programChange<<""<<instrEvts[i].name<<"\n";
		}//endfor i
		for(int i=0;i<keyEvts.size();i+=1){
			ofs<<"# Key\t"<<keyEvts[i].stime<<"\t"<<keyEvts[i].tonic<<"\t"<<keyEvts[i].mode<<"\t"<<keyEvts[i].keyfifth<<"\n";
		}//endfor i
		for(int i=0;i<spqnEvts.size();i+=1){
			ofs<<"# SPQN\t"<<spqnEvts[i].stime<<"\t"<<spqnEvts[i].value<<"\t"<<spqnEvts[i].bpm<<"\n";
		}//endfor i

		for(int m=0;m<bars.size();m+=1){
			ofs<<bars[m].info.stime<<"\t"<<bars[m].info.barlen<<" ================================================== Bar "<<m<<"\n";
			for(int n=0;n<bars[m].notes.size();n+=1){
				QPREvt evt=bars[m].notes[n];
				ofs<<evt.ID<<"\t"<<evt.onstime<<"\t"<<evt.offstime<<"\t";
				ofs<<evt.sitch<<"\t";
				ofs<<evt.channel<<"\t"<<evt.subvoice<<"\t"<<evt.label<<"\n";
			}//endfor n
		}//endfor m
		ofs.close();
}//endif
	}//end WriteFile

	void ReadMusicXMLFile(string filename){
		Clear();

		if(filename.rfind("/")!=string::npos){
			name=filename.substr(filename.rfind("/")+1);
		}else{
			name=filename;
		}//endif

		Fmt1x fmt1x;
		fmt1x.ReadMusicXML(filename);
		fmt1x.Sort();
		Fmt3 fmt3;
		fmt3.ConvertFromFmt1x(fmt1x);
		TPQN=fmt3.TPQN;

		for(int i=0;i<fmt1x.comments.size();i+=1){
			if(fmt1x.comments[i].find("Instrument:")!=string::npos || fmt1x.comments[i].find("Part-name:")!=string::npos){
				comments.push_back(fmt1x.comments[i]);
			}//endif
		}//endfor i

		stringstream ss;
		string str;

		for(int n=0;n<fmt1x.evts.size();n+=1){

			if(n==0){
				BarEvt barEvt(fmt1x.evts[n].stime);
				barEvts.push_back(barEvt);
			}else if(fmt1x.evts[n].barnum!=fmt1x.evts[n-1].barnum){
				bool found=false;
				for(int i=0;i<barEvts.size();i+=1){
					if(barEvts[i].stime==fmt1x.evts[n].stime){found=true;}
				}//endfor i
				if(!found){
					BarEvt barEvt(fmt1x.evts[n].stime);
					barEvts.push_back(barEvt);
//cout<<barEvt.stime<<endl;
				}//endif
			}//endif

			if(fmt1x.evts[n].eventtype!="attributes"){continue;}

			bool found=false;
			for(int i=0;i<keyEvts.size();i+=1){
				if(keyEvts[i].stime==fmt1x.evts[n].stime){found=true;}
			}//endfor i
			if(found){continue;}

//cout<<fmt1x.evts[n].stime<<"\t"<<fmt1x.evts[n].info;

			ss.str(fmt1x.evts[n].info);
			ss>>str;
			KeyEvt keyEvt;
			keyEvt.stime=fmt1x.evts[n].stime;
			ss>>keyEvt.keyfifth>>keyEvt.mode;
			keyEvt.tonic=KeyFromKeySignature(keyEvt.keyfifth,keyEvt.mode);
			keyEvt.tonic=keyEvt.tonic.substr(0,keyEvt.tonic.find("m"));
			keyEvt.tonic_int=SitchClassToPitchClass(keyEvt.tonic);
			keyEvts.push_back(keyEvt);

			MeterEvt meterEvt;
			meterEvt.stime=fmt1x.evts[n].stime;
			ss>>meterEvt.num>>meterEvt.den;
			meterEvt.barlen=(4*TPQN*meterEvt.num)/meterEvt.den;
			meterEvts.push_back(meterEvt);

			int ip,ipp;
			ss>>ip;
			for(int i=0;i<ip;i+=1){
				ClefEvt clefEvt;
				clefEvt.stime=fmt1x.evts[n].stime;
				ss>>clefEvt.clef>>ipp;
				clefEvt.channel=i;
				clefEvts.push_back(clefEvt);
			}//endfor i

		}//endfor n

		vector<int> channelsForStaffs;//channelsForStaffs[0,1,...]=1,2,3...?
		vector<int> firstVoiceForStaffs;//firstVoiceForStaffs[0,1,...]=1,3,6,...?
		int maxVoice=0;

		for(int n=0;n<fmt3.evts.size();n+=1){
			channelsForStaffs.push_back(fmt3.evts[n].staff);
		}//endfor n
		sort(channelsForStaffs.begin(),channelsForStaffs.end());
		for(int i=channelsForStaffs.size()-1;i>=1;i-=1){
			if(channelsForStaffs[i]==channelsForStaffs[i-1]){channelsForStaffs.erase(channelsForStaffs.begin()+i);}
		}//endfor i
		firstVoiceForStaffs.assign(channelsForStaffs.size(),9999);

		for(int n=0;n<fmt3.evts.size();n+=1){
			int foundPos=-1;
			for(int i=0;i<channelsForStaffs.size();i+=1){
				if(channelsForStaffs[i]==fmt3.evts[n].staff){foundPos=i;break;}
			}//endfor i
			assert(foundPos>=0);
			if(fmt3.evts[n].voice<firstVoiceForStaffs[foundPos]){firstVoiceForStaffs[foundPos]=fmt3.evts[n].voice;}
			if(fmt3.evts[n].voice>maxVoice){maxVoice=fmt3.evts[n].voice;}
		}//endfor n

		vector<int> staffToChannel(100);
		for(int i=0;i<channelsForStaffs.size();i+=1){
			staffToChannel[channelsForStaffs[i]]=i;
		}//endfor i

		int lastOffstime=-1;
		QPREvt evt;
		for(int n=0;n<fmt3.evts.size();n+=1){
			if(fmt3.evts[n].dur==0){continue;}
			evt.onstime=fmt3.evts[n].stime;
			evt.offstime=fmt3.evts[n].stime+fmt3.evts[n].dur;
			if(evt.offstime>lastOffstime){lastOffstime=evt.offstime;}
			evt.channel=staffToChannel[fmt3.evts[n].staff];
			evt.subvoice=fmt3.evts[n].voice-firstVoiceForStaffs[staffToChannel[fmt3.evts[n].staff]];
			if(fmt3.evts[n].eventtype=="rest"){
				evt.sitch="R";
				evt.pitch=-1;
				evt.label="-";
				evts.push_back(evt);
			}else if(fmt3.evts[n].eventtype=="chord"){
				for(int i=0;i<fmt3.evts[n].sitches.size();i+=1){
					evt.sitch=fmt3.evts[n].sitches[i].substr(0,fmt3.evts[n].sitches[i].find(","));
					evt.pitch=SitchToPitch(evt.sitch);
					evt.label=fmt3.evts[n].fmt1IDs[i];
					evts.push_back(evt);
				}//endfor i
			}else{
				continue;
			}//endif

		}//endfor n

		vector<int> appearedChannel; appearedChannel.assign(100,0);//0(not appeared)/1(appeared);
		for(int n=0;n<evts.size();n+=1){
			appearedChannel[evts[n].channel]=1;
		}//endfor n
		for(int i=0;i<appearedChannel.size();i+=1){
			if(appearedChannel[i]==0){continue;}
			if(instrEvts[i].programChange<0){
				instrEvts[i].programChange=0;
				instrEvts[i].name="\tDefault";
			}//endif
		}//endif

		Sort();
		for(int n=0;n<evts.size();n+=1){
			ss.str(""); ss<<n;
			evts[n].ID=ss.str();
		}//endfor n

		for(int i=meterEvts.size()-1;i>=1;i-=1){
			if(meterEvts[i].num==meterEvts[i-1].num && meterEvts[i].den==meterEvts[i-1].den && meterEvts[i].barlen==meterEvts[i-1].barlen){
				meterEvts.erase(meterEvts.begin()+i);
			}//endif
		}//endfor i

		for(int i=keyEvts.size()-1;i>=1;i-=1){
			if(keyEvts[i].tonic==keyEvts[i-1].tonic && keyEvts[i].mode==keyEvts[i-1].mode && keyEvts[i].keyfifth==keyEvts[i-1].keyfifth){
				keyEvts.erase(keyEvts.begin()+i);
			}//endif
		}//endfor i

{
		BarEvt barEvt(lastOffstime);
		barEvt.barlen=TPQN;
		barEvts.push_back(barEvt);
}//
		for(int i=1;i<barEvts.size();i+=1){
			barEvts[i-1].barlen=barEvts[i].stime-barEvts[i-1].stime;
		}//endfor i

		vector<MeterEvt> meterEvts_;
		for(int i=0;i<barEvts.size()-1;i+=1){
			int jp;
			for(int j=0;j<meterEvts.size();j+=1){
				if(meterEvts[j].stime<=barEvts[i].stime){
					jp=j;
				}else{
					break;
				}//endif
			}//endfor j
			MeterEvt meterEvt(barEvts[i].stime,meterEvts[jp].num,meterEvts[jp].den,barEvts[i].barlen);
			meterEvts_.push_back(meterEvt);
		}//endfor i

		for(int i=meterEvts_.size()-1;i>=1;i-=1){
			if(meterEvts_[i].barlen==meterEvts_[i-1].barlen && meterEvts_[i].num==meterEvts_[i-1].num
			   && meterEvts_[i].den==meterEvts_[i-1].den){
				meterEvts_.erase(meterEvts_.begin()+i);
			}//endif
		}//endfor i

		meterEvts=meterEvts_;
// 		for(int i=0;i<meterEvts_.size();i+=1){
// cout<<meterEvts_[i].stime<<"\t"<<meterEvts_[i].num<<"\t"<<meterEvts_[i].den<<"\t"<<meterEvts_[i].barlen<<endl;
// 		}//endfor i

// 		for(int i=1;i<barEvts.size()-1;i+=1){
// 			int j,jp;
// 			for(j=1;j<meterEvts.size()-1;j+=1){
// 				if(meterEvts[j].stime>barEvts[i].stime){break;}
// 			}//endfor j
// 			j-=1;
// 			if( (barEvts[i].stime-meterEvts[j].stime)%meterEvts[j].barlen!=0 ){
// //cout<<barEvts[i].stime<<"\t"<<meterEvts[j].stime<<"\t"<<barEvts[i].barlen<<"\t"<<meterEvts[j].barlen<<endl;
// 				for(jp=1;jp<=j;jp+=1){
// 					if(meterEvts[jp].stime>barEvts[i-1].stime){break;}
// 				}//endfor j
// 				jp-=1;
// 				if(barEvts[i-1].stime!=meterEvts[jp].stime){
// 					meterEvts.insert(meterEvts.begin()+j,meterEvts[j]);
// 					meterEvts.insert(meterEvts.begin()+j,meterEvts[j]);
// 					meterEvts[j+1].stime=barEvts[i-1].stime;
// 					meterEvts[j+1].barlen=barEvts[i-1].barlen;
// 					meterEvts[j+2].stime=barEvts[i].stime;
// 				}else{
// 					meterEvts.insert(meterEvts.begin()+j,meterEvts[j]);
// 					meterEvts[j].barlen=(barEvts[i].stime-meterEvts[j].stime);
// 					meterEvts[j+1].stime=barEvts[i].stime;
// 				}//endif
// 			}//endif
// 		}//endfor i

{
		MeterEvt meterEvt(lastOffstime,0,0,0);//end bar
		meterEvts.push_back(meterEvt);
}//

		barEvts.clear();

/*
		for(int i=barEvts.size()-1;i>=1;i-=1){
			int tmpBarlen=meterEvts[0].barlen;
			for(int j=0;j<meterEvts.size();j+=1){
				if(meterEvts[j].stime<=barEvts[i-1].stime){tmpBarlen=meterEvts[j].barlen;
				}else{break;}
			}//endfor j
			if(barEvts[i].stime==barEvts[i-1].stime+tmpBarlen){
				barEvts.erase(barEvts.begin()+i);
			}//endif
		}//endfor i
*/

		for(int i=clefEvts.size()-1;i>=1;i-=1){
			for(int j=i-1;j>=0;j-=1){
				if(clefEvts[j].channel==clefEvts[i].channel){
					if(clefEvts[j].clef==clefEvts[i].clef){
						clefEvts.erase(clefEvts.begin()+i);
					}//endif
					break;
				}//endif
			}//endfor j
		}//endfor i

		//Tempo events


		//Read chord symbols and tempo events
		MusicXML musicxml;
		musicxml.ReadFile(filename);
		ChordEvt chordEvt;
		chordEvt.channel=0;

//cout<<musicxml.part.measures.size()<<endl;
		for(int m=0;m<musicxml.part.measures.size();m+=1){
			for(int i=0;i<musicxml.part.measures[m].types.size();i+=1){
				if(musicxml.part.measures[m].types[i]=="Harmony"){
	//cout<<m+1<<"\t"<<musicxml.part.measures[m].pos[i]<<"\t"<<musicxml.part.measures[m].stimes[i]<<"\t"<<musicxml.part.measures[m].harmonies[musicxml.part.measures[m].pos[i]].fullname<<endl;
					chordEvt.stime=musicxml.part.measures[m].stimes[i];
					chordEvt.symb=musicxml.part.measures[m].harmonies[musicxml.part.measures[m].pos[i]].fullname;
					chordEvts.push_back(chordEvt);
				}else if(musicxml.part.measures[m].types[i]=="TempoEvt"){

					SPQNEvt spqnEvt;
					spqnEvt.stime=musicxml.part.measures[m].stimes[i];
					spqnEvt.bpm=musicxml.part.measures[m].tempos[musicxml.part.measures[m].pos[i]].value;
					spqnEvt.value=60./spqnEvt.bpm;
					spqnEvts.push_back(spqnEvt);
//					musicxml.part.measures[m].stimes[i];
//					musicxml.part.measures[m].tempos[musicxml.part.measures[m].pos[i]].value;
				}//endif
			}//endfor i
		}//endfor m

	if(spqnEvts.size()==0){
		SPQNEvt spqnEvt;
		spqnEvts.push_back(spqnEvt);
	}else{
		spqnEvts[0].stime=0;
	}//endif

	}//end ReadMusicXMLFile

	void WriteMIDIFile(string filename,int outputType){//outputType=0(MIDI track=channel)/1(MIDI channel=voice)
		assert(outputType==0 || outputType==1);
		Sort();
// 		Midi midi;
// 		midi=ToMidi();
// 		midi.SetStrData();
		stringstream ss,sspart;
		int prevtick,dtick;

		int nTrack=1;//+1 for controlling track
		for(int i=0;i<16;i+=1){
			if(instrEvts[i].programChange<0){continue;}
			nTrack+=1;
		}//endfor i

//cout<<"nTrack:\t"<<nTrack<<endl;

		assert(nTrack<17);

		ss.str("");
		ss<<"MThd";
		ss<<fnum(6);//number of bytes
		ss<<ch(0)<<ch(1);//format type
		ss<<ch(0)<<ch(nTrack);//number of tracks
		ss<<ch(TPQN/(16*16))<<ch(TPQN%(16*16));

		vector<MidiMessage> midiMesList;
		int nBytes;

		for(int i=0;i<spqnEvts.size();i+=1){
			MidiMessage midiMes;
			midiMes.tick=spqnEvts[i].stime;
			midiMes.mes.push_back(255); midiMes.mes.push_back(81); midiMes.mes.push_back(3);
			tnumAdd(midiMes.mes,int(60./spqnEvts[i].value));
			midiMesList.push_back(midiMes);
		}//endfor i

		for(int i=0;i<meterEvts.size()-1;i+=1){
			MidiMessage midiMes;
			midiMes.tick=meterEvts[i].stime;
			midiMes.mes.push_back(255); midiMes.mes.push_back(88); midiMes.mes.push_back(4);
			if((meterEvts[i].num*TPQN*4)/meterEvts[i].den!=meterEvts[i].barlen){//incomplete bar
				midiMes.mes.push_back(meterEvts[i].barlen*4/TPQN);
				midiMes.mes.push_back(4);// /16
			}else{//complete bar
				midiMes.mes.push_back(meterEvts[i].num);
				if(meterEvts[i].den==1){midiMes.mes.push_back(0);
				}else if(meterEvts[i].den==2){midiMes.mes.push_back(1);
				}else if(meterEvts[i].den==4){midiMes.mes.push_back(2);
				}else if(meterEvts[i].den==8){midiMes.mes.push_back(3);
				}else if(meterEvts[i].den==16){midiMes.mes.push_back(4);
				}else if(meterEvts[i].den==32){midiMes.mes.push_back(5);
				}else{midiMes.mes.push_back(2);
				}//endif
			}//endif
			midiMes.mes.push_back(24); midiMes.mes.push_back(8);//MIDI clock, what's that??
			midiMesList.push_back(midiMes);
		}//endfor i

		for(int i=0;i<keyEvts.size();i+=1){
			MidiMessage midiMes;
			midiMes.tick=keyEvts[i].stime;
			midiMes.mes.push_back(255); midiMes.mes.push_back(89); midiMes.mes.push_back(2);
			midiMes.mes.push_back(keyEvts[i].keyfifth);//negative value (flats) OK??
			if(keyEvts[i].mode=="minor"){midiMes.mes.push_back(1);
			}else{midiMes.mes.push_back(0);
			}//endif
			midiMesList.push_back(midiMes);
		}//endfor i

		stable_sort(midiMesList.begin(), midiMesList.end(), LessTickMidiMessage());

		sspart.str("");
		prevtick=0;
		nBytes=0;
		for(int i=0;i<midiMesList.size();i+=1){
			dtick=midiMesList[i].tick-prevtick;
			sspart<<vnum(dtick);
			nBytes+=vnum2(dtick);
			for(int j=0;j<midiMesList[i].mes.size();j+=1){
				sspart<<ch(midiMesList[i].mes[j]);
				nBytes+=1;
			}//endfor j
			prevtick=midiMesList[i].tick;
		}//endfor i

		//Controlling track
		ss<<"MTrk";//Track 1 -> for controlling
		ss<<fnum(nBytes+4);//number of bytes
		ss<<sspart.str();
		ss<<ch(0)<<ch(255)<<ch(47)<<ch(0);//Track end

		//Instrument tracks
		for(int i=0;i<16;i+=1){
			if(instrEvts[i].programChange<0){continue;}

//cout<<"write channel:\t"<<i<<endl;

			midiMesList.clear();
			MidiMessage midiMes;
			for(int n=0;n<evts.size();n+=1){
				if(evts[n].channel!=i){continue;}
				if(evts[n].pitch<0){continue;}
				//note-on
				midiMes.tick=evts[n].onstime;
				midiMes.mes.resize(3);
				if(outputType==0){
					midiMes.mes[0]=144+i; midiMes.mes[1]=evts[n].pitch; midiMes.mes[2]=80;
				}else{
					midiMes.mes[0]=144+evts[n].subvoice; midiMes.mes[1]=evts[n].pitch; midiMes.mes[2]=80;
				}//endif
				midiMesList.push_back(midiMes);
				//note-off
				midiMes.tick=evts[n].offstime;
				midiMes.mes.resize(3);
				if(outputType==0){
					midiMes.mes[0]=128+i; midiMes.mes[1]=evts[n].pitch; midiMes.mes[2]=80;
				}else{
					midiMes.mes[0]=128+evts[n].subvoice; midiMes.mes[1]=evts[n].pitch; midiMes.mes[2]=80;
				}//endif
				midiMesList.push_back(midiMes);
			}//endfor n

			stable_sort(midiMesList.begin(), midiMesList.end(), LessTickMidiMessage());

			sspart.str("");
			prevtick=0;
			nBytes=0;
			sspart<<ch(0)<<ch(192+i)<<ch(instrEvts[i].programChange);
			nBytes+=3;
			for(int i=0;i<midiMesList.size();i+=1){
				dtick=midiMesList[i].tick-prevtick;
				sspart<<vnum(dtick);
				nBytes+=vnum2(dtick);
				for(int j=0;j<midiMesList[i].mes.size();j+=1){
					sspart<<ch(midiMesList[i].mes[j]);
					nBytes+=1;
				}//endfor j
				prevtick=midiMesList[i].tick;
			}//endfor i

			ss<<"MTrk";//Track 1
			ss<<fnum(nBytes+4);//number of bytes
			ss<<sspart.str();
			ss<<ch(0)<<ch(255)<<ch(47)<<ch(0);//Track end
		}//endfor i

		ofstream ofs;
		ofs.open(filename.c_str(), std::ios::binary);
		ofs<<ss.str();
		ofs.close();
	}//end WriteMIDIFile

	void SetOnMetPos(){
		for(int i=0;i<meterEvts.size()-1;i+=1){
			for(int tau=meterEvts[i].stime;tau<meterEvts[i+1].stime;tau+=meterEvts[i].barlen){
				for(int n=0;n<evts.size();n+=1){
					if(evts[n].onstime<tau){continue;}
					evts[n].onMetpos=evts[n].onstime-tau;
					if(evts[n].onstime>=tau+meterEvts[i].barlen){break;}
				}//endfor n
			}//endfor tau
		}//endfor i
	}//end SetOnMetPos

	void SplitBars(int resol){//resolution is used for setting onFractime = 48
		bars.clear();
		Sort();
		int lastStime=meterEvts[meterEvts.size()-1].stime;
		//Construct bars
		QPRBar bar;
		int curKeyEvtPos=0;
		for(int i=0;i<meterEvts.size()-1;i+=1){
			for(int tau=meterEvts[i].stime;tau<meterEvts[i+1].stime;tau+=meterEvts[i].barlen){
				bar.info=meterEvts[i];
				bar.info.stime=tau;
				if(curKeyEvtPos<keyEvts.size()-1){
					if(bar.info.stime>=keyEvts[curKeyEvtPos+1].stime){
						curKeyEvtPos+=1;
					}//endif
				}//endif
				bar.key=keyEvts[curKeyEvtPos];
				bar.notes.clear();
				bars.push_back(bar);
			}//endfor tau
		}//endfor i

		//Put notes
		int barpos=0;
		for(int n=0;n<evts.size();n+=1){
			if(evts[n].onstime>=bars[barpos].info.stime+bars[barpos].info.barlen){
				for(int m=barpos+1;m<bars.size();m+=1){
					if(evts[n].onstime<bars[m].info.stime+bars[m].info.barlen){
						barpos=m;
						break;
					}//endif
				}//endfor m
			}//endif

			evts[n].barPos=barpos;
			evts[n].onMetpos=evts[n].onstime-bars[barpos].info.stime;
			evts[n].onFractime=(((evts[n].onstime-bars[barpos].info.stime)*resol)/bars[barpos].info.barlen)%resol;
			evts[n].barNotePos=bars[barpos].notes.size();
			bars[barpos].notes.push_back(evts[n]);

		}//endfor n

	}//end SplitBars

	void SplitNotesAcrossBars(){
		for(int m=0;m<bars.size();m+=1){
			for(int l=0;l<bars[m].notes.size();l+=1){
				bars[m].notes[l].tied=0;
			}//endfor l
		}//endfor m
		for(int m=0;m<bars.size();m+=1){
			for(int l=0;l<bars[m].notes.size();l+=1){
				if(bars[m].notes[l].pitch<0){continue;}
				if(bars[m].notes[l].offstime<=bars[m].info.stime+bars[m].info.barlen){continue;}
				QPREvt evt=bars[m].notes[l];
				bars[m].notes[l].offstime=bars[m].info.stime+bars[m].info.barlen;
				evt.tied=1;
				evt.onstime=bars[m].info.stime+bars[m].info.barlen;
				evt.onMetpos=0;
				if(m<bars.size()-1){
					bars[m+1].notes.push_back(evt);
				}//endif
			}//endfor l
		}//endfor m
	}//end SplitNotesAcrossBars

	void RemoveRests(){
		for(int l=evts.size()-1;l>=0;l-=1){
			if(evts[l].pitch<0){evts.erase(evts.begin()+l);}
		}//endfor l
	}//end RemoveRests

	void ChangeTPQN(int newTPQN,int quantizationMode=0){//quantizationMode= 0:closest / 1:omit
		if(TPQN==newTPQN){return;}

//		SplitBars();
//		cout<<"WARNING: ChangeTPQN erases all notes that are not integral in the new TPQN"<<endl;

		for(int n=evts.size()-1;n>=0;n-=1){

			if((evts[n].onstime*newTPQN)%TPQN!=0){//onset stime is not integral with the new TPQN
				if(quantizationMode==1){
//cout<<"on\t"<<evts[n].ID<<endl;
					evts.erase(evts.begin()+n); continue;
				}else{
					evts[n].onstime=(evts[n].onstime*newTPQN)/TPQN + (((evts[n].onstime*newTPQN)%TPQN>TPQN/2)? 1:0);
				}//endif
			}else{//onset stime is integral with the new TPQN
				evts[n].onstime=(evts[n].onstime*newTPQN)/TPQN;
			}//endif

			if((evts[n].offstime*newTPQN)%TPQN!=0){//offset stime is not integral with the new TPQN
				if(quantizationMode==1){
//cout<<"off\t"<<evts[n].ID<<endl;
					evts.erase(evts.begin()+n); continue;
				}else{
					evts[n].offstime=(evts[n].offstime*newTPQN)/TPQN + (((evts[n].offstime*newTPQN)%TPQN>TPQN/2)? 1:0);
				}//endif
			}else{//offset stime is integral with the new TPQN
				evts[n].offstime=(evts[n].offstime*newTPQN)/TPQN;
			}//endif

		}//endfor n

		for(int i=keyEvts.size()-1;i>=0;i-=1){
//			if((keyEvts[i].stime*newTPQN)%TPQN!=0){keyEvts.erase(keyEvts.begin()+i); continue;}
			keyEvts[i].stime=(keyEvts[i].stime*newTPQN)/TPQN + (((keyEvts[i].stime*newTPQN)%TPQN>TPQN/2)? 1:0);
		}//endfor i
		for(int i=meterEvts.size()-2;i>=0;i-=1){
//			if((meterEvts[i].stime*newTPQN)%TPQN!=0){meterEvts.erase(meterEvts.begin()+i); continue;}
// 			meterEvts[i].stime=(meterEvts[i].stime*newTPQN)/TPQN;
// 			meterEvts[i].barlen=(meterEvts[i].barlen*newTPQN)/TPQN;
			meterEvts[i].stime=(meterEvts[i].stime*newTPQN)/TPQN + (((meterEvts[i].stime*newTPQN)%TPQN>TPQN/2)? 1:0);
			meterEvts[i].barlen=(meterEvts[i].barlen*newTPQN)/TPQN + (((meterEvts[i].barlen*newTPQN)%TPQN>TPQN/2)? 1:0);
		}//endfor i

//		meterEvts[meterEvts.size()-1].stime=(meterEvts[meterEvts.size()-1].stime*newTPQN)/TPQN;
		meterEvts[meterEvts.size()-1].stime=(meterEvts[meterEvts.size()-1].stime*newTPQN)/TPQN + (((meterEvts[meterEvts.size()-1].stime*newTPQN)%TPQN>TPQN/2)? 1:0);
		for(int i=spqnEvts.size()-1;i>=0;i-=1){
//			if((spqnEvts[i].stime*newTPQN)%TPQN!=0){spqnEvts.erase(spqnEvts.begin()+i); continue;}
			spqnEvts[i].stime=(spqnEvts[i].stime*newTPQN)/TPQN + (((spqnEvts[i].stime*newTPQN)%TPQN>TPQN/2)? 1:0);
		}//endfor i
		for(int i=chordEvts.size()-1;i>=0;i-=1){
			chordEvts[i].stime=(chordEvts[i].stime*newTPQN)/TPQN + (((chordEvts[i].stime*newTPQN)%TPQN>TPQN/2)? 1:0);
		}//endfor i

		TPQN=newTPQN;
	}//end ChangeTPQN

	PianoRoll ToPianoRoll(int outputType){//outputType=0(MIDI track=channel)/1(MIDI channel=voice)
		assert(outputType==0 || outputType==1);
		PianoRoll pr;
		PianoRollEvt prevt;
		stringstream ss;

		if(evts.size()==0){return pr;}

		vector<int> stimes;//all distinct stimes
		vector<double> times;//corresponding times (reflecting tempo changes)

		for(int n=0;n<evts.size();n+=1){
			if(find(stimes.begin(),stimes.end(),evts[n].onstime)==stimes.end()){
				stimes.push_back(evts[n].onstime);
			}//endif
			if(find(stimes.begin(),stimes.end(),evts[n].offstime)==stimes.end()){
				stimes.push_back(evts[n].offstime);
			}//endif
		}//endfor n
		for(int i=0;i<spqnEvts.size();i+=1){
			if(find(stimes.begin(),stimes.end(),spqnEvts[i].stime)==stimes.end()){
				stimes.push_back(spqnEvts[i].stime);
			}//endif
		}//endfor i
		sort(stimes.begin(),stimes.end());
		times.resize(stimes.size());

{
		double curSecPerTick=spqnEvts[0].value/double(TPQN);
		times[0]=stimes[0]*curSecPerTick;
		for(int i=1;i<stimes.size();i+=1){
			times[i]=times[i-1]+(stimes[i]-stimes[i-1])*curSecPerTick;
			for(int j=0;j<spqnEvts.size();j+=1){
				if(stimes[i]!=spqnEvts[j].stime){continue;}
				curSecPerTick=spqnEvts[j].value/double(TPQN);
				break;
			}//endfor j
		}//endfor i
}//

		for(int n=0;n<evts.size();n+=1){
			if(evts[n].sitch=="R"){continue;}
			ss.str(""); ss<<n;
			prevt.ID=ss.str();
			for(int i=0;i<stimes.size();i+=1){
				if(evts[n].onstime==stimes[i]){
					prevt.ontime=times[i];
				}//endif
				if(evts[n].offstime==stimes[i]){
					prevt.offtime=times[i];
					break;
				}//endif
			}//endfor i
			prevt.sitch=evts[n].sitch;
			prevt.pitch=evts[n].pitch;
			prevt.onvel=80;
			prevt.offvel=80;
			if(outputType==0){
				prevt.channel=evts[n].channel;
			}else{
				prevt.channel=evts[n].subvoice;
			}//endif
			prevt.endtime=prevt.offtime;
			pr.evts.push_back(prevt);
		}//endfor n

		return pr;
	}//end ToPianoRoll


	void ReadMIDIFile(string midiFile){
		Clear();
		SetDefault();
		Midi midi;
		midi.ReadFile(midiFile);
		stringstream ss;

		TPQN=midi.TPQN;

		int onPosition[16][128];
		for(int i=0;i<16;i+=1)for(int j=0;j<128;j+=1){onPosition[i][j]=-1;}//endfor i,j
		QPREvt evt;
		evt.subvoice=0;
		evt.label="-";
		int curChan;

		int endtime=-1;

		for(int n=0;n<midi.content.size();n+=1){

			if(midi.content[n].mes[0]>=192 && midi.content[n].mes[0]<208){//Program change event
				instrEvts[midi.content[n].mes[0]-192].channel=midi.content[n].mes[0]-192;
				instrEvts[midi.content[n].mes[0]-192].programChange=midi.content[n].mes[1];
				ss.str(""); ss<<"\tGM-"<<midi.content[n].mes[1];
				instrEvts[midi.content[n].mes[0]-192].name=ss.str();
			}//endif

			if(midi.content[n].mes[0]==255 && midi.content[n].mes[1]==89){//Key event
				KeyEvt keyEvt;
				keyEvt.stime=midi.content[n].tick;
				keyEvt.mode=((midi.content[n].mes[4]==1)? "minor":"major");
				if(midi.content[n].mes[3]>244){
					keyEvt.keyfifth=midi.content[n].mes[3]-256;
				}else{
					keyEvt.keyfifth=midi.content[n].mes[3]%12;
				}//endif
				keyEvt.tonic=KeyFromKeySignature(keyEvt.keyfifth,keyEvt.mode);
				ss.str("");
				if(keyEvt.tonic.size()==5){ss<<keyEvt.tonic[0]<<keyEvt.tonic[1];}else{ss<<keyEvt.tonic[0];}//endif
				keyEvt.tonic=ss.str();
				keyEvt.tonic_int=SitchClassToPitchClass(keyEvt.tonic);
				keyEvts.push_back(keyEvt);
			}//endif

			if(midi.content[n].mes[0]==255 && midi.content[n].mes[1]==88){//Meter event
				MeterEvt meterEvt;
				meterEvt.stime=midi.content[n].tick;
				meterEvt.num=midi.content[n].mes[3];
				meterEvt.den=1;
				for(int i=0;i<midi.content[n].mes[4];i+=1){meterEvt.den*=2;}
				meterEvt.barlen=(meterEvt.num*4*TPQN)/meterEvt.den;
				meterEvts.push_back(meterEvt);
			}//endif

			if(midi.content[n].mes[0]==255 && midi.content[n].mes[1]==81){//Tempo event
				SPQNEvt spqnEvt;
				spqnEvt.stime=midi.content[n].tick;
				spqnEvt.value=(midi.content[n].mes[3]*16*16*16*16+midi.content[n].mes[4]*16*16+midi.content[n].mes[5])/1000000.;
				spqnEvts.push_back(spqnEvt);
			}//endif

			if(midi.content[n].mes[0]>=128 && midi.content[n].mes[0]<160){//note-on or note-off event
				if(midi.content[n].tick>endtime){endtime=midi.content[n].tick;}
				curChan=midi.content[n].mes[0]%16;
				if(midi.content[n].mes[0]>=144 && midi.content[n].mes[2]>0){//note-on
					if(onPosition[curChan][midi.content[n].mes[1]]>=0){
// cout<<"Warning: (Double) note-on event before a note-off event "<<PitchToSitch(midi.content[n].mes[1])<<endl;
						evts[onPosition[curChan][midi.content[n].mes[1]]].offstime=midi.content[n].tick;
						evts[onPosition[curChan][midi.content[n].mes[1]]].offvel=-1;
					}//endif
					onPosition[curChan][midi.content[n].mes[1]]=evts.size();
					evt.channel=curChan;
					evt.pitch=midi.content[n].mes[1];
					evt.sitch=PitchToSitch(evt.pitch);
					evt.onvel=midi.content[n].mes[2];
					evt.offvel=0;
					evt.onstime=midi.content[n].tick;
					evt.offstime=evt.onstime+1;
					evts.push_back(evt);
				}else{//note-off
					if(onPosition[curChan][midi.content[n].mes[1]]<0){
// cout<<"Warning: Note-off event before a note-on event "<<PitchToSitch(midi.content[n].mes[1])<<endl;
// cout<<midi.content[n].time<<endl;
						continue;
					}//endif
					evts[onPosition[curChan][midi.content[n].mes[1]]].offstime=midi.content[n].tick;
					if(midi.content[n].mes[2]>0){
						evts[onPosition[curChan][midi.content[n].mes[1]]].offvel=midi.content[n].mes[2];
					}else{
						evts[onPosition[curChan][midi.content[n].mes[1]]].offvel=80;
					}//endif
					onPosition[curChan][midi.content[n].mes[1]]=-1;
				}//endif
			}//endif
		}//endfor n
		for(int i=0;i<16;i+=1)for(int j=0;j<128;j+=1){
			if(onPosition[i][j]>=0){
cout<<"Warning: Note without a note-off event"<<endl;
cout<<"onstime channel sitch : "<<evts[onPosition[i][j]].onstime<<"\t"<<evts[onPosition[i][j]].channel<<"\t"<<evts[onPosition[i][j]].sitch<<endl;
//				return;
			}//endif
		}//endfor i,j

		//Endtime event
{
		MeterEvt meterEvt;
		meterEvt.stime=endtime;
		meterEvt.num=0;
		meterEvt.den=0;
		meterEvt.barlen=0;
		meterEvts.push_back(meterEvt);
}//

		if(keyEvts.size()>1){keyEvts.erase(keyEvts.begin());}
		if(meterEvts.size()>1){meterEvts.erase(meterEvts.begin());}
		if(spqnEvts.size()>1){spqnEvts.erase(spqnEvts.begin());}

		for(int n=0;n<evts.size();n+=1){
			stringstream ss;
			ss.str(""); ss<<n;
			evts[n].ID=ss.str();
		}//endfor n

	}//end ReadMIDIFile

	void SelectChannel(int channel_){
		assert(channel_>=0 && channel_<instrEvts.size());

		for(int i=0;i<instrEvts.size();i+=1){
			if(instrEvts[i].channel==channel_){continue;}
			InstrEvt instrEvt;
			instrEvt.channel=i;
			instrEvts[i]=instrEvt;
		}//endfor i

		for(int n=evts.size()-1;n>=0;n-=1){
			if(evts[n].channel!=channel_){
				evts.erase(evts.begin()+n);
			}//endif
		}//endfor n

	}//end SelectChannel

	void Sort(){
		stable_sort(evts.begin(), evts.end(), LessQPREvt());
	}//end Sort

	void FilterChannelAndSubvoice(int channel_=-1,int subvoice_=-1){
		for(int n=evts.size()-1;n>=0;n-=1){
			if(channel_>=0 && evts[n].channel!=channel_){
				evts.erase(evts.begin()+n);
				continue;
			}//endif
			if(subvoice_>=0 && evts[n].subvoice!=subvoice_){
				evts.erase(evts.begin()+n);
				continue;
			}//endif
		}//endfor n
	}//end FilterChannelAndSubvoice

	void SetEpcBarwise(int nEpc=36){
		if(bars.size()==0){return;}
		int curEpc=-1;//q:0~nEpc-1; q':-nEpc~-1 = rest after epc q
		for(int m=0;m<bars.size();m+=1){
			for(int n=0;n<bars[m].notes.size();n+=1){
				if(bars[m].notes[n].pitch<0){continue;}
				curEpc=bars[m].notes[n].pitch%12;
				break;
			}//endfor n
			if(curEpc>=0){break;}
		}//endfor m
		if(curEpc<0){curEpc=0;}
		int prePitch=-1;
		int intvl;
		for(int m=0;m<bars.size();m+=1){
			for(int n=0;n<bars[m].notes.size();n+=1){
				if(bars[m].notes[n].pitch<0){
					bars[m].notes[n].epc=curEpc-nEpc;
				}else if(prePitch<0){
					bars[m].notes[n].epc=curEpc;
					prePitch=bars[m].notes[n].pitch;
				}else{
					intvl=bars[m].notes[n].pitch-prePitch;
					if(intvl>nEpc/2){intvl-=12*((intvl-nEpc/2+11)/12);}
					if(intvl<=-nEpc/2){intvl+=12*(-(intvl+nEpc/2)/12+1);}
					bars[m].notes[n].epc=(curEpc+intvl+nEpc)%nEpc;
 					curEpc=bars[m].notes[n].epc;
					prePitch=bars[m].notes[n].pitch;
				}//endif
			}//endfor n
		}//endfor m
	}//end SetEpcBarwise


};//end class QuantizedPianoRoll

#endif // QuantizedPianoRoll_HPP

