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
#include"Fmt3x_v170225.hpp"
#include"Midi_v170101.hpp"
#include"PianoRoll_v170503.hpp"
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
//	int offMetpos;

	double ontime;
	double offtime;
	int onvel;
	int offvel;

	vector<string> labs;
	vector<int> idxs;
	vector<double> vals;

	QPREvt(){}//end QPREvt
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
				if(a.pitch<b.pitch){
					return true;
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
		name="NA";
	}//end InstrEvt
	~InstrEvt(){
	}//end ~InstrEvt
};//endclass InstrEvt

class KeyEvt{
public:
	int stime;
	string tonic;
	string mode;//major, minor, etc.
	int keyfifth;//0=natural, 1=sharp, -1=flat, 2,-2,...
	int tonic_int;
};//endclass KeyEvt

class MeterEvt{
public:
	int stime;
	int num;
	int den;
	int barlen;//bar length
};//endclass MeterEvt

class SPQNEvt{//sec per QN
public:
	int stime;
	double value;
};//endclass SPQNEvt

class QPRBar{
public:
	MeterEvt info;
	vector<QPREvt> notes;
};//endclass QPRBar

class QuantizedPianoRoll{
public:
	int TPQN;
	vector<InstrEvt> instrEvts;
	vector<QPREvt> evts;
	vector<KeyEvt> keyEvts;
	vector<MeterEvt> meterEvts;
	vector<SPQNEvt> spqnEvts;
	vector<string> comments;//starting with //

	vector<QPRBar> bars;

	QuantizedPianoRoll(){
		Init();
	}//end QuantizedPianoRoll
	~QuantizedPianoRoll(){}//end ~QuantizedPianoRoll

	void Clear(){
		instrEvts.clear();
		evts.clear();
		keyEvts.clear();
		meterEvts.clear();
		spqnEvts.clear();
		comments.clear();
		Init();
	}//end Clear

	void Init(){
		InstrEvt instrEvt;
		for(int i=0;i<16;i+=1){
			instrEvt.channel=i;
			instrEvts.push_back(instrEvt);
		}//endfor i
	}//end Init

	void ReadFile(string filename,int filetype=0){//filetype=0(qpr)/1(qipr)
		vector<int> appearedChannel; appearedChannel.assign(16,0);//0(not appeared)/1(appeared);
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
				ss<<s[0]<<""<<s[99];
				comments.push_back(ss.str());
				continue;
			}else if(s[0][0]=='#'){
				ifs>>s[1];
				if(s[1]=="TPQN"){
					ifs>>TPQN;
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
					ifs>>spqnEvt.stime>>spqnEvt.value;
					spqnEvts.push_back(spqnEvt);
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
		for(int i=0;i<meterEvts.size();i+=1){
			ofs<<"# Meter\t"<<meterEvts[i].stime<<"\t"<<meterEvts[i].num<<"\t"<<meterEvts[i].den<<"\t"<<meterEvts[i].barlen<<"\n";
		}//endfor i
		for(int i=0;i<spqnEvts.size();i+=1){
			ofs<<"# SPQN\t"<<spqnEvts[i].stime<<"\t"<<spqnEvts[i].value<<"\n";
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
		SplitBars();
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
			ofs<<"# SPQN\t"<<spqnEvts[i].stime<<"\t"<<spqnEvts[i].value<<"\n";
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
		Fmt1x fmt1x;
		fmt1x.ReadMusicXML(filename);
		Fmt3x fmt3x;
		fmt3x.ConvertFromFmt1x(fmt1x);
		TPQN=fmt3x.TPQN;

		stringstream ss;
		string str;

		for(int n=0;n<fmt1x.evts.size();n+=1){
			if(fmt1x.evts[n].eventtype!="attributes"){continue;}

			bool found=false;
			for(int i=0;i<keyEvts.size();i+=1){
				if(keyEvts[i].stime==fmt1x.evts[n].stime){found=true;}
			}//endfor i
			if(found){continue;}
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

		}//endfor n

		vector<int> channelsForStaffs;//channelsForStaffs[0,1,...]=1,2,3...?
		vector<int> firstVoiceForStaffs;//firstVoiceForStaffs[0,1,...]=1,3,6,...?
		int maxVoice=0;

		for(int n=0;n<fmt3x.evts.size();n+=1){
			channelsForStaffs.push_back(fmt3x.evts[n].staff);
		}//endfor n
		sort(channelsForStaffs.begin(),channelsForStaffs.end());
		for(int i=channelsForStaffs.size()-1;i>=1;i-=1){
			if(channelsForStaffs[i]==channelsForStaffs[i-1]){channelsForStaffs.erase(channelsForStaffs.begin()+i);}
		}//endfor i
		firstVoiceForStaffs.assign(channelsForStaffs.size(),9999);

		for(int n=0;n<fmt3x.evts.size();n+=1){
			int foundPos=-1;
			for(int i=0;i<channelsForStaffs.size();i+=1){
				if(channelsForStaffs[i]==fmt3x.evts[n].staff){foundPos=i;break;}
			}//endfor i
			assert(foundPos>=0);
			if(fmt3x.evts[n].voice<firstVoiceForStaffs[foundPos]){firstVoiceForStaffs[foundPos]=fmt3x.evts[n].voice;}
			if(fmt3x.evts[n].voice>maxVoice){maxVoice=fmt3x.evts[n].voice;}
		}//endfor n

		vector<int> staffToChannel(100);
		for(int i=0;i<channelsForStaffs.size();i+=1){
			staffToChannel[channelsForStaffs[i]]=i;
		}//endfor i

		QPREvt evt;
		for(int n=0;n<fmt3x.evts.size();n+=1){
			if(fmt3x.evts[n].dur==0){continue;}
			evt.onstime=fmt3x.evts[n].stime;
			evt.offstime=fmt3x.evts[n].stime+fmt3x.evts[n].dur;
			evt.channel=staffToChannel[fmt3x.evts[n].staff];
			evt.subvoice=fmt3x.evts[n].voice-firstVoiceForStaffs[staffToChannel[fmt3x.evts[n].staff]];
			if(fmt3x.evts[n].eventtype=="rest"){
				evt.sitch="R";
				evt.pitch=-1;
				evt.label="-";
				evts.push_back(evt);
			}else if(fmt3x.evts[n].eventtype=="chord"){
				for(int i=0;i<fmt3x.evts[n].sitches.size();i+=1){
					evt.sitch=fmt3x.evts[n].sitches[i];
					evt.pitch=SitchToPitch(evt.sitch);
					evt.label=fmt3x.evts[n].fmt1IDs[i];
					evts.push_back(evt);
				}//endfor i
			}else{
				continue;
			}//endif

		}//endfor n

		SPQNEvt spqnEvt;
		spqnEvt.stime=0;
		spqnEvt.value=0.5;
		spqnEvts.push_back(spqnEvt);

		vector<int> appearedChannel; appearedChannel.assign(16,0);//0(not appeared)/1(appeared);
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

		for(int i=0;i<meterEvts.size();i+=1){
			MidiMessage midiMes;
			midiMes.tick=meterEvts[i].stime;
			midiMes.mes.push_back(255); midiMes.mes.push_back(88); midiMes.mes.push_back(4);
			midiMes.mes.push_back(meterEvts[i].num);
			if(meterEvts[i].den==1){midiMes.mes.push_back(0);
			}else if(meterEvts[i].den==2){midiMes.mes.push_back(1);
			}else if(meterEvts[i].den==4){midiMes.mes.push_back(2);
			}else if(meterEvts[i].den==8){midiMes.mes.push_back(3);
			}else if(meterEvts[i].den==16){midiMes.mes.push_back(4);
			}else if(meterEvts[i].den==32){midiMes.mes.push_back(5);
			}else{midiMes.mes.push_back(2);
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

	void SplitBars(){
		bars.clear();
		Sort();
		int lastStime=0;
		for(int n=evts.size()-1;n>=0;n-=1){
			if(evts[n].offstime<lastStime){continue;}
			lastStime=evts[n].offstime;
		}//endfor n

		//Construct bars
		QPRBar bar;
		for(int i=0;i<meterEvts.size();i+=1){
			if(i==meterEvts.size()-1){
				for(int tau=meterEvts[i].stime;tau<lastStime;tau+=meterEvts[i].barlen){
					bar.info=meterEvts[i];
					bar.info.stime=tau;
					bar.notes.clear();
					bars.push_back(bar);
				}//endfor tau
				continue;
			}//endif
			for(int tau=meterEvts[i].stime;tau<meterEvts[i+1].stime;tau+=meterEvts[i].barlen){
				bar.info=meterEvts[i];
				bar.info.stime=tau;
				bar.notes.clear();
				bars.push_back(bar);
			}//endfor tau
		}//endfor i

		//Put notes
		for(int n=0;n<evts.size();n+=1){
			int barpos=-1;
			for(int m=0;m<bars.size();m+=1){
				if(evts[n].onstime>=bars[m].info.stime && evts[n].onstime<bars[m].info.stime+bars[m].info.barlen){
					barpos=m; break;
				}//endif
			}//endfor m
			assert(barpos>=0);
			evts[n].onMetpos=evts[n].onstime-bars[barpos].info.stime;
			bars[barpos].notes.push_back(evts[n]);
		}//endfor n

	}//end SplitBars

	void ChangeTPQN(int newTPQN){
//		SplitBars();
//		cout<<"WARNING: ChangeTPQN erases all notes that are not integral in the new TPQN"<<endl;

		for(int n=evts.size()-1;n>=0;n-=1){
			if((evts[n].onstime*newTPQN)%TPQN!=0){evts.erase(evts.begin()+n); continue;}
			if((evts[n].offstime*newTPQN)%TPQN!=0){evts.erase(evts.begin()+n); continue;}
			evts[n].onstime=(evts[n].onstime*newTPQN)/TPQN;
			evts[n].offstime=(evts[n].offstime*newTPQN)/TPQN;
		}//endfor n

		for(int i=keyEvts.size()-1;i>=0;i-=1){
			if((keyEvts[i].stime*newTPQN)%TPQN!=0){keyEvts.erase(keyEvts.begin()+i); continue;}
			keyEvts[i].stime=(keyEvts[i].stime*newTPQN)/TPQN;
		}//endfor i
		for(int i=meterEvts.size()-1;i>=0;i-=1){
			if((meterEvts[i].stime*newTPQN)%TPQN!=0){meterEvts.erase(meterEvts.begin()+i); continue;}
			meterEvts[i].stime=(meterEvts[i].stime*newTPQN)/TPQN;
			meterEvts[i].barlen=(meterEvts[i].barlen*newTPQN)/TPQN;
		}//endfor i
		for(int i=spqnEvts.size()-1;i>=0;i-=1){
			if((spqnEvts[i].stime*newTPQN)%TPQN!=0){spqnEvts.erase(spqnEvts.begin()+i); continue;}
			spqnEvts[i].stime=(spqnEvts[i].stime*newTPQN)/TPQN;
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

/*
	void ReadMIDIFile(string midiFile){
		Clear();
		Midi midi;
		midi.ReadFile(midiFile);

		int onPosition[16][128];
		for(int i=0;i<16;i+=1)for(int j=0;j<128;j+=1){onPosition[i][j]=-1;}//endfor i,j
		PianoRollEvt evt;
		int curChan;

		for(int n=0;n<midi.content.size();n+=1){
			if(midi.content[n].mes[0]>=128 && midi.content[n].mes[0]<160){//note-on or note-off event
				curChan=midi.content[n].mes[0]%16;
				if(midi.content[n].mes[0]>=144 && midi.content[n].mes[2]>0){//note-on
					if(onPosition[curChan][midi.content[n].mes[1]]>=0){
// cout<<"Warning: (Double) note-on event before a note-off event "<<PitchToSitch(midi.content[n].mes[1])<<endl;
						evts[onPosition[curChan][midi.content[n].mes[1]]].offtime=midi.content[n].time;
						evts[onPosition[curChan][midi.content[n].mes[1]]].offvel=-1;
					}//endif
					onPosition[curChan][midi.content[n].mes[1]]=evts.size();
					evt.channel=curChan;
					evt.sitch=PitchToSitch(midi.content[n].mes[1]);
					evt.pitch=midi.content[n].mes[1];
					evt.onvel=midi.content[n].mes[2];
					evt.offvel=0;
					evt.ontime=midi.content[n].time;
					evt.offtime=evt.ontime+0.1;
					evts.push_back(evt);
				}else{//note-off
					if(onPosition[curChan][midi.content[n].mes[1]]<0){
// cout<<"Warning: Note-off event before a note-on event "<<PitchToSitch(midi.content[n].mes[1])<<endl;
// cout<<midi.content[n].time<<endl;
						continue;
					}//endif
					evts[onPosition[curChan][midi.content[n].mes[1]]].offtime=midi.content[n].time;
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
cout<<"Error: Note without a note-off event"<<endl;
cout<<"ontime channel sitch : "<<evts[onPosition[i][j]].ontime<<"\t"<<evts[onPosition[i][j]].channel<<"\t"<<evts[onPosition[i][j]].sitch<<endl;
				return;
			}//endif
		}//endfor i,j

		/// Extract pedal information
		PedalEvt pedal;
		for(int n=0;n<midi.content.size();n+=1){
			if(midi.content[n].mes[0]<176 || midi.content[n].mes[0]>=192){continue;}
			if(midi.content[n].mes[1]==64){
				pedal.type="SusPed";
				pedal.time=midi.content[n].time;
				pedal.value=midi.content[n].mes[2];
				pedal.channel=midi.content[n].mes[0]%16;
			}else if(midi.content[n].mes[1]==66){
				pedal.type="SosPed";
				pedal.time=midi.content[n].time;
				pedal.value=midi.content[n].mes[2];
				pedal.channel=midi.content[n].mes[0]%16;
			}else if(midi.content[n].mes[1]==67){
				pedal.type="SofPed";
				pedal.time=midi.content[n].time;
				pedal.value=midi.content[n].mes[2];
				pedal.channel=midi.content[n].mes[0]%16;
			}else{
				continue;
			}//endif
			pedals.push_back(pedal);
		}//endfor n

		for(int n=0;n<evts.size();n+=1){
			stringstream ss;
			ss.str(""); ss<<n;
			evts[n].ID=ss.str();
		}//endfor n

		SetPedalIntervals();
		SetEndtimes();

	}//end ReadMIDIFile
*/

	void Sort(){
		stable_sort(evts.begin(), evts.end(), LessQPREvt());
	}//end Sort

};//end class QuantizedPianoRoll

#endif // QuantizedPianoRoll_HPP

