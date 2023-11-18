#ifndef Trx_HPP
#define Trx_HPP

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
#include "NoteValueCalculation_v170108_2.hpp"

using namespace std;

class TrxEvt : public PianoRollEvt{
public:
//	string ID;
//	double ontime;
//	double offtime;
//	string sitch;
//	int pitch;//integral pitch
//	int onvel;
//	int offvel;
//	int channel;
//	double endtime;//Including pedalling. Not written in spr/ipr files.
	int onstime;
	int offstime;
	int voice;
	double secPerQN;//secPerTick=secPerQN/TPQN
	int fmt3xPos1;//Cluster ID, -1 if N/A.
	int fmt3xPos2;//Note ID, -1 if N/A.
	IrredFrac normStime;//Not written in tr files

	TrxEvt(){}//end TrxEvt
	TrxEvt(PianoRollEvt prevt){
		ID=prevt.ID;
		ontime=prevt.ontime;
		offtime=prevt.offtime;
		sitch=prevt.sitch;
		pitch=prevt.pitch;
		onvel=prevt.onvel;
		offvel=prevt.offvel;
		channel=prevt.channel;
		endtime=prevt.endtime;

		onstime=-1;
		offstime=-1;
		voice=-1;
		secPerQN=-1;
		fmt3xPos1=-1;
		fmt3xPos2=-1;
	}//end TrEvt
	~TrxEvt(){}//end ~TrxEvt

};//endclass TrxEvt

class Trx{
public:
	string version;
	int TPQN;
	string meter;//Duple or Triple
	string fmt3xFile;
	double logP;
	vector<string> comments;
	vector<TrxEvt> evts;

	Trx(){
		version="";
		meter="Duple";
		fmt3xFile="NA";
		logP=0;
	}//end Tr
	Trx(Trx const &tr){
		version=tr.version;
		TPQN=tr.TPQN;
		meter=tr.meter;
		fmt3xFile=tr.fmt3xFile;
		logP=tr.logP;
		comments=tr.comments;
		evts=tr.evts;
	}//end ScorePerfmMatch
	~Trx(){}//end ~Trx

	void Clear(){
		version="";
		meter="Duple";
		fmt3xFile="NA";
		logP=0;
		comments.clear();
		evts.clear();
	}//end Clear

	void ReadFile(string filename){
		Clear();
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		TrxEvt evt;
		ifstream ifs(filename.c_str());
		while(ifs>>s[0]){
			if(s[0][0]=='/' || s[0][0]=='#'){
				if(s[0]=="//TPQN:"){
					ifs>>TPQN;
					getline(ifs,s[99]); continue;
				}else if(s[0]=="//Version:"){
					ifs>>version;
					getline(ifs,s[99]); continue;
				}else if(s[0]=="//LogProb:"){
					ifs>>logP;
					getline(ifs,s[99]); continue;
				}else if(s[0]=="//Meter:"){
					ifs>>meter;
					getline(ifs,s[99]); continue;
				}else if(s[0]=="//Fmt3xFile:"){
					ifs>>fmt3xFile;
					getline(ifs,s[99]); continue;
				}//endif
				getline(ifs,s[99]);
				comments.push_back(s[99]);
				continue;
			}//endif

			evt.ID = s[0];
			ifs>>evt.ontime>>evt.offtime>>evt.sitch>>evt.onvel>>evt.offvel>>evt.channel;
			ifs>>evt.onstime>>evt.offstime>>evt.voice>>evt.secPerQN>>evt.fmt3xPos1>>evt.fmt3xPos2>>evt.endtime;
			evt.pitch=SitchToPitch(evt.sitch);
			evts.push_back(evt);
			getline(ifs,s[99]);
		}//endwhile
		ifs.close();
	}//end ReadFile

	void WriteFile(string filename){
		ofstream ofs(filename.c_str());
		ofs<<"//Version: Trx_v170203\n";
		ofs<<"//TPQN: "<<TPQN<<"\n";
		ofs<<"//Meter: "<<meter<<"\n";
		ofs<<"//Fmt3xFile: "<<fmt3xFile<<"\n";
		ofs<<"//LogProb: "<<logP<<"\n";
		for(int i=0;i<comments.size();i+=1){
			ofs<<"//"<<comments[i]<<"\n";
		}//endfor i
		for(int n=0;n<evts.size();n+=1){
			TrxEvt evt=evts[n];
			ofs<<evt.ID<<"\t"<<evt.ontime<<"\t"<<evt.offtime<<"\t"<<evt.sitch<<"\t"<<evt.onvel<<"\t"<<evt.offvel<<"\t"<<evt.channel<<"\t";
			ofs<<evt.onstime<<"\t"<<evt.offstime<<"\t"<<evt.voice<<"\t";
			ofs<<evt.secPerQN<<"\t"<<evt.fmt3xPos1<<"\t"<<evt.fmt3xPos2<<"\t"<<evt.endtime<<"\n";
		}//endfor n
		ofs.close();
	}//end WriteFile

	void ReadSprFile(string filename){
		Clear();
		stringstream ss;
		ss.str("");
		ss<<" Converted from spr file: "<<filename;
		comments.push_back(ss.str());

		PianoRoll pr;
		pr.ReadFileSpr(filename);

		TPQN=-1;
		for(int n=0;n<pr.evts.size();n+=1){
			evts.push_back(TrxEvt(pr.evts[n]));
		}//endfor n
	}//end ReadSprFile

	void ReadMIDIFile(string filename){
		Clear();
		stringstream ss;
		ss.str("");
		ss<<" Converted from MIDI file: "<<filename;
		comments.push_back(ss.str());

		PianoRoll pr;
		pr.ReadMIDIFile(filename);

		TPQN=-1;
		for(int n=0;n<pr.evts.size();n+=1){
			evts.push_back(TrxEvt(pr.evts[n]));
		}//endfor n
	}//end ReadMIDIFile

	void WriteMIDIFile(string filename,int tempo_=120,int num_=4,int den_=4){
		Midi midi;
		MidiMessage midiMes;
		for(int i=0;i<evts.size();i+=1){
			midiMes.tick=evts[i].onstime;
			midiMes.mes.assign(3,0);
			midiMes.mes[0]=144+evts[i].channel;
			midiMes.mes[1]=evts[i].pitch;
			midiMes.mes[2]=evts[i].onvel;
			midi.content.push_back(midiMes);

			midiMes.tick=evts[i].offstime;
			midiMes.mes.assign(3,0);
			midiMes.mes[0]=128+evts[i].channel;
			midiMes.mes[1]=evts[i].pitch;
			midiMes.mes[2]=((evts[i].offvel>0)? evts[i].offvel:80);
			midi.content.push_back(midiMes);
		}//endfor i
		stable_sort(midi.content.begin(),midi.content.end(),LessTickMidiMessage());

		midi.SetStrDataTickBased(TPQN,tempo_,num_,den_);

		ofstream ofs;
		ofs.open(filename.c_str(), std::ios::binary);
		ofs<<midi.strData;
		ofs.close();
	}//end WriteMIDIFile


};//end class Trx

#endif // Trx_HPP

