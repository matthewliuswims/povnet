#ifndef MusicXML_HPP
#define MusicXML_HPP

#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
#include<vector>
#include<fstream>
#include<cassert>
#include<algorithm>
#include"NoteValueCalculation_v170108_2.hpp"
using namespace std;

// inline void DeleteHeadSpace(string &buf){
// 	size_t pos;
// 	while((pos = buf.find_first_of(" 　\t")) == 0){
// 		buf.erase(buf.begin());
// 		if(buf.empty()) break;
// 	}//endwhile
// }//end DeleteHeadSpace
// 
// inline vector<string> UnspaceString(string str){
// 	vector<string> vs;
// 	while(str.size()>0){
// 		DeleteHeadSpace(str);
// 		if(str=="" || isspace(str[0])!=0){break;}
// 		vs.push_back(str.substr(0,str.find_first_of(" 　\t")));
// 		for(int i=0;i<vs[vs.size()-1].size();i+=1){str.erase(str.begin());}
// 	}//endwhile
// 	return vs;
// }//end UnspaceString

class NotatableNoteValues{
public:
	vector<IrredFrac> notatableNVs;//Sorted from long ones

	NotatableNoteValues(){
		notatableNVs.push_back( IrredFrac(12,1) );
		notatableNVs.push_back( IrredFrac(8,1) );
		notatableNVs.push_back( IrredFrac(6,1) );
		notatableNVs.push_back( IrredFrac(4,1) );
		notatableNVs.push_back( IrredFrac(3,1) );
		notatableNVs.push_back( IrredFrac(2,1) );
		notatableNVs.push_back( IrredFrac(3,2) );
		notatableNVs.push_back( IrredFrac(1,1) );
		notatableNVs.push_back( IrredFrac(3,4) );
		notatableNVs.push_back( IrredFrac(1,2) );
		notatableNVs.push_back( IrredFrac(3,8) );
		notatableNVs.push_back( IrredFrac(1,4) );
		notatableNVs.push_back( IrredFrac(3,16) );
		notatableNVs.push_back( IrredFrac(1,8) );
		notatableNVs.push_back( IrredFrac(3,32) );
		notatableNVs.push_back( IrredFrac(1,16) );
		notatableNVs.push_back( IrredFrac(3,64) );
		notatableNVs.push_back( IrredFrac(1,32) );
	}//end NotatableNoteValues
	~NotatableNoteValues(){
	}//end ~NotatableNoteValues

	IrredFrac FindLongest(IrredFrac nv){
		for(int i=0;i<notatableNVs.size();i+=1){
			if(notatableNVs[i].Value()<=nv.Value()){
				return notatableNVs[i];
				break;
			}//endif
		}//endfor i
		cout<<nv.Value()<<endl;
		assert(false);
		return IrredFrac(0,1);
	}//end FindLongest

	vector<IrredFrac> NVBreakDown(IrredFrac nv){
		vector<IrredFrac> nvs;
		IrredFrac notatableLongestNV;
		while(nv.Show()!="0"){
			notatableLongestNV=FindLongest(nv);
			nvs.push_back(notatableLongestNV);
			nv=SubtrIrredFrac(nv,notatableLongestNV);
		}//endwhile
		return nvs;
	}//end NVBreakDown

};//endclass NotatableNoteValues


class XMLEvt{
public:
	int depth;
	int type;//0(bar)/1(open tag)/2(close tag)/3(self-closed)/4(other)
	string content;
	XMLEvt(){
	}//end XMLEvt
	XMLEvt(int depth_,int type_,string content_){
		depth=depth_;
		type=type_;
		content=content_;
	}//end XMLEvt
	~XMLEvt(){
	}//end ~XMLEvt

	void Print(){
cout<<depth<<"\t"<<type<<"\t"<<content<<endl;
	}//end Print

};//endclass XMLEvt

class XML{
public:
	vector<XMLEvt> evts;

	void ReadFile(string filename){

		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		ifstream ifs(filename.c_str());
		string all;
		while(!ifs.eof()){
			getline(ifs,s[99]);
			all+=s[99];
		}//endwhile
		ifs.close();

		evts.clear();
		XMLEvt evt;
{
		bool isInBracket=false;
		int depth=0;
		for(int i=0;i<all.size();i+=1){
			if(all[i]=='<'){
				isInBracket=true;
				if(all[i+1]=='/'){depth-=1;
				}else if(all[i+1]=='!' || all[i+1]=='?'){;
				}else{depth+=1;
				}//endif
				evt.content=s[0];
				evt.depth=depth;
				evt.type=0;
				evts.push_back(evt);
				s[0]="";
				continue;
			}else if(all[i]=='>'){
				isInBracket=false;
				if(all[i-1]=='/'){depth-=1;}
				evt.content=s[0];
				evt.depth=depth;
				evt.type=1;
				evts.push_back(evt);
				s[0]="";
				continue;
			}//endif
			s[0]+=all[i];
		}//endfor i
}//

{
		vector<XMLEvt> pre_evts;
		pre_evts=evts;
		evts.clear();
		for(int i=0;i<pre_evts.size();i+=1){
			DeleteHeadSpace(pre_evts[i].content);
			if(pre_evts[i].content=="" || isspace(pre_evts[i].content[0])!=0){continue;}
			if(pre_evts[i].type){
				if(pre_evts[i].content[0]!='/'&&pre_evts[i].content[0]!='!'&&pre_evts[i].content[0]!='?'){
					pre_evts[i].depth-=1;
					if(pre_evts[i].content[pre_evts[i].content.size()-1]=='/'){
						pre_evts[i].depth+=1;
					}//endif
				}//endif
			}//endif
			if(pre_evts[i].content[0]=='/'){pre_evts[i].type=2;}
			if(pre_evts[i].type==1&&pre_evts[i].content[pre_evts[i].content.size()-1]=='/'){pre_evts[i].type=3;}
			if(pre_evts[i].content[0]=='!'||pre_evts[i].content[0]=='?'){pre_evts[i].type=4;}
			evts.push_back(pre_evts[i]);
		}//endfor i
}//

//	for(int i=0;i<evts.size();i+=1){cout<<evts[i].depth<<"\t"<<evts[i].type<<"\t"<<evts[i].content<<endl;}//endfor i

	}//end ReadFile

	void WriteFile(string filename){

		bool newline=true;
		ofstream ofs(filename.c_str());
		for(int i=0;i<evts.size();i+=1){
			if(newline){
				for(int k=0;k<evts[i].depth;k+=1){ofs<<"  ";}
			}//endif
			if(evts[i].type==0){
				ofs<<evts[i].content;
				newline=false;
			}else if(evts[i].type==2){
				ofs<<"<"<<evts[i].content<<">\n";
				newline=true;
			}else if(evts[i].type==1||evts[i].type==3||evts[i].type==4){
				ofs<<"<"<<evts[i].content<<">";
				newline=false;
				if(evts[i].type==3||evts[i].type==4){
					ofs<<"\n";
					newline=true;
				}else if(evts[i+1].depth==evts[i].depth+1){
					ofs<<"\n";
					newline=true;
				}//endif
			}else{//Never reach here
			}//endif
		}//endfor i
		ofs.close();

	}//end WriteFile

	void WriteXMLEvts(string filename){

		ofstream ofs(filename.c_str());
		for(int i=0;i<evts.size();i+=1){
			ofs<<evts[i].depth<<"\t"<<evts[i].type<<"\t"<<evts[i].content<<"\n";
		}//endfor i
		ofs.close();

	}//end WriteXMLEvts

};//endclass XML


class Pitch{
public:
	string step;//A,...,G or rest
	int alter;
	int octave;
	string sitch;

	void ReadSitch(){
		if(sitch=="rest"){
			step="rest";
			return;
		}//endif
		alter=0;
		stringstream ss;
		step=sitch.substr(0,1);
		ss.str(""); ss<<sitch[sitch.size()-1];
		octave=atoi(ss.str().c_str());
		for(int j=1;j<sitch.size();j+=1){
			if(sitch[j]=='#'){alter+=1;}
			if(sitch[j]=='b'){alter-=1;}
		}//endfor j
	}//end ReadSitch

	void WriteSitch(){
		if(step=="rest"){
			sitch="rest";
			return;
		}//endif
		stringstream ss;
		ss<<step;
		if(alter==1){ss<<"#";
		}else if(alter==-1){ss<<"b";
		}else if(alter==2){ss<<"##";
		}else if(alter==-2){ss<<"bb";
		}//endif
		ss<<octave;
		sitch=ss.str();
	}//end WriteSitch

	vector<XMLEvt> Output(){
		vector<XMLEvt> out;
		stringstream ss;
		if(step=="rest"){
			out.push_back( XMLEvt(4,3,"rest/") );
			return out;
		}//endif
		out.push_back( XMLEvt(4,1,"pitch") );
		out.push_back( XMLEvt(5,1,"step") );
		out.push_back( XMLEvt(5,0,step) );
		out.push_back( XMLEvt(5,2,"/step") );
		if(alter!=0){
		out.push_back( XMLEvt(5,1,"alter") );
		ss.str(""); ss<<alter;
		out.push_back( XMLEvt(5,0,ss.str()) );
		out.push_back( XMLEvt(5,2,"/alter") );
		}//endif
		out.push_back( XMLEvt(5,1,"octave") );
		ss.str(""); ss<<octave;
		out.push_back( XMLEvt(5,0,ss.str()) );
		out.push_back( XMLEvt(5,2,"/octave") );
		out.push_back( XMLEvt(4,2,"/pitch") );
		return out;
	}//end Output

};//endclass Pitch

class Note{
public:
	Pitch pitch;//includes rest
	int duration;
	int voice;
	int staff;//-1 if N/A
	string type;//note symbol type: breve/whole/half/quarter/eighth/16th/32nd/64th/128th
	bool chord;
	int dot;//0,1,2,...
	int tie;//0(none)/1(start)/2(stop)/3(stop & start)

	void ReadDuration(int TPQN){//Set type and dot from duration
		IrredFrac nv=IrredFrac(duration,TPQN);

		if(nv.Show()=="8"){
			type="breve"; dot=0;
		}else if(nv.Show()=="12"){
			type="breve"; dot=1;
		}else if(nv.Show()=="14"){
			type="breve"; dot=2;
		}else if(nv.Show()=="15"){
			type="breve"; dot=3;

		}else if(nv.Show()=="4"){
			type="whole"; dot=0;
		}else if(nv.Show()=="6"){
			type="whole"; dot=1;
		}else if(nv.Show()=="7"){
			type="whole"; dot=2;
		}else if(nv.Show()=="15/2"){
			type="whole"; dot=3;

		}else if(nv.Show()=="2"){
			type="half"; dot=0;
		}else if(nv.Show()=="3"){
			type="half"; dot=1;
		}else if(nv.Show()=="7/2"){
			type="half"; dot=2;
		}else if(nv.Show()=="15/4"){
			type="half"; dot=3;

		}else if(nv.Show()=="1"){
			type="quarter"; dot=0;
		}else if(nv.Show()=="3/2"){
			type="quarter"; dot=1;
		}else if(nv.Show()=="7/4"){
			type="quarter"; dot=2;
		}else if(nv.Show()=="15/8"){
			type="quarter"; dot=3;

		}else if(nv.Show()=="1/2"){
			type="eighth"; dot=0;
		}else if(nv.Show()=="3/4"){
			type="eighth"; dot=1;
		}else if(nv.Show()=="7/8"){
			type="eighth"; dot=2;
		}else if(nv.Show()=="15/16"){
			type="eighth"; dot=3;

		}else if(nv.Show()=="1/4"){
			type="16th"; dot=0;
		}else if(nv.Show()=="3/8"){
			type="16th"; dot=1;
		}else if(nv.Show()=="7/16"){
			type="16th"; dot=2;
		}else if(nv.Show()=="15/32"){
			type="16th"; dot=3;

		}else if(nv.Show()=="1/8"){
			type="32nd"; dot=0;
		}else if(nv.Show()=="3/16"){
			type="32nd"; dot=1;
		}else if(nv.Show()=="7/32"){
			type="32nd"; dot=2;
		}else if(nv.Show()=="15/64"){
			type="32nd"; dot=3;

		}else if(nv.Show()=="1/16"){
			type="64th"; dot=0;
		}else if(nv.Show()=="3/32"){
			type="64th"; dot=1;
		}else if(nv.Show()=="7/64"){
			type="64th"; dot=2;
		}else if(nv.Show()=="15/128"){
			type="64th"; dot=3;

		}else if(nv.Show()=="1/32"){
			type="128th"; dot=0;
		}else if(nv.Show()=="3/64"){
			type="128th"; dot=1;
		}else if(nv.Show()=="7/128"){
			type="128th"; dot=2;
		}else if(nv.Show()=="15/256"){
			type="128th"; dot=3;

		}else{
			cout<<"Unknown note value : "<<nv.Show()<<endl;
			assert(false);
		}//endif

	}//end ReadDuration

	void Print(){
		cout<<"sitch : "<<pitch.sitch<<endl;
		cout<<"duration : "<<duration<<endl;
		cout<<"voice : "<<voice<<endl;
		cout<<"staff : "<<staff<<endl;
		cout<<"type : "<<type<<endl;
		cout<<"chord : "<<chord<<endl;
		cout<<"dot : "<<dot<<endl;
		cout<<"tie : "<<tie<<endl;
	Pitch pitch;//includes rest

	}//end Print

	vector<XMLEvt> Output(){
		vector<XMLEvt> out,tmp;
		stringstream ss;
		out.insert(out.begin(), XMLEvt(3,1,"note") );
		if(chord){
			out.push_back( XMLEvt(4,3,"chord/") );
		}//endif
		tmp=pitch.Output();
		for(int j=0;j<tmp.size();j+=1){out.push_back(tmp[j]);}
		out.push_back( XMLEvt(4,1,"duration") );
		ss.str(""); ss<<duration;
		out.push_back( XMLEvt(4,0,ss.str()) );
		out.push_back( XMLEvt(4,2,"/duration") );
		if(tie>=2){//if stop
			out.push_back( XMLEvt(4,3,"tie type=\"stop\"/") );
		}//endif
		if(tie%2==1){//if start
			out.push_back( XMLEvt(4,3,"tie type=\"start\"/") );
		}//endif
		out.push_back( XMLEvt(4,1,"voice") );
		ss.str(""); ss<<voice;
		out.push_back( XMLEvt(4,0,ss.str()) );
		out.push_back( XMLEvt(4,2,"/voice") );
		out.push_back( XMLEvt(4,1,"type") );
		out.push_back( XMLEvt(4,0,type) );
		out.push_back( XMLEvt(4,2,"/type") );
		for(int ndots=0;ndots<dot;ndots+=1){
			out.push_back( XMLEvt(4,3,"dot/") );
		}//endfor ndots
		if(staff!=-1){
			out.push_back( XMLEvt(4,1,"staff") );
			ss.str(""); ss<<staff;
			out.push_back( XMLEvt(4,0,ss.str()) );
			out.push_back( XMLEvt(4,2,"/staff") );
		}//endif
		out.push_back( XMLEvt(3,2,"/note") );
		return out;
	}//end Output

};//endclass Note

class Harmony{
public:
	string fullname;
	string root_step;
	int root_alter;
	string kind_text;
	string kind;
	bool withBass;
	string bass_step;
	int bass_alter;
	int offset;
	int degree_value;
	int degree_alter;
	string degree_type;

	void WriteFullname(){
		stringstream ss;
		ss<<root_step;
		if(root_alter==1){ss<<"#";
		}else if(root_alter==-1){ss<<"b";
		}else if(root_alter==2){ss<<"##";
		}else if(root_alter==-2){ss<<"bb";
		}//endif
		ss<<kind_text;
//		ss<<":"<<kind_text;

		if(degree_value>=0){
//cout<<degree_value<<"\t"<<degree_alter<<"\t"<<degree_type<<endl;
			ss<<"(";
			if(degree_type=="add"){
				ss<<"add";
			}else if(degree_type=="subtract"){
				ss<<"omit";
			}else if(degree_type=="alter"){
			}else{
cout<<"Unknown chord degree: "<<degree_value<<"\t"<<degree_alter<<"\t"<<degree_type<<endl;
			}//endif
			if(degree_alter==1){ss<<"#";
			}else if(degree_alter==-1){ss<<"b";
			}else if(degree_alter==0){
			}else{
cout<<"Unknown chord degree: "<<degree_value<<"\t"<<degree_alter<<"\t"<<degree_type<<endl;
			}//endif
			ss<<degree_value<<")";
		}//endif

		if(!withBass){
			fullname=ss.str();
			return;
		}//endif
		ss<<"/"<<bass_step;
		if(bass_alter==1){ss<<"#";
		}else if(bass_alter==-1){ss<<"b";
		}else if(bass_alter==2){ss<<"##";
		}else if(bass_alter==-2){ss<<"bb";
		}//endif
		fullname=ss.str();
	}//end WriteFullname

	void SetKindTextFromKind(){
		if(kind=="dominant"){kind_text="7";
		}else if(kind=="dominant-seventh"){kind_text="7";
		}else if(kind=="minor-seventh"){kind_text="m7";
		}else if(kind=="major-seventh"){kind_text="M7";
		}else if(kind=="diminished-seventh"){kind_text="dim7";
		}else if(kind=="augmented-seventh"){kind_text="aug7";
		}else if(kind=="major-minor"){kind_text="mM7";
		}else if(kind=="minor-major"){kind_text="mM7";
		}else if(kind=="half-diminished"){kind_text="m7(b5)";
		}else if(kind=="minor"){kind_text="m";
		}else if(kind=="diminished"){kind_text="dim";
		}else if(kind=="augmented"){kind_text="aug";
		}else if(kind=="suspended-fourth"){kind_text="sus4";
		}else if(kind=="dominant-ninth"){kind_text="9";
		}else if(kind=="minor-ninth"){kind_text="m9";
		}else if(kind=="major-ninth"){kind_text="M9";
		}else if(kind=="augmented-ninth"){kind_text="aug9";
		}else if(kind=="major-11th"){kind_text="M11";
		}else if(kind=="minor-11th"){kind_text="m11";
		}else if(kind=="dominant-13th"){kind_text="13";
		}else if(kind=="major-13th"){kind_text="M13";
		}else if(kind=="major-sixth"){kind_text="6";
		}else if(kind=="minor-sixth"){kind_text="m6";

		}else if(kind=="min"){kind_text="m";
		}else if(kind=="dim"){kind_text="dim";
		}else if(kind=="aug"){kind_text="aug";
		}else if(kind=="min7"){kind_text="m7";
		}else if(kind=="7"){kind_text="7";
		}else if(kind=="sus47"){kind_text="7sus4";
		}else if(kind=="maj69"){kind_text="6(add9)";
		}else if(kind=="maj7"){kind_text="M7";
		}else if(kind=="m7b5"){kind_text="m7(b5)";
		}else if(kind=="9"){kind_text="9";
		}else if(kind=="min6"){kind_text="m6";

// 		}else if(kind==""){kind_text="";
// 		}else if(kind==""){kind_text="";
		}else{
			cout<<"Unknown chord type: "<<kind<<endl;
		}//endif
	}//end SetKindTextFromKind

	void NormalizeKindText(){
		if(kind_text=="+"){kind_text="aug";
		}else if(kind_text=="sus"){kind_text="sus4";
		}else if(kind_text=="min" || kind_text=="mi" || kind_text=="-"){kind_text="m";
		}else if(kind_text=="o"){kind_text="dim";
		}else if(kind_text=="7+"){kind_text="aug7";
		}else if(kind_text=="5"){kind_text="(omit3)";
		}else if(kind_text=="m7b5" || kind_text=="mi7b5"){kind_text="m7(b5)";
		}else if(kind_text=="mi7" || kind_text=="min7" || kind_text=="-7"){kind_text="m7";
		}else if(kind_text=="ma7" || kind_text=="Ma7"){kind_text="M7";
		}else if(kind_text=="mMaj7" || kind_text=="m(maj7)" || kind_text=="mma7"){kind_text="mM7";
		}else if(kind_text=="7(b9)"){kind_text="7(addb9)";
		}else if(kind_text=="mi9"){kind_text="m9";
		}else if(kind_text=="Maj9" || kind_text=="ma9" || kind_text=="maj9"){kind_text="M9";
		}else if(kind_text=="mi11"){kind_text="m11";
		}else if(kind_text=="Maj13" || kind_text=="ma13"){kind_text="M13";
		}else if(kind_text=="mi6" || kind_text=="min6"){kind_text="m6";

		}//endif
	}//end NormalizeKind

	void ReadFullname(){
		string bassname;
		string chordform;
		int posKind;
		withBass=false;
		if(fullname.find("/")!=string::npos){
			withBass=true;
			bassname=fullname.substr(fullname.find("/")+1,fullname.size());
			chordform=fullname.substr(0,fullname.find("/"));
		}else if(fullname.find("on")!=string::npos){
			withBass=true;
			bassname=fullname.substr(fullname.find("on")+2,fullname.size());
			chordform=fullname.substr(0,fullname.find("on"));
		}else{
			chordform=fullname;
		}//endif
		root_step=chordform.substr(0,1);
		root_alter=0;
		posKind=1;
		for(int j=1;j<chordform.size();j+=1){
			if(chordform[j]!='#'&&chordform[j]!='b'){break;}
			if(chordform[j]=='#'){root_alter+=1;}
			if(chordform[j]=='b'){root_alter-=1;}
			posKind=j+1;
		}//endfor j
		kind_text=chordform.substr(posKind,chordform.size());
		if(kind_text==""){
			kind="major";
		}else if(kind_text=="m"){
			kind="minor";
		}else if(kind_text=="7"){
			kind="seventh";//??????????????
		}else if(kind_text=="dim"){
			kind="diminished";
		}else if(kind_text=="aug"){
			kind="augmented";
		}else if(kind_text=="M7"){
			kind="major-seventh";
		}else if(kind_text=="m7"){
			kind="minor-seventh";
		}else if(kind_text=="m7-5"){
			kind="half-diminished";
		}else if(kind_text=="aug7"){
			kind="augmented-seventh";
		}else if(kind_text=="dim7"){
			kind="diminished-seventh";//??????????????
		}else if(kind_text=="sus4"){
			kind="suspended-fourth";//??????????????
		}else if(kind_text=="6"){
			kind="sixth";//??????????????
		}else if(kind_text=="9"){
			kind="ninth";//??????????????
		}else if(kind_text=="M9"){
			kind="major-ninth";//??????????????
		}else if(kind_text=="m9"){
			kind="minor-ninth";//??????????????
		}else if(kind_text=="sus47"){
			kind="suspended-fourth-seventh";//??????????????
		}else{
			kind="unknown";
		}//endif
		bass_step="";
		bass_alter=0;
		if(!withBass){return;}
		bass_step=bassname.substr(0,1);
		for(int j=1;j<bassname.size();j+=1){
			if(bassname[j]=='#'){bass_alter+=1;}
			if(bassname[j]=='b'){bass_alter-=1;}
		}//endfor j
	}//end ReadFullname

	vector<XMLEvt> Output(){
		vector<XMLEvt> out;
		stringstream ss;
		out.push_back( XMLEvt(3,1,"harmony font-family=\"Arial\"") );
		out.push_back( XMLEvt(4,1,"root") );
		out.push_back( XMLEvt(5,1,"root-step") );
		out.push_back( XMLEvt(5,0,root_step) );
		out.push_back( XMLEvt(5,2,"/root-step") );
		if(root_alter!=0){
			out.push_back( XMLEvt(5,1,"root-alter") );
			ss.str(""); ss<<root_alter;
			out.push_back( XMLEvt(5,0,ss.str()) );
			out.push_back( XMLEvt(5,2,"/root-alter") );
		}//endif
		out.push_back( XMLEvt(4,2,"/root") );
		ss.str(""); ss<<"kind halign=\"left\" text=\""<<kind_text<<"\"";
		out.push_back( XMLEvt(4,1,ss.str()) );
		out.push_back( XMLEvt(4,0,kind) );
		out.push_back( XMLEvt(4,2,"/kind") );
		if(withBass){
			out.push_back( XMLEvt(4,1,"bass") );
			out.push_back( XMLEvt(5,1,"bass-step") );
			out.push_back( XMLEvt(5,0,bass_step) );
			out.push_back( XMLEvt(5,2,"/bass-step") );
			if(bass_alter!=0){
				out.push_back( XMLEvt(5,1,"bass-alter") );
				ss.str(""); ss<<bass_alter;
				out.push_back( XMLEvt(5,0,ss.str()) );
				out.push_back( XMLEvt(5,2,"/bass-alter") );
			}//endif
			out.push_back( XMLEvt(4,2,"/bass") );
		}//endif
		out.push_back( XMLEvt(3,2,"/harmony") );
		return out;
	}//end Output

};//endclass Harmony

class Attributes{
public:
	int divisions;
	int key_fifth;
	string key_mode;//major/minor
	int time_beats;
	int time_beat_type;
	string clef_sign;//def G
	int clef_line;//def 2

	vector<XMLEvt> Output(){
		vector<XMLEvt> out;
		stringstream ss;
		out.push_back( XMLEvt(3,1,"attributes") );
		out.push_back( XMLEvt(4,1,"divisions") );
		ss.str("");ss<<divisions;
		out.push_back( XMLEvt(4,0,ss.str()) );
		out.push_back( XMLEvt(4,2,"/divisions") );
		out.push_back( XMLEvt(4,1,"key") );
		out.push_back( XMLEvt(5,1,"fifths") );
		ss.str("");ss<<key_fifth;
		out.push_back( XMLEvt(5,0,ss.str()) );
		out.push_back( XMLEvt(5,2,"/fifths") );
		out.push_back( XMLEvt(5,1,"mode") );
		out.push_back( XMLEvt(5,0,key_mode) );
		out.push_back( XMLEvt(5,2,"/mode") );
		out.push_back( XMLEvt(4,2,"/key") );
		out.push_back( XMLEvt(4,1,"time") );
		out.push_back( XMLEvt(5,1,"beats") );
		ss.str("");ss<<time_beats;
		out.push_back( XMLEvt(5,0,ss.str()) );
		out.push_back( XMLEvt(5,2,"/beats") );
		out.push_back( XMLEvt(5,1,"beat-type") );
		ss.str("");ss<<time_beat_type;
		out.push_back( XMLEvt(5,0,ss.str()) );
		out.push_back( XMLEvt(5,2,"/beat-type") );
		out.push_back( XMLEvt(4,2,"/time") );
		out.push_back( XMLEvt(4,1,"clef") );
		out.push_back( XMLEvt(5,1,"sign") );
		out.push_back( XMLEvt(5,0,clef_sign) );
		out.push_back( XMLEvt(5,2,"/sign") );
		out.push_back( XMLEvt(5,1,"line") );
		ss.str("");ss<<clef_line;
		out.push_back( XMLEvt(5,0,ss.str()) );
		out.push_back( XMLEvt(5,2,"/line") );
		out.push_back( XMLEvt(4,2,"/clef") );
		out.push_back( XMLEvt(3,2,"/attributes") );
		return out;
	}//end Output

};//endclass Attributes

class TempoEvt{
public:
	double value;
	vector<XMLEvt> Output(){
		vector<XMLEvt> out;
		stringstream ss;
		ss.str("");
		ss<<"sound tempo=\""<<value<<"\"/";
		out.push_back( XMLEvt(3,2,ss.str()) );
		return out;
	}//end Output
};//endclass TempoEvt

class BackupForward{
public:
	int value;//forward if value >= 0 and backup if value < 0

	vector<XMLEvt> Output(){
		vector<XMLEvt> out;
		stringstream ss;
		if(value>=0){
			out.push_back( XMLEvt(3,1,"forward") );
			out.push_back( XMLEvt(4,1,"duration") );
			ss.str(""); ss<<value;
			out.push_back( XMLEvt(4,0,ss.str()) );
			out.push_back( XMLEvt(4,2,"/duration") );
			out.push_back( XMLEvt(3,2,"/forward") );
		}else{
			out.push_back( XMLEvt(3,1,"backup") );
			out.push_back( XMLEvt(4,1,"duration") );
			ss.str(""); ss<<(-1*value);
			out.push_back( XMLEvt(4,0,ss.str()) );
			out.push_back( XMLEvt(4,2,"/duration") );
			out.push_back( XMLEvt(3,2,"/backup") );
		}//endif
		return out;
	}//end Output
};//endclass BackupForward

class Measure{
public:
	string ID;//bar ID
	string barstyle;//at the start position, default=light 
	string sectionLabel;//at the start position

	vector<Attributes> attributes;
	vector<TempoEvt> tempos;
	vector<Harmony> harmonies;
	vector<Note> notes;
	vector<BackupForward> backupforwards;

	vector<string> types;//One of above five types Attributes/TempoEvt/Harmony/Note/BackupForward
	vector<int> pos;//pos inside each type
	vector<int> stimes;

	void Clear(){
		ID="-1";
		attributes.clear();
		tempos.clear();
		harmonies.clear();
		notes.clear();
		backupforwards.clear();
		types.clear();
		pos.clear();
		stimes.clear();
	}//end Clear

	void Print(){
		cout<<"***** measure "<<ID<<endl;
		for(int i=0;i<attributes.size();i+=1){
			cout<<"attributes "<<i<<endl;
			cout<<"divisions : "<<attributes[i].divisions<<endl;
			cout<<"key_fifth : "<<attributes[i].key_fifth<<endl;
			cout<<"key_mode : "<<attributes[i].key_mode<<endl;
			cout<<"time_beats : "<<attributes[i].time_beats<<endl;
			cout<<"time_beat_type : "<<attributes[i].time_beat_type<<endl;
			cout<<"clef_sign : "<<attributes[i].clef_sign<<endl;
			cout<<"clef_line : "<<attributes[i].clef_line<<endl;
		}//endfor i
	}//end Print


	vector<XMLEvt> Output(){
		vector<XMLEvt> out;
		stringstream ss;
		out.push_back( XMLEvt(1,4,"!--=========================================================--") );
		ss.str(""); ss<<"measure number=\""<<ID<<"\"";
		out.push_back( XMLEvt(2,1,ss.str()) );
		vector<XMLEvt> tmp;
		for(int i=0;i<types.size();i+=1){
			if(types[i]=="Attributes"){
				tmp=attributes[pos[i]].Output();
			}else if(types[i]=="TempoEvt"){
				tmp=tempos[pos[i]].Output();
			}else if(types[i]=="Harmony"){
				tmp=harmonies[pos[i]].Output();
			}else if(types[i]=="Note"){
				tmp=notes[pos[i]].Output();
			}else if(types[i]=="BackupForward"){
				tmp=backupforwards[pos[i]].Output();
			}//endif
			for(int j=0;j<tmp.size();j+=1){
				out.push_back(tmp[j]);
			}//endfor j
		}//endfor i
		out.push_back( XMLEvt(2,2,"/measure") );
		return out;
	}//end Output

};//endclass Measure

class Part{
public:
	vector<Measure> measures;

	void Print(){
	}//end Print

	vector<XMLEvt> Output(){
		vector<XMLEvt> out,tmp;
		out.push_back( XMLEvt(1,1,"part id=\"P1\"") );
		for(int i=0;i<measures.size();i+=1){
			tmp=measures[i].Output();
			for(int j=0;j<tmp.size();j+=1){
				out.push_back(tmp[j]);
			}//endfor j
		}//endfor i
		out.push_back( XMLEvt(1,2,"/part") );
		return out;
	}//end Output
};//endclass Part

class MusicXML{
public:
	XML xml;
	int TPQN;
	XMLEvt xmlversion;
	XMLEvt doctype;
	string movement_title;
	vector<XMLEvt> defaults;
	vector<XMLEvt> part_list;
	Part part;

	void Init(){

		xmlversion=XMLEvt(0,4,"?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?");
		doctype=XMLEvt(0,4,"!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 3.0 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\"");

		defaults.clear();
		defaults.push_back( XMLEvt(1,1,"defaults") );
		defaults.push_back( XMLEvt(2,1,"scaling") );
		defaults.push_back( XMLEvt(3,1,"millimeters") );
		defaults.push_back( XMLEvt(3,0,"5.5739") );
		defaults.push_back( XMLEvt(3,2,"/millimeters") );
		defaults.push_back( XMLEvt(3,1,"tenths") );
		defaults.push_back( XMLEvt(3,0,"40") );
		defaults.push_back( XMLEvt(3,2,"/tenths") );
		defaults.push_back( XMLEvt(2,2,"/scaling") );
		defaults.push_back( XMLEvt(2,3,"music-font font-family=\"Kousaku,engraved\" font-size=\"15.8\"/") );
		defaults.push_back( XMLEvt(1,2,"/defaults") );

		part_list.clear();
		part_list.push_back( XMLEvt(1,1,"part-list") );
		part_list.push_back( XMLEvt(2,1,"score-part id=\"P1\"") );
		part_list.push_back( XMLEvt(3,1,"part-name print-object=\"no\"") );
		part_list.push_back( XMLEvt(3,0,"MusicXML Part") );
		part_list.push_back( XMLEvt(3,2,"/part-name") );
		part_list.push_back( XMLEvt(3,1,"score-instrument id=\"P1-I1\"") );
		part_list.push_back( XMLEvt(4,1,"instrument-name") );
		part_list.push_back( XMLEvt(4,0,"ARIA Player") );
		part_list.push_back( XMLEvt(4,2,"/instrument-name") );
		part_list.push_back( XMLEvt(4,3,"virtual-instrument/") );
		part_list.push_back( XMLEvt(3,2,"/score-instrument") );
		part_list.push_back( XMLEvt(3,1,"midi-device") );
		part_list.push_back( XMLEvt(3,0,"ARIA Player") );
		part_list.push_back( XMLEvt(3,2,"/midi-device") );
		part_list.push_back( XMLEvt(3,1,"midi-instrument id=\"P1-I1\"") );
		part_list.push_back( XMLEvt(4,1,"midi-channel") );
		part_list.push_back( XMLEvt(4,0,"1") );
		part_list.push_back( XMLEvt(4,2,"/midi-channel") );
		part_list.push_back( XMLEvt(4,1,"midi-program") );
		part_list.push_back( XMLEvt(4,0,"1") );
		part_list.push_back( XMLEvt(4,2,"/midi-program") );
		part_list.push_back( XMLEvt(4,1,"volume") );
		part_list.push_back( XMLEvt(4,0,"80") );
		part_list.push_back( XMLEvt(4,2,"/volume") );
		part_list.push_back( XMLEvt(4,1,"pan") );
		part_list.push_back( XMLEvt(4,0,"0") );
		part_list.push_back( XMLEvt(4,2,"/pan") );
		part_list.push_back( XMLEvt(3,2,"/midi-instrument") );
		part_list.push_back( XMLEvt(2,2,"/score-part") );
		part_list.push_back( XMLEvt(1,2,"/part-list") );

	}//end Init

	void ReadFile(string filename){
		xml.ReadFile(filename);

{
		vector<int> divs;
		for(int i=0;i<xml.evts.size();i+=1){
			if(xml.evts[i].content=="divisions"){
				divs.push_back(atoi(xml.evts[i+1].content.c_str()));
			}//endif
		}//endfor i
		TPQN=divs[0];
		for(int i=1;i<divs.size();i+=1){
			TPQN=lcm(TPQN,divs[i]);
		}//endfor i
}//

		Measure measure;
		Attributes attribute;
		TempoEvt tempoEvt;
		Harmony harmony;
		Note note;
		BackupForward backupforward;
		bool directionIsForChordSymbol;
		vector<string> chopped;
		stringstream ss;
		string str;
		int curDiv=TPQN;
//		int curDiv=1;
		int curStime=0;
		bool notein=false;
		bool backupin=false;
		bool forwardin=false;
		string preBarStyle="light";

//		attribute.divisions=curDiv;
		attribute.divisions=TPQN;
		attribute.key_fifth=0;
		attribute.key_mode="major";
		attribute.time_beats=4;
		attribute.time_beat_type=4;
		attribute.clef_sign="G";
		attribute.clef_line=2;

		for(int i=0;i<xml.evts.size();i+=1){
//cout<<i<<endl;
			ss.clear(stringstream::goodbit);
			ss.str(xml.evts[i].content);
			chopped.clear();
//cout<<i<<"\t"<<ss.str()<<endl;
			while(ss>>str){chopped.push_back(str);}
			if(chopped.size()==0){continue;}

			if(chopped[0]=="movement-title"){
				movement_title=xml.evts[i+1].content;
			}else if(chopped[0]=="measure"){
				measure.Clear();
				measure.ID=chopped[1].substr(chopped[1].find("\"")+1,chopped[1].rfind("\"")-chopped[1].find("\"")-1);
				measure.barstyle=preBarStyle;
				measure.sectionLabel="";
				preBarStyle="light";
			}else if(chopped[0]=="bar-style"){
				preBarStyle=xml.evts[i+1].content;
			}else if(chopped[0]=="rehearsal"){
				measure.sectionLabel=xml.evts[i+1].content;
			}else if(chopped[0]=="/measure"){
				part.measures.push_back(measure);

			}else if(chopped[0]=="attributes"){

			}else if(chopped[0]=="divisions"){
				curDiv=atoi(xml.evts[i+1].content.c_str());
//				attribute.divisions=curDiv;
				attribute.divisions=TPQN;//=(curDiv*TPQN)/curDiv
			}else if(chopped[0]=="fifths"){
				attribute.key_fifth=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="mode"){
				attribute.key_mode=xml.evts[i+1].content;
			}else if(chopped[0]=="beats"){
				attribute.time_beats=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="beat-type"){
				attribute.time_beat_type=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="sign"){
				attribute.clef_sign=xml.evts[i+1].content;
			}else if(chopped[0]=="line"){
				attribute.clef_line=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="/attributes"){
				measure.types.push_back("Attributes");
				measure.pos.push_back(measure.attributes.size());
				measure.attributes.push_back(attribute);
				measure.stimes.push_back(curStime);

			}else if(chopped[0]=="sound"){
				if(chopped.size()>1){
					if(chopped[1].find("tempo")!=string::npos){
						tempoEvt.value=atof(chopped[1].substr(chopped[1].find("\"")+1,chopped[1].rfind("\"")-chopped[1].find("\"")-1).c_str());
						measure.types.push_back("TempoEvt");
						measure.pos.push_back(measure.tempos.size());
						measure.tempos.push_back(tempoEvt);
						measure.stimes.push_back(curStime);
					}//endif
				}//endif

			}else if(chopped[0]=="note"){
				notein=true;
				note.staff=-1;
				note.chord=false;
				note.dot=0;
				note.tie=0;
				note.pitch.alter=0;
				measure.stimes.push_back(curStime);
			}else if(chopped[0]=="rest/" || chopped[0]=="rest"){
				note.pitch.step="rest";
			}else if(chopped[0]=="step"){
				note.pitch.step=xml.evts[i+1].content;
			}else if(chopped[0]=="alter"){
				note.pitch.alter=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="octave"){
				note.pitch.octave=atoi(xml.evts[i+1].content.c_str());
			}else if(notein && chopped[0]=="duration"){
//				note.duration=atoi(xml.evts[i+1].content.c_str());
				note.duration=(atoi(xml.evts[i+1].content.c_str())*TPQN)/curDiv;
				curStime+=note.duration;
			}else if(chopped[0]=="voice"){
				note.voice=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="staff"){
				note.staff=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="type"){
				note.type=xml.evts[i+1].content;
			}else if(chopped[0]=="dot/"){
				note.dot+=1;
			}else if(chopped[0]=="chord/"){
				note.chord=true;
				measure.stimes[measure.stimes.size()-1]-=note.duration;
			}else if(chopped[0]=="tie"){
				if(chopped[1].find("start")!=string::npos){note.tie+=1;}
				if(chopped[1].find("stop")!=string::npos){note.tie+=2;}
			}else if(chopped[0]=="/note"){
				if(note.chord){curStime-=note.duration;}
				notein=false;
				note.pitch.WriteSitch();
				measure.types.push_back("Note");
				measure.pos.push_back(measure.notes.size());
				measure.notes.push_back(note);

			}else if(chopped[0]=="harmony"){
				harmony.withBass=false;
				harmony.offset=0;
				harmony.degree_value=-1;
				harmony.degree_alter=-1;
				harmony.degree_type="";
			}else if(chopped[0]=="root-step"){
				harmony.root_step=xml.evts[i+1].content;
				harmony.root_alter=0;
			}else if(chopped[0]=="root-alter"){
				harmony.root_alter=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="kind"){
				harmony.kind_text="";
				for(int j=1;j<chopped.size();j+=1){
					if(chopped[j].find("text=")!=string::npos){
						harmony.kind_text=chopped[j].substr(chopped[j].find("\"")+1,chopped[j].rfind("\"")-chopped[j].find("\"")-1);
					}//endif
				}//endfor j
				harmony.kind=xml.evts[i+1].content;
//cout<<measure.ID<<"\t"<<harmony.kind_text<<"\t"<<harmony.kind<<endl;
				if(harmony.kind_text=="" && harmony.kind!="major"){
					harmony.SetKindTextFromKind();
				}//endif
			}else if(chopped[0]=="bass-step"){
				harmony.withBass=true;
				harmony.bass_step=xml.evts[i+1].content;
				harmony.bass_alter=0;
			}else if(chopped[0]=="bass-alter"){
				harmony.bass_alter=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="offset"){
				harmony.offset=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="degree-value"){
				harmony.degree_value=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="degree-alter"){
				harmony.degree_alter=atoi(xml.evts[i+1].content.c_str());
			}else if(chopped[0]=="degree-type"){
				harmony.degree_type=xml.evts[i+1].content;
			}else if(chopped[0]=="/harmony"){

// if(harmony.degree_value>=0){
// cout<<harmony.degree_value<<"\t"<<harmony.degree_alter<<"\t"<<harmony.degree_type<<endl;
// }//endif
				if(harmony.degree_value==2){harmony.degree_value=9;}
				if(harmony.degree_value==4){harmony.degree_value=11;}

				harmony.NormalizeKindText();
				harmony.WriteFullname();
				measure.types.push_back("Harmony");
				measure.pos.push_back(measure.harmonies.size());
				measure.harmonies.push_back(harmony);
				measure.stimes.push_back(curStime+harmony.offset);

// 			}else if(chopped[0]=="direction"){//chord can be written as directions
// 				directionIsForChordSymbol=false;
// 			}else if(chopped[0]=="words"){//chord can be written as directions
// 				directionIsForChordSymbol=true;
// 				harmony.fullname=xml.evts[i+1].content;
// //				harmony.ReadFullname(); -> ERROR
// 			}else if(chopped[0]=="/direction"){
// 				if(directionIsForChordSymbol){
// 					measure.types.push_back("Harmony");
// 					measure.pos.push_back(measure.harmonies.size());
// 					measure.harmonies.push_back(harmony);
// 					measure.stimes.push_back(curStime);
// 				}//endif

			}else if(xml.evts[i].depth==3 && xml.evts[i].type==1 && chopped[0]=="backup"){
				backupin=true;
				measure.stimes.push_back(curStime);
			}else if(backupin && chopped[0]=="duration"){
//				curStime-=atoi(xml.evts[i+1].content.c_str());
//				backupforward.value=-1*(atoi(xml.evts[i+1].content.c_str()));
				curStime-=(atoi(xml.evts[i+1].content.c_str())*TPQN)/curDiv;
				backupforward.value=-1*(atoi(xml.evts[i+1].content.c_str())*TPQN)/curDiv;
			}else if(chopped[0]=="/backup"){
				backupin=false;
				measure.types.push_back("BackupForward");
				measure.pos.push_back(measure.backupforwards.size());
				measure.backupforwards.push_back(backupforward);

			}else if(xml.evts[i].depth==3 && xml.evts[i].type==1 && chopped[0]=="forward"){
				forwardin=true;
				measure.stimes.push_back(curStime);
			}else if(forwardin && chopped[0]=="duration"){
//				curStime+=atoi(xml.evts[i+1].content.c_str());
//				backupforward.value=atoi(xml.evts[i+1].content.c_str());
				curStime+=(atoi(xml.evts[i+1].content.c_str())*TPQN)/curDiv;
				backupforward.value=(atoi(xml.evts[i+1].content.c_str())*TPQN)/curDiv;
			}else if(chopped[0]=="/forward"){
				forwardin=false;
				measure.types.push_back("BackupForward");
				measure.pos.push_back(measure.backupforwards.size());
				measure.backupforwards.push_back(backupforward);
			}//endif

		}//endfor i

//		Init();
	}//end ReadFile

// 	void WriteFile(string filename){
// 		Init();
// 		xml.evts.clear();
// 		xml.evts.push_back(xmlversion);
// 		xml.evts.push_back(doctype);
// 		xml.evts.push_back( XMLEvt(0,1,"score-partwise version=\"3.0\"") );
// 		for(int i=0;i<defaults.size();i+=1){
// 			xml.evts.push_back(defaults[i]);
// 		}//endfor i
// 		for(int i=0;i<part_list.size();i+=1){
// 			xml.evts.push_back(part_list[i]);
// 		}//endfor i
// 		vector<XMLEvt> tmp=part.Output();
// 		for(int i=0;i<tmp.size();i+=1){
// 			xml.evts.push_back(tmp[i]);
// 		}//endfor i
// 		xml.evts.push_back( XMLEvt(0,2,"/score-partwise") );
// 
// 		xml.WriteFile(filename);
// 	}//end WriteFile

};//endclass MusicXML

#endif // MusicXML_HPP


