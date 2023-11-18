//g++ -O2 -I/Users/eita/Dropbox/Research/Tool/All/ QprToPianoVoiceMIDI_v190906.cpp -o QprToPianoVoiceMIDI
#include<iostream>
#include<string>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<vector>
#include<fstream>
#include<cassert>
//#include"QuantizedPianoRoll_v190822.hpp"
#include"QuantizedPianoRoll_v191003.hpp"
using namespace std;

int main(int argc, char** argv) {

	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=3){cout<<"Error in usage! : $./this in_qpr.txt out.mid"<<endl; return -1;}
//	int outType=atoi(argv[1]);
	string inFile=string(argv[1]);
	string outFile=string(argv[2]);

	if(inFile.find("qpr.txt")==string::npos && inFile.find("qipr.txt")==string::npos){
		cout<<"Input file formats: qpr.txt or qipr.txt"<<endl; return -1;
	}//endif

	QuantizedPianoRoll qpr;
	qpr.ReadFile(inFile,0);

	for(int n=0;n<qpr.evts.size();n+=1){
		bool unexpected=false;
		if(qpr.evts[n].channel==0){//RH
			if(qpr.evts[n].subvoice>=0 && qpr.evts[n].subvoice<4){
				
			}else{
				qpr.evts[n].subvoice=0;
				unexpected=true;
			}//endif
		}else if(qpr.evts[n].channel==1){//LH
			if(qpr.evts[n].subvoice>=0 && qpr.evts[n].subvoice<4){
				qpr.evts[n].subvoice+=4;
			}else{
				qpr.evts[n].subvoice=4;
				unexpected=true;
			}//endif
		}else{
			unexpected=true;
		}//endif
		if(unexpected){
cout<<"Unexpected channel or subvoice:\t"<<qpr.evts[n].channel<<"\t"<<qpr.evts[n].subvoice<<endl;
		}//endif
	}//endfor n

	qpr.WriteMIDIFile(outFile,1);

	return 0;
}//end main
