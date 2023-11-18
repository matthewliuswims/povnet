//g++ -O2 -I/Users/eita/Dropbox/Research/Tool/All/ QprHandSeparation_v190831.cpp -o QprHandSeparation
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<cassert>
#include "PianoFingeringDeterminationEngine_v170101_2.hpp"
//#include "QuantizedPianoRoll_v190822.hpp"
#include "QuantizedPianoRoll_v191003.hpp"
using namespace std;

int main(int argc, char** argv) {
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=3){cout<<"Error in usage: $./this in_qpr.txt out_qpr.txt"<<endl; return -1;}
	string inQprFile=string(argv[1]);
	string outQprFile=string(argv[2]);

	QuantizedPianoRoll qpr;
	qpr.ReadFile(inQprFile,0);
	qpr.RemoveRests();

	qpr.Sort();
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
	for(int n=0;n<qpr.evts.size();n+=1){
		qpr.evts[n].channel=fingering.evts[n].channel;
	}//endfor n

	qpr.WriteFile(outQprFile);

	return 0;
}//end main


