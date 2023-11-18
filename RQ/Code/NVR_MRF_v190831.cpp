//g++ -O2 -I/Users/eita/Dropbox/Research/Tool/All/ NVR_MRF_v190831.cpp -o NVR_MRF
#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
#include<vector>
#include<fstream>
#include<cassert>
#include "NVR_MRF_v190831.hpp"
using namespace std;

int main(int argc,char** argv){

	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=6){cout<<"Error in usage! : $./this treeFile InterDepIONVProbFile in_qpr.txt in_spr.txt out_qpr.txt"<<endl; return -1;}
	string treeFile=string(argv[1]);
	string InterDepIONVProbFile=string(argv[2]);
	string inQprFile=string(argv[3]);
	string inSprFile=string(argv[4]);
	string outFile=string(argv[5]);

	NVR_MRF mrf(InterDepIONVProbFile);

	ContextTree tree;
	tree.ReadFile(treeFile);
	mrf.SetContextTree(tree);

	QuantizedPianoRoll qpr;
	qpr.ReadFile(inQprFile,0);
	PianoRoll pr;
	pr.ReadFileSpr(inSprFile);

	mrf.ConstructTrData(qpr,pr);

// 	Trx trx;
// 	trx.ReadFile(inFile);
// 	mrf.SetTrData(trx);

	mrf.SetParameters(0.965,0.03,0.21,0.003,12);

	mrf.EstimateNVs();

	mrf.WriteQprFile(outFile);

//	mrf.WriteFile(outFile);//-> out_trx.txt

	return 0;
}//end main
