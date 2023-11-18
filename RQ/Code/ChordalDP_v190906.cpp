//g++ -O2 -I/Users/eita/Dropbox/Research/Tool/All/ ChordalDP_v190906.cpp -o ChordalDP
#include<iostream>
#include<string>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<vector>
#include<fstream>
#include<cassert>
#include"ChordalDP_v190906.hpp"
using namespace std;

int main(int argc, char** argv) {

	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=5){cout<<"Error in usage! : $./this maxNumVoiceRH maxNumVoiceLH in_qpr.txt out_qpr.txt"<<endl; return -1;}
	int maxNumVoiceRH=atoi(argv[1]);
	int maxNumVoiceLH=atoi(argv[2]);
	string infile=string(argv[3]);
	string outfile=string(argv[4]);

	ChordalDP model;
	model.qpr.ReadFile(infile,0);
//cout<<model.qpr.evts.size()<<endl;
	model.DP(0,maxNumVoiceRH);
	model.DP(1,maxNumVoiceLH);
	model.qpr.WriteFile(outfile,0);

	return 0;
}//end main
