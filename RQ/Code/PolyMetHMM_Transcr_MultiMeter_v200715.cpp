#include<fstream>
#include<iostream>
#include<cmath>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include"stdio.h"
#include"stdlib.h"
#include"PolyMetHMM_v200715.hpp"
using namespace std;

int main(int argc, char** argv){
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;
	clock_t start, end;
	start = clock();

	if(argc<5){cout<<"Error in usage: $./this infile(mid/spr/ipr) outfile(mid/qpr/qipr) nModel param1.txt ... paramN.txt "<<endl; return -1;}//endif
	string inFile=string(argv[1]);
	string outFile=string(argv[2]);
	int nModel=atoi(argv[3]);//2 or 3
	vector<string> paramFiles;
	for(int i=0;i<nModel;i+=1){
		paramFiles.push_back( string(argv[4+i]) );
	}//endfor i
// 	string paramFile=string(argv[1]);

	if(inFile.find(".mid")==string::npos && inFile.find("spr.txt")==string::npos && inFile.find("ipr.txt")==string::npos){
cout<<"Input file format must be .mid spr.txt or ipr.txt"<<endl; return -1;
	}//endif

	if(outFile.find(".mid")==string::npos && outFile.find("qpr.txt")==string::npos && outFile.find("qipr.txt")==string::npos){
cout<<"Output file format must be .mid qpr.txt or qipr.txt"<<endl; return -1;
	}//endif

	PianoRoll pr;
	if(inFile.find(".mid")!=string::npos){
		pr.ReadMIDIFile(inFile);
	}else if(inFile.find("spr.txt")!=string::npos){
		pr.ReadFileSpr(inFile);
	}else if(inFile.find("ipr.txt")!=string::npos){
		pr.ReadFileIpr(inFile);
	}//endif

	vector<double> LPs(nModel);
	vector<QuantizedPianoRoll> results(nModel);

	for(int i=0;i<nModel;i+=1){
		PolyMetHMM model;
		model.ReadFile(paramFiles[i]);
		model.testData.push_back(pr);
		model.Transcribe();
		LPs[i]=model.maxLP[0];
		results[i]=model.estimatedData[0];
	}//endfor i

	int amax=0;
	for(int i=0;i<nModel;i+=1){
		if(LPs[i]>LPs[amax]){amax=i;}
	}//endfor i

cout<<"max: "<<amax<<"\t"<<paramFiles[amax]<<endl;

	if(outFile.find(".mid")!=string::npos){
		results[amax].WriteMIDIFile(outFile,0);
	}else if(outFile.find("qpr.txt")!=string::npos){
		results[amax].WriteFile(outFile,0);
	}else if(outFile.find("qipr.txt")!=string::npos){
		results[amax].WriteFile(outFile,1);
	}//endif

//	end = clock(); cout<<"Elapsed time : "<<((double)(end - start) / CLOCKS_PER_SEC)<<" sec"<<endl; start=end;
	return 0;
}//end main
