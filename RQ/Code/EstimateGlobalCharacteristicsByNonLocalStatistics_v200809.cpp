//g++ -O2 -I/Users/eita/boost_1_63_0 -I/Users/eita/Dropbox/Research/Tool/All/ EstimateGlobalCharacteristicsByNonLocalStatistics_v200809.cpp -o EstimateGlobalCharacteristicsByNonLocalStatistics
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
#include"BasicCalculation_v170122.hpp"
#include"QuantizedPianoRoll_v191003.hpp"
#include"QprAnalyzer_v200726.hpp"

using namespace std;

int main(int argc, char** argv) {
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	string str;
	stringstream ss;

	if(argc!=7){cout<<"Error in usage: $./this Res_AnalyzePieceProfile_MuseScoreData.txt param_QprAnalyzer_44Time.txt param_QprAnalyzer_34Time.txt 0110110000110010 in_qpr.txt out_qpr.txt"<<endl; return -1;}
	string refFile=string(argv[1]);
	string paramFile44=string(argv[2]);
	string paramFile34=string(argv[3]);
	string criterionVec=string(argv[4]);
	string infile=string(argv[5]);
	string outfile=string(argv[6]);

	///Tempo scale estimation
	vector<double> dataBPM,dataNVMean;
{
	ifstream ifs(refFile.c_str());
	while(ifs>>s[1]>>s[2]>>d[3]>>s[4]>>s[5]>>d[6]){
		dataBPM.push_back(d[3]);
		dataNVMean.push_back(d[6]);
		getline(ifs,s[99]);
	}//endwhile
	ifs.close();
}//

	double sigScale=0.01;

	QuantizedPianoRoll qpr(infile);
	qpr.ChangeTPQN(12);

	double BPM=qpr.spqnEvts[0].bpm;
	double NVMean;

	vector<double> lognv;
	for(int l=0;l<qpr.evts.size();l+=1){
		if(qpr.evts[l].pitch<0){continue;}
		lognv.push_back(log((4*qpr.TPQN)/(qpr.evts[l].offstime-qpr.evts[l].onstime)));
	}//endfor l
	NVMean=exp(Mean(lognv));

	double applyRescale=0;//0=no

	if(BPM<100){

		d[0]=0; d[1]=0;
		for(int n=0;n<dataBPM.size();n+=1){
			d[0]+=exp(-0.5*pow((log(dataBPM[n])-log(BPM))/sigScale,2.)-0.5*pow((log(dataNVMean[n])-log(NVMean))/sigScale,2.));
			d[1]+=exp(-0.5*pow((log(dataBPM[n])-log(2*BPM))/sigScale,2.)-0.5*pow((log(dataNVMean[n])-log(0.5*NVMean))/sigScale,2.));
		}//endfor n

		if(d[0]<d[1]){//double the tempo
			applyRescale=1;
//			rescale=0.5;
		}//endif

	}//endif

	if(applyRescale>0){
		for(int l=0;l<qpr.evts.size();l+=1){
			qpr.evts[l].onstime*=2;
			qpr.evts[l].offstime*=2;
		}//endfor l
		for(int i=0;i<qpr.meterEvts.size();i+=1){
			qpr.meterEvts[i].stime*=2;
		}//endfor i
		for(int i=0;i<qpr.spqnEvts.size();i+=1){
			qpr.spqnEvts[i].stime*=2;
			qpr.spqnEvts[i].value/=2;
			qpr.spqnEvts[i].bpm*=2;
		}//endfor i
	}//endif


	///Meter identification

	int width=qpr.meterEvts[0].barlen/qpr.TPQN;
	int nBeat=qpr.meterEvts[qpr.meterEvts.size()-1].stime/qpr.TPQN-width;

	vector<vector<vector<int> > > bpSetList;//[t][][0,1]=b,p, no double count
	vector<vector<vector<int> > > brSetList;//[t][][0,1]=b,r, no double count
	bpSetList.resize(nBeat);
	brSetList.resize(nBeat);
	vector<int> bp(2),br(2);

	for(int t=0;t<nBeat;t+=1){
		for(int l=0;l<qpr.evts.size();l+=1){
			if(qpr.evts[l].onstime<t*qpr.TPQN){continue;}
			if(qpr.evts[l].onstime>t*qpr.TPQN+width){break;}
			if(qpr.evts[l].pitch<0){continue;}
			br[0]=qpr.evts[l].onstime-t*qpr.TPQN;
			br[1]=qpr.evts[l].offstime-qpr.evts[l].onstime;
			if(find(brSetList[t].begin(),brSetList[t].end(),br)==brSetList[t].end()){brSetList[t].push_back(br);}

			int nv=qpr.evts[l].offstime-qpr.evts[l].onstime;
			for(int b=0;b<nv;b+=1){
				if(qpr.evts[l].onstime+b-t*qpr.TPQN<0 || qpr.evts[l].onstime+b-t*qpr.TPQN>=width){continue;}
				bp[0]=qpr.evts[l].onstime+b-t*qpr.TPQN;
				bp[1]=qpr.evts[l].pitch;
				if(find(bpSetList[t].begin(),bpSetList[t].end(),bp)==bpSetList[t].end()){bpSetList[t].push_back(bp);}
			}//endfor b
		}//endfor l
	}//endfor t

	double A4=0;
	double A3=0;
	d[4]=0; d[3]=0;
	for(int t=0;t<nBeat;t+=1){
		for(int tp=t+4;tp<=t+16;tp+=4){
			if(tp>=nBeat){continue;}
			A4+=0.5*(BPSetDist(bpSetList[t],bpSetList[tp])+BPSetDist(brSetList[t],brSetList[tp]));
			d[4]+=1;
		}//endfor tp
		for(int tp=t+3;tp<=t+12;tp+=3){
			if(tp>=nBeat){continue;}
			A3+=0.5*(BPSetDist(bpSetList[t],bpSetList[tp])+BPSetDist(brSetList[t],brSetList[tp]));
			d[3]+=1;
		}//endfor tp
	}//endfor t

	A4/=d[4];
	A3/=d[3];

	if(A3>A4){// -> 3/4 time
		qpr.meterEvts[0].num=3;
		qpr.meterEvts[0].den=4;
		qpr.meterEvts[0].barlen=36;
	}else{// -> 4/4 time
		qpr.meterEvts[0].num=4;
		qpr.meterEvts[0].den=4;
		qpr.meterEvts[0].barlen=48;
	}//endif

	int lastStime=0;
	for(int l=0;l<qpr.evts.size();l+=1){
		if(qpr.evts[l].offstime>lastStime){lastStime=qpr.evts[l].offstime;}
	}//endfor l
	if(lastStime%qpr.meterEvts[0].barlen!=0){
		lastStime=lastStime-lastStime%qpr.meterEvts[0].barlen+qpr.meterEvts[0].barlen;
	}//endif
	qpr.meterEvts[qpr.meterEvts.size()-1].stime=lastStime;

//	int lastStime=0;

	///Downbeat estimation

	int numStatistics=16;
	assert(criterionVec.size()==numStatistics);

	QprAnalyzer analyzer;
	if(qpr.meterEvts[0].barlen==48){
		analyzer.ReadParamFile(paramFile44);
	}else{
		analyzer.ReadParamFile(paramFile34);
	}//endif

	vector<double> mean(numStatistics),stdev(numStatistics);

	if(analyzer.nBeat==48){
		mean[0]=-1.2296; stdev[0]=0.248958;
		mean[1]=-1.64764; stdev[1]=0.255821;
		mean[2]=-1.55884; stdev[2]=0.342683;
		mean[3]=-1.39795; stdev[3]=0.3248;
		mean[4]=-1.44717; stdev[4]=0.323358;
		mean[5]=-1.34979; stdev[5]=0.451546;
		mean[6]=-0.0282186; stdev[6]=0.0350847;
		mean[7]=-0.0358236; stdev[7]=0.0419771;
		mean[8]=-0.0220646; stdev[8]=0.0438911;
		mean[9]=-2.00053; stdev[9]=0.136624;
		mean[10]=-1.98311; stdev[10]=0.154705;
		mean[11]=-1.98643; stdev[11]=0.16344;
		mean[12]=0.0320554; stdev[12]=0.0281241;
		mean[13]=0.0769246; stdev[13]=0.0305044;
		mean[14]=0.0659158; stdev[14]=0.038126;
		mean[15]=-2.24688; stdev[15]=0.0361323;
	}else if(analyzer.nBeat==36){
		mean[0]=-1.20593245; stdev[0]=0.31747679;
		mean[1]=-1.6002105; stdev[1]=0.32244041;
		mean[2]=-1.413264333; stdev[2]=0.354232698;
		mean[3]=-1.24362535; stdev[3]=0.274672449;
		mean[4]=-1.33315605; stdev[4]=0.28133911;
		mean[5]=-1.112776667; stdev[5]=0.336186793;
		mean[6]=-0.016869709; stdev[6]=0.022564939;
		mean[7]=-0.02216503; stdev[7]=0.032032551;
		mean[8]=-0.012856205; stdev[8]=0.026837094;
		mean[9]=-2.084700167; stdev[9]=0.116097424;
		mean[10]=-2.070524167; stdev[10]=0.114217662;
		mean[11]=-2.0819805; stdev[11]=0.135587083;
		mean[12]=0.026372718; stdev[12]=0.027084072;
		mean[13]=0.079142732; stdev[13]=0.034836909;
		mean[14]=0.058940068; stdev[14]=0.037918587;
		mean[15]=-2.235574167; stdev[15]=0.044230026;
	}else{
		cerr<<"analyzer.nBeat unknown! Run QprAnalyzer_BareAnalysisList to get statistics."<<endl;
		return -1;
	}//endif

	for(int k=0;k<numStatistics;k+=1){
		if(criterionVec[k]=='0'){
			stdev[k]=1E200;
		}//endif
	}//endfor k

	vector<double> sumIndex(4);

	for(int beatShift=0;beatShift<analyzer.nBeat/12;beatShift+=1){
	
		analyzer.LoadQpr(qpr,2*beatShift);
		analyzer.AnalyzeAll();

		vector<double> value(numStatistics);
		value[0]=analyzer.metLP;
		value[1]=analyzer.metLPRH;
		value[2]=analyzer.metLPLH;
		value[3]=analyzer.nvLP;
		value[4]=analyzer.nvLPRH;
		value[5]=analyzer.nvLPLH;
		value[6]=analyzer.tieRate;
		value[7]=analyzer.tieRateRH;
		value[8]=analyzer.tieRateLH;
		value[9]=analyzer.rpcLP;
		value[10]=analyzer.rpcLPRH;
		value[11]=analyzer.rpcLPLH;
		value[12]=analyzer.ssmContrast;
		value[13]=analyzer.ssmContrastRH;
		value[14]=analyzer.ssmContrastLH;
		value[15]=analyzer.pitchRankLP;

		sumIndex[beatShift]=0;
	
		for(int k=0;k<numStatistics;k+=1){
			sumIndex[beatShift]+=(value[k]-mean[k])/stdev[k];
		}//endfor k

	}//endfor beatShift

	int amax=0;
	for(int beatShift=0;beatShift<analyzer.nBeat/12;beatShift+=1){
		if(sumIndex[beatShift]>sumIndex[amax]){amax=beatShift;}
	}//endfor beatShift

//cout<<"amax:\t"<<amax<<"\t"<<((amax==0)? 0:1)<<endl;

	if(amax!=0){
		for(int l=0;l<qpr.evts.size();l+=1){
			qpr.evts[l].onstime+=amax*12;
			qpr.evts[l].offstime+=amax*12;
		}//endfor l
		lastStime=qpr.meterEvts[qpr.meterEvts.size()-1].stime+amax*12;
		if(lastStime%qpr.meterEvts[0].barlen!=0){
			lastStime=lastStime-lastStime%qpr.meterEvts[0].barlen+qpr.meterEvts[0].barlen;
		}//endif
		qpr.meterEvts[qpr.meterEvts.size()-1].stime=lastStime;
	}//endif

	qpr.WriteFile(outfile,0);

	return 0;

}//end main
