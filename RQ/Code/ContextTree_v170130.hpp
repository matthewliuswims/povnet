#ifndef ContextTree_HPP
#define ContextTree_HPP

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
#include<algorithm>
#include"BasicCalculation_v170122.hpp"

#define EPS (0.1)

using namespace std;

class SampleWithContext{
public:
	int val;//Assume val>=0
	vector<int> context;//Feature (discrete) vector c1,...,cF 

	SampleWithContext(){}//end SampleWithContext
	~SampleWithContext(){}//end ~SampleWithContext

};//endclass SampleWithContext

class Criterion{
public:
	int featureID;
	int splitValue;
	int belowOrAbove;//=0/1. If 0, feauter<=splitValue; if 1, feature>splitValue.
};//endclass Criterion

class ContextNode{
public:
	int depth;//Root = 0
	int parentID;
	int leftChildID;// -1 if N/A
	int rightChildID;// -1 of N/A
	Criterion splitCriterion;
//	int splitFeatureID;//-1 for leaves.
	vector<Criterion> criteria;
	Prob<int> prob;
	double logP;
	vector<int> sampleIDs;
	int dataSize;//=sampleIDs.size()

	ContextNode(){
		leftChildID=-1;
		rightChildID=-1;
//		splitFeatureID=-1;
	}//end ContextNode
	ContextNode(int depth_,int parentID_,int leftChildID_,int rightChildID_){
		depth=depth_;
		parentID=parentID_;
		leftChildID=leftChildID_;
		rightChildID=rightChildID_;
//		splitFeatureID=-1;
	}//end ContextNode
	~ContextNode(){}//end ~ContextNode

};//endclass ContextNode


class ContextTree{
public:
	vector<SampleWithContext> data;//samples
	vector<int> featureMins,featureMaxs;//Each feature c_f has featureMins[f]<=c_f<=featureMaxs[f]
	vector<ContextNode> nodes;
	vector<int> leafIDs;// Associated leaf for each sample
	int numFeatures;
	int sampleSpaceSize;//# of different samples;

	ContextTree(){}//end ContextTree
	ContextTree(vector<SampleWithContext> data_,vector<int> featureMins_,vector<int> featureMaxs_){
		Clear();
		data=data_;
		featureMins=featureMins_;
		featureMaxs=featureMaxs_;
		Init();
	}//end ContextTree
	~ContextTree(){}//end ~ContextTree

	void Clear(){
		nodes.clear();
		data.clear();
		leafIDs.clear();
	}//end Clear

	void ClearTreeStructure(){//Retaining data
		nodes.clear();
		leafIDs.clear();
		featureMins.clear();
		featureMaxs.clear();
	}//end ClearTreeStructure

	void WriteFile(string filename){
		ofstream ofs(filename.c_str());
		ofs<<"//numFeatures: "<<numFeatures<<"\n";
		ofs<<"//sampleSpaceSize: "<<sampleSpaceSize<<"\n";
		ofs<<"//nodes.size(): "<<nodes.size()<<"\n";
		ofs<<"########## Feature mins and maxs"<<"\n";
		for(int k=0;k<numFeatures;k+=1){
			ofs<<featureMins[k]<<"\t";
		}//endfor k
		ofs<<"\n";
		for(int k=0;k<numFeatures;k+=1){
			ofs<<featureMaxs[k]<<"\t";
		}//endfor k
		ofs<<"\n";
		ofs<<"########## Nodes: ID ( #samples ) depth parentID leftChildID rightChildID #criteria"<<"\n";
		for(int n=0;n<nodes.size();n+=1){
			ofs<<n<<"\t( "<<nodes[n].dataSize<<" )\t"<<nodes[n].depth<<"\t"<<nodes[n].parentID<<"\t"<<nodes[n].leftChildID<<"\t"<<nodes[n].rightChildID<<"\t"<<nodes[n].criteria.size()<<"\n";
			for(int k=0;k<nodes[n].criteria.size();k+=1){
				ofs<<nodes[n].criteria[k].featureID<<"\t"<<nodes[n].criteria[k].splitValue<<"\t"<<nodes[n].criteria[k].belowOrAbove<<"\n";
			}//endfor k
			for(int k=0;k<sampleSpaceSize;k+=1){
				ofs<<nodes[n].prob.P[k]<<"\t";
			}//endfor k
			ofs<<"\n";
		}//endfor n
		ofs.close();
	}//end WriteFile

	void ReadFile(string filename){
	/// Construct tree structure WITHOUT DATA
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);

		ClearTreeStructure();
		int nodeSize;
		ifstream ifs(filename.c_str());
		ifs>>s[0];
		if(s[0]!="//numFeatures:"){cout<<"Unexpected file format"<<endl; assert(false);}
		ifs>>numFeatures;
		getline(ifs,s[99]);
		ifs>>s[0];
		if(s[0]!="//sampleSpaceSize:"){cout<<"Unexpected file format"<<endl; assert(false);}
		ifs>>sampleSpaceSize;
		getline(ifs,s[99]);
		ifs>>s[0];
		if(s[0]!="//nodes.size():"){cout<<"Unexpected file format"<<endl; assert(false);}
		ifs>>nodeSize;
		getline(ifs,s[99]);

		getline(ifs,s[99]);//########## Feature mins and maxs
		featureMins.resize(numFeatures);
		featureMaxs.resize(numFeatures);
		for(int k=0;k<numFeatures;k+=1){
			ifs>>featureMins[k];
		}//endfor k
		getline(ifs,s[99]);
		for(int k=0;k<numFeatures;k+=1){
			ifs>>featureMaxs[k];
		}//endfor k
		getline(ifs,s[99]);

		getline(ifs,s[99]);//########## Nodes: ID
		for(int n=0;n<nodeSize;n+=1){
			ContextNode node;
			int nodeID;
			int numCriteria;
			ifs>>nodeID>>s[1]>>node.dataSize>>s[3]>>node.depth>>node.parentID>>node.leftChildID>>node.rightChildID>>numCriteria;
			getline(ifs,s[99]);
			Criterion criterion;
			for(int i=0;i<numCriteria;i+=1){
				ifs>>criterion.featureID>>criterion.splitValue>>criterion.belowOrAbove;
				node.criteria.push_back(criterion);
				getline(ifs,s[99]);
			}//endfor i
			node.prob.Resize(sampleSpaceSize);
			for(int i=0;i<sampleSpaceSize;i+=1){
				ifs>>node.prob.P[i];
			}//endfor i
			node.prob.PToLP();
			getline(ifs,s[99]);
			nodes.push_back(node);
		}//endfor n

		ifs.close();

		for(int n=0;n<nodeSize;n+=1){
			if(nodes[n].leftChildID==-1){continue;}
			nodes[n].splitCriterion=nodes[nodes[n].rightChildID].criteria[nodes[nodes[n].rightChildID].criteria.size()-1];
		}//endfor n

	}//end ReadFile

	void InputData(vector<SampleWithContext> data_){
//leafIDs,data
//sampleIDs,logP,
	}//end InputData


	Prob<int> GetProb(vector<int> &sampleIDs_,int sampleSpaceSize_){
		Prob<int> prob;
		prob.P.assign(sampleSpaceSize_,EPS);
		for(int k=0;k<sampleIDs_.size();k+=1){
			prob.P[data[sampleIDs_[k]].val]+=1;
		}//endfor k
		prob.Normalize();
		return prob;
	}//end GetProb

	void SetLogP(ContextNode& node){
		node.logP=0;
		for(int i=0;i<node.sampleIDs.size();i+=1){
			node.logP+=node.prob.LP[data[node.sampleIDs[i]].val];
		}//endfor i
	}//end SetLogP

	void Init(){//Set up root node
//cout<<"################ Enter Init ################"<<endl;
		if(data.size()==0){cout<<"No data. Not Initialised."<<endl; return;}
		numFeatures=data[0].context.size();
		sampleSpaceSize=0;
		for(int i=0;i<data.size();i+=1){
			if(data[i].val+1>sampleSpaceSize){sampleSpaceSize=data[i].val+1;}
		}//endfor i

//cout<<"numFeatures: "<<numFeatures<<"\tsampleSpaceSize: "<<sampleSpaceSize<<endl;

		ContextNode node(0,-1,-1,-1);

		for(int i=0;i<data.size();i+=1){
			node.sampleIDs.push_back(i);
		}//endfor i

		node.criteria.clear();
		node.prob=GetProb(node.sampleIDs,sampleSpaceSize);
		SetLogP(node);

		node.dataSize=node.sampleIDs.size();
		nodes.push_back(node);
		leafIDs.assign(data.size(),0);
//cout<<"################ Exit Init ################"<<endl;
	}//end Init

	bool CheckCriterion(vector<int>& context,Criterion& criterion){
		bool satisfies=true;
		if(criterion.belowOrAbove==0){//Should be below
			if(context[criterion.featureID]>criterion.splitValue){
				satisfies=false;
			}//endif
		}else{//Should be above
			if(context[criterion.featureID]<=criterion.splitValue){
				satisfies=false;
			}//endif
		}//endif
		return satisfies;
	}//end CheckCriterion

	bool CheckCriteria(vector<int>& context,vector<Criterion>& criteria){
		bool satisfies=true;
		for(int i=0;i<criteria.size();i+=1){
			satisfies=CheckCriterion(context,criteria[i]);
			if(!satisfies){break;}
		}//endfor i
		return satisfies;
	}//end CheckCriteria

	vector<vector<int> > SeparateSampleIDs(vector<int>& sampleIDs,vector<Criterion>& criteria){
		vector<vector<int> > separatedSampleIDs(2);
		//separatedSampleIDs[0]: samples not satisfying the criterion
		//separatedSampleIDs[1]: samples satisfying the criterion
		for(int i=0;i<sampleIDs.size();i+=1){
			if(CheckCriteria(data[sampleIDs[i]].context,criteria)){
				separatedSampleIDs[1].push_back(sampleIDs[i]);
			}else{
				separatedSampleIDs[0].push_back(sampleIDs[i]);
			}//endif
		}//endfor i
		return separatedSampleIDs;
	}//end SeparateSampleIDs

	vector<vector<int> > SeparateSampleIDs(vector<int>& sampleIDs,Criterion& criterion){
		vector<vector<int> > separatedSampleIDs(2);
		//separatedSampleIDs[0]: samples not satisfying the criterion
		//separatedSampleIDs[1]: samples satisfying the criterion
		for(int i=0;i<sampleIDs.size();i+=1){
			if(CheckCriterion(data[sampleIDs[i]].context,criterion)){
				separatedSampleIDs[1].push_back(sampleIDs[i]);
			}else{
				separatedSampleIDs[0].push_back(sampleIDs[i]);
			}//endif
		}//endfor i
		return separatedSampleIDs;
	}//end SeparateSampleIDs


	double ExpandNode(int nodeID){

		if(nodes[nodeID].leftChildID!=-1 || nodes[nodeID].rightChildID!=-1){
			cout<<"Cannot expand a node with a child"<<endl;
			return -1;
		}//endif

		vector<Criterion> candidateCriteria;
{
		Criterion criterion;
		int min,max;
		for(int k=0;k<numFeatures;k+=1){
			min=featureMins[k];
			max=featureMaxs[k]-1;
			for(int j=0;j<nodes[nodeID].criteria.size();j+=1){
				if(nodes[nodeID].criteria[j].featureID!=k){continue;}
				if(nodes[nodeID].criteria[j].belowOrAbove==0){
					max=nodes[nodeID].criteria[j].splitValue-1;
				}else{
					min=nodes[nodeID].criteria[j].splitValue+1;
				}//endif
			}//endfor j
			for(int j=min;j<=max;j+=1){
				criterion.featureID=k;
				criterion.splitValue=j;
				criterion.belowOrAbove=1;
				candidateCriteria.push_back(criterion);
			}//endfor j
		}//endfor k

		vector<vector<int> > separatedSampleIDs;
		for(int k=candidateCriteria.size()-1;k>=0;k-=1){
			separatedSampleIDs=SeparateSampleIDs(nodes[nodeID].sampleIDs,candidateCriteria[k]);
			if(separatedSampleIDs[0].size()<100 || separatedSampleIDs[1].size()<100){
				candidateCriteria.erase(candidateCriteria.begin()+k);
			}//endif
		}//endfor k

}//

		if(candidateCriteria.size()==0){
			cout<<"No candidate feature to split"<<endl;
			return -1;
		}//endif
//cout<<"candidateCriteria.size() : "<<candidateCriteria.size()<<endl;

		vector<ContextNode> amax_nodes;
		double maxDiffLogP;
		vector<vector<int> > separatedSampleIDs;

		for(int k=0;k<candidateCriteria.size();k+=1){

			vector<ContextNode> tmp_nodes;
			Criterion tmp_criterion;
			tmp_nodes=nodes;
			ContextNode nodeL,nodeR;
			nodeL=tmp_nodes[nodeID];//0(no)
			nodeR=tmp_nodes[nodeID];//1(yes)
			nodeL.depth+=1;
			nodeR.depth+=1;
			nodeL.parentID=nodeID;
			nodeR.parentID=nodeID;

			tmp_criterion=candidateCriteria[k];
			nodeR.criteria.push_back(tmp_criterion);
			tmp_criterion.belowOrAbove=0;
			nodeL.criteria.push_back(tmp_criterion);
			tmp_criterion.belowOrAbove=1;

			separatedSampleIDs=SeparateSampleIDs(tmp_nodes[nodeID].sampleIDs,tmp_criterion);
			nodeL.sampleIDs=separatedSampleIDs[0];
			nodeR.sampleIDs=separatedSampleIDs[1];

			nodeL.prob=GetProb(nodeL.sampleIDs,sampleSpaceSize);
			nodeR.prob=GetProb(nodeR.sampleIDs,sampleSpaceSize);

			SetLogP(nodeL);
			SetLogP(nodeR);

			for(int n=0;n<tmp_nodes.size();n+=1){
				if(tmp_nodes[n].parentID>nodeID){tmp_nodes[n].parentID+=2;}
				if(tmp_nodes[n].leftChildID>nodeID){tmp_nodes[n].leftChildID+=2;}
				if(tmp_nodes[n].rightChildID>nodeID){tmp_nodes[n].rightChildID+=2;}
			}//endfor n

			nodeL.dataSize=nodeL.sampleIDs.size();
			nodeR.dataSize=nodeR.sampleIDs.size();
			tmp_nodes.insert(tmp_nodes.begin()+nodeID+1,nodeL);
			tmp_nodes.insert(tmp_nodes.begin()+nodeID+2,nodeR);
			tmp_nodes[nodeID].leftChildID=nodeID+1;
			tmp_nodes[nodeID].rightChildID=nodeID+2;
			tmp_nodes[nodeID].splitCriterion=tmp_criterion;

			double diffLogP=nodeL.logP+nodeR.logP-nodes[nodeID].logP;
			if(k==0){
				amax_nodes=tmp_nodes;
				maxDiffLogP=diffLogP;
			}else if(diffLogP>maxDiffLogP){
				maxDiffLogP=diffLogP;
				amax_nodes=tmp_nodes;
			}//endif

		}//endfor k

		nodes=amax_nodes;
		for(int i=0;i<nodes[nodeID+1].sampleIDs.size();i+=1){
			leafIDs[nodes[nodeID+1].sampleIDs[i]]=nodeID+1;
		}//endfor i
		for(int i=0;i<nodes[nodeID+2].sampleIDs.size();i+=1){
			leafIDs[nodes[nodeID+2].sampleIDs[i]]=nodeID+2;
		}//endfor i

		return maxDiffLogP;

	}//end ExpandNode

	double ExpandTree(){
		ContextTree preTree;
		preTree=*this;
		vector<int> leafs;
		for(int n=0;n<nodes.size();n+=1){
			if(nodes[n].leftChildID!=-1){continue;}
			leafs.push_back(n);
		}//endfor n

		vector<double> diffLogPs(leafs.size());
		for(int l=0;l<leafs.size();l+=1){
			*this=preTree;
			diffLogPs[l]=ExpandNode(leafs[l]);
		}//endfor l

		double max=diffLogPs[0];
		int amax=0;
		for(int l=0;l<leafs.size();l+=1){
			if(diffLogPs[l]>max){max=diffLogPs[l]; amax=l;}
		}//endfor l

		*this=preTree;
		ExpandNode(leafs[amax]);
		return max;
	}//end ExpandTree

	void ExpandTree_MDL(int maxIter=10){
		int count=0;
		while(count<maxIter){
cout<<"Now iter "<<count<<endl;
			ContextTree preTree;
			preTree=*this;
			vector<int> leafs;
			for(int n=0;n<nodes.size();n+=1){
				if(nodes[n].leftChildID!=-1){continue;}
				leafs.push_back(n);
			}//endfor n

			vector<double> diffLogPs(leafs.size());
			for(int l=0;l<leafs.size();l+=1){
				*this=preTree;
				diffLogPs[l]=ExpandNode(leafs[l]);
			}//endfor l

			double max=diffLogPs[0];
			int amax=0;
			for(int l=0;l<leafs.size();l+=1){
				if(diffLogPs[l]>max){max=diffLogPs[l]; amax=l;}
			}//endfor l

			*this=preTree;

			if(-diffLogPs[amax]+0.5*(sampleSpaceSize-1)*log(data.size())<0){
cout<<"Diff DL "<<-diffLogPs[amax]+0.5*(sampleSpaceSize-1)*log(data.size())<<endl;
				ExpandNode(leafs[amax]);
			}else{//No further expansion
cout<<"Iteration stopped by the MDL criterion"<<endl;
				break;
			}//endif
			count+=1;
		}//endwhile
	}//end ExpandTree_MDL


	void WriteSVG(string filename){
		vector<double> x,y;//Coordinates for centres of nodes
		x.assign(nodes.size(),-10);
		y.assign(nodes.size(),-10);

		double DeltaX=170;
		double DeltaY=200;
		double maxX=0;
		double maxY=0;

		vector<int> cand;
		for(int n=0;n<nodes.size();n+=1){
			if(nodes[n].leftChildID!=-1){continue;}
			cand.push_back(n);
		}//endfor n
		for(int i=0;i<cand.size();i+=1){
			x[cand[i]]=i*DeltaX+DeltaX;
			if(x[cand[i]]>maxX){maxX=x[cand[i]];}
		}//endfor i

		while(true){
			cand.clear();
			for(int n=0;n<nodes.size();n+=1){
				if(x[n]>-1){continue;}
				cand.push_back(n);
			}//endfor n
			if(cand.size()==0){break;}

			for(int i=0;i<cand.size();i+=1){
				if(x[nodes[cand[i]].leftChildID]<-1 || x[nodes[cand[i]].rightChildID]<-1){continue;}
				x[cand[i]]=0.5*(x[nodes[cand[i]].leftChildID]+x[nodes[cand[i]].rightChildID]);
			}//endfor i
		}//endwhile

		for(int n=0;n<nodes.size();n+=1){
			y[n]=DeltaY*nodes[n].depth+DeltaY;
			if(y[n]>maxY){maxY=y[n];}
		}//endfor n

		double width=1200;
		double height=750;
		double globalScale=width/(maxX+1*DeltaX);
		if(height/(maxY+1*DeltaY)<globalScale){globalScale=height/(maxY+1*DeltaY);}

		for(int n=0;n<nodes.size();n+=1){
			x[n]*=globalScale;
			y[n]*=globalScale;
		}//endfor n

		DeltaX*=globalScale;
		DeltaY*=globalScale;

		double DeltaXDistr=DeltaX/1.2;
		double DeltaYDistr=DeltaY/1.5;
		double BoxSizeX=DeltaXDistr*1.1;
		double BoxSizeY=DeltaYDistr*1.;
		double scale=DeltaYDistr*0.85;
		double delta=DeltaXDistr/double(sampleSpaceSize);
		double offsetDistrX=-0.5*DeltaXDistr;
		double offsetDistrY=0.35*DeltaYDistr;

		double fontSize1=15.0*globalScale;
		double fontSize2=20.0*globalScale;
//		double fontSize3=1.5*delta;
		double fontSize3=1.*delta;
		double lineWidth=2.*globalScale;

		int allDataSize=nodes[0].dataSize;

		ofstream ofs(filename.c_str());
ofs<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"<<"\n";
ofs<<"<svg version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" width=\""<<width<<"\" height=\""<<height<<"\">"<<"\n";
ofs<<"\n";

		for(int n=0;n<nodes.size();n+=1){

			if(nodes[n].leftChildID!=-1){//Internal node

ofs<<"<line x1=\""<<x[n]<<"\" x2=\""<<x[nodes[n].leftChildID]<<"\" y1=\""<<y[n]<<"\" y2=\""<<y[nodes[n].leftChildID]<<"\" stroke=\"rgb(160,160,160)\" fill=\"transparent\" stroke-width=\""<<lineWidth<<"\"/>"<<"\n";
ofs<<"<line x1=\""<<x[n]<<"\" x2=\""<<x[nodes[n].rightChildID]<<"\" y1=\""<<y[n]<<"\" y2=\""<<y[nodes[n].rightChildID]<<"\" stroke=\"rgb(160,160,160)\" fill=\"transparent\" stroke-width=\""<<lineWidth<<"\"/>"<<"\n";

ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]+0.3*BoxSizeX<<","<<y[n]+0.5*BoxSizeY+2*fontSize2<<")\" font-size=\""<<fontSize1<<"\" fill=\"black\">"<<"Yes"<<"</text>"<<"\n";
ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]-0.3*BoxSizeX<<","<<y[n]+0.5*BoxSizeY+2*fontSize2<<")\" font-size=\""<<fontSize1<<"\" fill=\"black\">"<<"No"<<"</text>"<<"\n";
				if(featureMins[nodes[n].splitCriterion.featureID]==0&&featureMaxs[nodes[n].splitCriterion.featureID]==1){//If binary categorical feature
ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]<<","<<y[n]+0.5*BoxSizeY+fontSize2<<")\" font-size=\""<<fontSize2<<"\" fill=\"black\">c("<<nodes[n].splitCriterion.featureID+1<<") = True ?</text>"<<"\n";
				}else{//Dimensional feature
ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]<<","<<y[n]+0.5*BoxSizeY+fontSize2<<")\" font-size=\""<<fontSize2<<"\" fill=\"black\">c("<<nodes[n].splitCriterion.featureID+1<<") > "<<nodes[n].splitCriterion.splitValue<<" ?</text>"<<"\n";
				}//endif

			}//endif

ofs<<"<rect x=\""<<x[n]-0.5*BoxSizeX<<"\" y=\""<<y[n]-0.5*BoxSizeY<<"\" width=\""<<BoxSizeX<<"\" height=\""<<BoxSizeY<<"\" opacity=\""<<1.<<"\" fill=\""<<"rgb(255,255,255)"<<"\" stroke=\"rgb(30,120,255)\" stroke-width=\""<<lineWidth<<"\"/>"<<"\n";

			///Draw distributions
			for(int j=0;j<sampleSpaceSize;j+=1){
ofs<<"<rect x=\""<<x[n]+j*delta+offsetDistrX<<"\" y=\""<<y[n]-nodes[n].prob.P[j]*scale+offsetDistrY<<"\" width=\""<<0.8*delta<<"\" height=\""<<nodes[n].prob.P[j]*scale<<"\" opacity=\""<<1.<<"\" fill=\""<<"rgb(255,30,120)"<<"\"/>"<<"\n";
//				if(j%5==0 && j>0){
				if(true){
ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]+(j+0.5)*delta+offsetDistrX<<","<<y[n]+offsetDistrY+fontSize3<<")\" font-size=\""<<fontSize3<<"\" fill=\"black\">"<<j<<"</text>"<<"\n";
				}//endif
			}//
ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]+0.5*sampleSpaceSize*delta+offsetDistrX<<","<<y[n]-0.95*scale+offsetDistrY+fontSize3<<")\" font-size=\""<<fontSize3<<"\" fill=\"rgb(30,180,30)\">"<<0.001*int(1000*nodes[n].prob.MaxP())<<"</text>"<<"\n";

ofs<<"<text text-anchor=\"middle\" transform=\"translate("<<x[n]<<","<<y[n]-0.5*BoxSizeY-0.5*fontSize1<<")\" font-size=\""<<fontSize1<<"\" fill=\"black\">["<<n<<"] "<<nodes[n].dataSize<<" ("<<0.01*int(10000.*nodes[n].dataSize/double(allDataSize))<<"%)</text>"<<"\n";

		}//endfor n

ofs<<"\n";
ofs<<"</svg>"<<"\n";
		ofs.close();

	}//end WriteSVG


	int FindLeafID(vector<int> &context){
		int curNodeID=0;
		int nextNodID;
		while(true){
			if(nodes[curNodeID].leftChildID==-1){break;}
			nextNodID=nodes[curNodeID].leftChildID;
			if(CheckCriterion(context,nodes[curNodeID].splitCriterion)){
				nextNodID=nodes[curNodeID].rightChildID;
			}//endif
			curNodeID=nextNodID;
		}//endwhile
		return curNodeID;
	}//end FindLeafID



};//endclass ContextTree


#endif // ContextTree_HPP

