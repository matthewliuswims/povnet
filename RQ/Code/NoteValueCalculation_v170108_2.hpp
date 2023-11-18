#ifndef NOTEVALUECALCULATION_HPP
#define NOTEVALUECALCULATION_HPP

#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
#include<vector>
#include<fstream>
#include<cassert>
#include<algorithm>
#include"BasicCalculation_v170122.hpp"
using namespace std;

class IrredFrac{
public:
	int num;
	int den;
	double value;

	IrredFrac(){}//end IrredFrac
	IrredFrac(string str){
		if(str.find("/")==string::npos){
			den=1;
			num=atoi(str.c_str());
		}else{
			string numsymb=str.substr(0,str.find("/"));
			str=str.substr(str.find("/")+1);
			num=atoi(numsymb.c_str());
			den=atoi(str.c_str());
		}//endif
		value=double(num)/double(den);
	}//end IrredFrac
	IrredFrac(int num_,int den_){
		int sign=1;
		assert(den_!=0);
		if(num_==0){
			num=num_;
			den=1;
		}else{
			if(den_<0){
				sign*=-1;
				den_*=-1;
			}//endif
			if(num_<0){
				sign*=-1;
				num_*=-1;
			}//endif
			int fac=gcd(num_,den_);
			num=sign*(num_/fac);
			den=den_/fac;
		}//endif
		value=double(num)/double(den);
	}//end IrredFrac
	~IrredFrac(){}//end ~IrredFrac

	string Show(){
		stringstream ss;
		ss.str("");
		if(num==0||den==1){
			ss<<num;
		}else{
			ss<<num<<"/"<<den;
		}//endif
		return ss.str();
	}//end Show

	double Value(){
		if(den==0){return 0;}
		return double(num)/double(den);
	}//end Value

};//end class IrredFrac

	IrredFrac AddIrredFrac(IrredFrac irf1,IrredFrac irf2){
		return IrredFrac(irf2.den*irf1.num+irf1.den*irf2.num,irf1.den*irf2.den);
	}//end AddIrredFrac

	IrredFrac MultIrredFrac(IrredFrac irf1,IrredFrac irf2){
		return IrredFrac(irf1.num*irf2.num,irf1.den*irf2.den);
	}//end MultIrredFrac

	IrredFrac MinusIrredFrac(IrredFrac irf1){
		return IrredFrac(-1*irf1.num,irf1.den);
	}//end MinusIrredFrac

	IrredFrac InvIrredFrac(IrredFrac irf1){
		return IrredFrac(irf1.den,irf1.num);
	}//end InvIrredFrac

	IrredFrac SubtrIrredFrac(IrredFrac irf1,IrredFrac irf2){
		return AddIrredFrac(irf1,MinusIrredFrac(irf2));
	}//end SubtrIrredFrac

#endif // NOTEVALUECALCULATION_HPP

