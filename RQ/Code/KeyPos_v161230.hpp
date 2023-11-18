#ifndef KEYPOS_HPP
#define KEYPOS_HPP

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

using namespace std;

struct KeyPos{
	int x;
	int y;
};

KeyPos SubtrKeyPos(KeyPos kp1, KeyPos kp2){
// interval from kp2 to kp1
	KeyPos keyPos;
	keyPos.x=kp1.x-kp2.x;
	keyPos.y=kp1.y-kp2.y;
	return keyPos;
}//

KeyPos AddKeyPos(KeyPos kp1, KeyPos kp2){
	KeyPos keyPos;
	keyPos.x=kp1.x+kp2.x;
	keyPos.y=kp1.y+kp2.y;
	return keyPos;
}//

KeyPos PitchToKeyPos(int pitch){
//Convention: C4=60=(0,0), D4=62=(1,0), Eb4=63=(1,1)
	int pc=pitch%12;
	int oct=pitch/12-1;
	KeyPos keyPos;
	if(pc==0||pc==1){keyPos.x=0;
	}else if(pc==2||pc==3){keyPos.x=1;
	}else if(pc==4){keyPos.x=2;
	}else if(pc==5||pc==6){keyPos.x=3;
	}else if(pc==7||pc==8){keyPos.x=4;
	}else if(pc==9||pc==10){keyPos.x=5;
	}else if(pc==11){keyPos.x=6;
	}//endif
	keyPos.x+=7*(oct-4);
	if(pc==0||pc==2||pc==4||pc==5||pc==7||pc==9||pc==11){keyPos.y=0;
	}else if(pc==1||pc==3||pc==6||pc==8||pc==10){keyPos.y=1;
	}//endif
	return keyPos;
};

int KeyPosToPitch(KeyPos keyPos){
	if(keyPos.x+70<0 || (keyPos.y!=0&&keyPos.y!=1) ){
		cout<<"undefined keyPos! ("<<keyPos.x<<","<<keyPos.y<<")"<<endl;
		assert(false);
	}//endif

	int oct,pc,xmod7;
	oct=(keyPos.x+70)/7-6;
	xmod7=(keyPos.x+70)%7;
	if(xmod7==0){pc=0;
	}else if(xmod7==1){pc=2;
	}else if(xmod7==2){pc=4;
	}else if(xmod7==3){pc=5;
	}else if(xmod7==4){pc=7;
	}else if(xmod7==5){pc=9;
	}else if(xmod7==6){pc=11;
	}//endif
	pc+=keyPos.y;
	return 12*(oct-4)+pc+60;
};

#endif // KEYPOS_HPP
