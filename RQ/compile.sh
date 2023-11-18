#! /bin/bash

mkdir ./Binary

g++ -O2 Code/PolyMetHMM_Transcr_MultiMeter_v200715.cpp -o ./Binary/PolyMetHMM_Transcr_MultiMeter

g++ -O2 Code/PolyMetHMM_Transcr_MultiMeter_LooseTempo_v200715.cpp -o ./Binary/PolyMetHMM_Transcr_MultiMeter_LooseTempo

g++ -O2 Code/midi2pianoroll_v170504.cpp -o ./Binary/midi2pianoroll

g++ -O2 Code/EstimateGlobalCharacteristicsByNonLocalStatistics_v200809.cpp -o ./Binary/EstimateGlobalCharacteristicsByNonLocalStatistics

g++ -O2 Code/NVR_MRF_v190831.cpp -o ./Binary/NVR_MRF

g++ -O2 Code/QprHandSeparation_v190831.cpp -o ./Binary/QprHandSeparation

g++ -O2 Code/ChordalDP_v190906.cpp -o ./Binary/ChordalDP

g++ -O2 Code/QprToPianoVoiceMIDI_v190906.cpp -o ./Binary/QprToPianoVoiceMIDI

