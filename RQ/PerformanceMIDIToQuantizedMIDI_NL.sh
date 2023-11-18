#! /bin/bash

if [ $# -ne 1 ]; then
echo "Error in usage: $./MIDIToMIDIAlign.sh input(.mid)"
exit 1
fi

Filename=$1

./Binary/midi2pianoroll 2 ${Filename}

#Classical music condition
#./Binary/PolyMetHMM_Transcr_MultiMeter_LooseTempo test_spr.txt out_qpr.txt 3 Code/param_PolyMetHMM_44time_TrainData_TASLP1.txt Code/param_PolyMetHMM_34time_TrainData_TASLP1.txt Code/param_PolyMetHMM_24time_TrainData_TASLP1.txt

#Popular music condition
./Binary/PolyMetHMM_Transcr_MultiMeter ${Filename}_spr.txt ${Filename}_qpr.txt 2 Code/param_PolyMetHMM_44time_MuseScoreData.txt Code/param_PolyMetHMM_34time_MuseScoreData.txt

#The following steps are applied for both classical music and popular music conditions

./Binary/NVR_MRF ./Code/tree_IONVDistr_MLFull.txt ./Code/InterDepIONVProb.txt ${Filename}_qpr.txt ${Filename}_spr.txt ${Filename}_off_qpr.txt

./Binary/EstimateGlobalCharacteristicsByNonLocalStatistics ./Code/Res_AnalyzePieceProfile_MuseScoreData.txt ./Code/param_QprAnalyzer_Complete44Time_MuseScoreData.txt ./Code/param_QprAnalyzer_Complete34Time_MuseScoreData.txt 0110110000110010 ${Filename}_off_qpr.txt ${Filename}_corr_qpr.txt

./Binary/QprHandSeparation ${Filename}_corr_qpr.txt ${Filename}_hs_qpr.txt

./Binary/ChordalDP 2 2 ${Filename}_hs_qpr.txt ${Filename}_vs_qpr.txt

./Binary/QprToPianoVoiceMIDI ${Filename}_vs_qpr.txt ${Filename}_vs_qpr.mid


rm ${Filename}_spr.txt
rm ${Filename}_qpr.txt
rm ${Filename}_off_qpr.txt
rm ${Filename}_corr_qpr.txt
rm ${Filename}_hs_qpr.txt
rm ${Filename}_vs_qpr.txt


