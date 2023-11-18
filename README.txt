This package includes data used for the following paper.

Kentaro Shibata, Eita Nakamura, Kazuyoshi Yoshii
Non-local musical statistics as guides for audio-to-score piano transcription


- JPopDataLink.txt

List of URLs used to collect the J-pop data.


- MuseScoreDataLink.txt

List of URLs used to collect the MuseScore data.


- Transcriptions

Transcribed results for the MAPS-ENSTDkCl dataset.
Performance MIDI outputs from POVNet are given as standard MIDI files (*.mid).
MusicXML outputs from the rhythm quantization method (*_merge.xml).


- POVNet

Source code for the multipitch detection method.
Note: It's not possible to run the programs by themselves.


- RQ

Source code for the rhythm quantization method.

To compile, run
./compile.sh
A c++ compiler is required.

For an input performance MIDI file (e.g. input.mid), run

./PerformanceMIDIToQuantizedMIDI.sh input
or
./PerformanceMIDIToQuantizedMIDI_NL.sh input

The file extension of the input must be ".mid".
The output (quantized MIDI sequence) is input_vs_qpr.mid
PerformanceMIDIToQuantizedMIDI.sh outputs the result before the application of the method of using non-local statistics.
PerformanceMIDIToQuantizedMIDI_NL.sh outputs the result after the application of the method of using non-local statistics.


For any inquiries, please contact Eita Nakamura (eita.nakamura@i.kyoto-u.ac.jp).

27 Oct 2020
