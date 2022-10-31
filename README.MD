# Self-Supervised Hierarchical Metrical Analysis

This is the offical repository for the paper Deep Self-Supervised Hierarchical Metrical Structure Modeling under review for ICASSP 2023 (single-blinded).

![visualization.png](https://github.com/music-x-lab/Self-Supervised-Metrical-Structure/blob/main/screenshot/a.png)

## Resources

* Code and README
* Pre-trained models (download the models from [here](https://drive.google.com/drive/folders/1Er9CJkGLCOTd5ztCblOoGpRx3c3GNcg7?usp=sharing))
* Demos
* Data annotations (for evaluation only)
* More information on training data (to be added)
* Baseline models for reproduction (to be added)

## Demo

See ``output`` for the full results of the MIDI analysis model on RWC-POP.

* The file ends with ``_crf.mid`` is the results decoded by Conditional Random Field (CRF). Each output MIDI file contains an extra MIDI track with name Layers besides tracks from the original MIDI file. Layers is a drum track that labels L drum notes on a level-L boundary beyond measures (better viewed in an DAW with track-wise piano rolls). A downbeat without any drum notes is interpreted as a level-0 boundary.

* The file ends with ``_raw.mid`` is the raw output predicted by Temporal Convolutional Network (TCN) without CRF decoding. There are a total of L=8 midi tracks with names ``Layer-l`` for l=0...7 added to the MIDI file,  corresponding to the frame-wise probability of a metrical boundary of level at least (l + 1).

## Pretrained models

The pre-trained models are available [here](https://drive.google.com/drive/folders/1Er9CJkGLCOTd5ztCblOoGpRx3c3GNcg7?usp=sharing). Please download and extract all ``*.sdict`` files to the folder ``cache_data`` before continue.

## Run models on your own samples

Notice:

* you need correct beat (required) and downbeat (optional) labels for both MIDI and audio files.
* For MIDI files, the labels are automatically calculated based on tempo change and key signature events (as in most DAWs).
* For audio files, the labels need to be provided separately as a text file with format we will specify below.
* Downbeat labels are for hypermetrical structure analysis. If downbeat labels are wrong, the CRF-decoded results will be meaningless, but the raw TCN output is not influenced.
* Beat labels need to be accurete up to 32th-note level. Beat labels that fail to achieve this precision may cause degraded model performance. Notice that many automatic beat tracking models (e.g., madmom) might have precision issues.
* Currently, only 4/4 songs are supported.

### Run the MIDI model

For the MIDI model, run the following code to analyze ``example_input/midi_input/RM-P001.SMF_SYNC.MID``.

```cmd
python tcn_downbeat_eval.py example_input/midi_input/RM-P001.SMF_SYNC.MID
```

The program will produce two MIDI files ``*_crf.mid`` and ``*_raw.mid`` stored in the ``output`` folder. The file formats are already described above.

### Run the audio model

For the audio model, you also need to install [sonic visualizer](https://www.sonicvisualiser.org/) and add it to your environment variable.
Make sure your sonic visualizer folder is in your ``PATH`` environment variable. To check this, running the following command on your console should open sonic visualizer:
```cmd
"sonic visualiser"
```

To use pre-trained model on custom audio samples, you will need a beat label file like this:

```
0.04	1
0.49	2
0.93	3
1.37	4
1.82	1
2.26	2
2.71	3
3.15	4
3.60	1
4.04	2
...
```

For each row, the first number is the position of each beat (in second), and the second number is the downbeat label (1=downbeat). Two numbers are separated by tabs.

Then, run the following code:

```cmd
python tcn_downbeat_eval.py example_input/audio_input/RWC-POP-001-48kbps.mp3 example_input/audio_input/RWC-POP-001.lab
```

The program will call sonic visualizer (if correctly installed) to visualize the results. The results contain 3 layers:

* A beat-sync version of your music file (time-stretched according to your beat labels to ensure a constant BPM)
* A time instance layer showing hypermetrical structure analysis results (requires correct downbeat labels)
![image](https://user-images.githubusercontent.com/13694510/199000802-f76a11af-9cf6-4f67-ab44-6b8b564dbf9a.png)
* A spectrogram layer showing the raw prediction of the model on L=8 metrical layers.
![image](https://user-images.githubusercontent.com/13694510/199000769-c0815765-2eff-47eb-8d9d-78603289c718.png)

Notice: imprecise beat labels will likely cause low-confident predictions like this:
![image](https://user-images.githubusercontent.com/13694510/199000899-ba5e13cb-18be-421a-84e8-e26c5545a5a2.png)

## Credits

Credits to other repositories:
* Supervised  metrical Structure analysis from https://github.com/music-x-lab/Hierarchical-Metrical-Structure
* Osu parser adapted from https://github.com/Awlexus/python-osu-parser (used for data pre-processing)
