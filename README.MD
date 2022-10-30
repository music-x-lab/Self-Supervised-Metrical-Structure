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

The pre-trained models are available [here](https://drive.google.com/drive/folders/1Er9CJkGLCOTd5ztCblOoGpRx3c3GNcg7?usp=sharing). Please download and extract all ``*.pt`` files to the folder ``cache_data`` before continue.


## Run models on your own samples

TBA

## Credits

Credits to other repositories:
* Supervised  metrical Structure analysis from https://github.com/music-x-lab/Hierarchical-Metrical-Structure
* Osu parser adapted from https://github.com/Awlexus/python-osu-parser (used for data pre-processing)
