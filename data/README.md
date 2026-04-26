# Datasets

Two datasets are used. Neither is redistributed here.

## CREMA-D (emotion and emotion-aware transcription)

- Source: Cheyney Computer Science, https://github.com/CheyneyComputerScience/CREMA-D
- Contents: 7,442 clips from 91 actors, six emotions (neutral, happy, sad, anger, fear,
  disgust) plus a separate level of emotional intensity
- Access: free for research use, citation required
- Citation: Cao et al. (2014), "CREMA-D: Crowd-sourced Emotional Multimodal Actors
  Dataset", IEEE Transactions on Affective Computing

The notebook builds a manifest from the CREMA-D file names and loads audio lazily. Point
`DATA_ROOT` at the extracted `AudioWAV` folder.

## LibriSpeech (Conformer pretraining)

- Source: https://www.openslr.org/12
- Access: CC BY 4.0
- Used only for the from-scratch Conformer branch. The `train-clean-100` split is enough
  for the WER reported in the notebook.

## Model weights

`exports/whisper_lora_merged/` (merged Whisper-Small with LoRA adapters, plus
`emo_head.pt`, about 570 MB) is produced by the export cell at the end of the notebook and
is not committed.
