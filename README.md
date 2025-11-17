# Deep Learning based Audio Language Model (ALM)

## Overview
This project implements an Audio Language Model (ALM) capable of simultaneously recognizing and jointly understanding speech and non-speech audio elements with reasoning capabilities.

## Features
- Speech Recognition for Asian languages (Mandarin, Urdu, Hindi, Telugu, Tamil, Bangla) and English
- Non-Speech Audio Understanding (music, alarms, environmental noises)
- Speaker Diarization (differentiating between speakers)
- Paralinguistic Analysis (emotion, tone, hesitation)
- Audio Event Detection (car honking, dog barking, aircraft sounds, etc.)
- Joint understanding of speech and non-speech elements for complex reasoning

## Project Structure
```
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets.py
├── models/
│   ├── alm_model.py
│   ├── speech_encoder.py
│   ├── audio_encoder.py
│   └── fusion_module.py
├── training/
│   ├── train.py
│   └── trainer.py
├── utils/
│   ├── preprocessing.py
│   └── evaluation.py
├── config/
│   └── config.yaml
└── main.py
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers
- Librosa
- SoundFile
- NumPy
- Pandas
```