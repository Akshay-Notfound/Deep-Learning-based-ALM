@echo off
echo Initializing Audio Language Model (ALM) Project...
echo =================================================

echo Setting up directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir checkpoints 2>nul
mkdir logs 2>nul
mkdir results 2>nul

echo Installing dependencies...
pip install -r requirements.txt

echo Generating sample dataset...
python init_project.py --samples 50

echo Project initialization complete!
echo.
echo To train the model, run:
echo   python training\train.py
echo.
echo To run inference, run:
echo   python main.py --checkpoint checkpoints\best_model.pt --audio path\to\audio.wav
echo.
pause