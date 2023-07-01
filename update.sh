#!/bin/sh
git pull origin main
. venv/bin/activate
pause
python -m pip uninstall calibrator dataloader
python -m pip install git+https://github.com/PlusF/Calibrator git+https://github.com/PlusF/DataLoader

