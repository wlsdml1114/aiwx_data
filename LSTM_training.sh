#!/bin/sh
python rad_preprocessing.py
python sat_preprocessing.py
python rad_ae.py
python sat_ae.py
python lstm_making_dataset.py
python LSTM_training.py