#!/bin/sh
python rad_preprocessing.py
python sat_preprocessing.py
python maskrcnn_making_mask.py
python maskrcnn_making_testset.py
python maskrcnn_test.py
python maskrcnn_score.py