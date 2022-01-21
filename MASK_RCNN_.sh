#!/bin/sh
python maskrcnn_making_mask.py
python maskrcnn_making_testset.py
python maskrcnn_test.py
python maskrcnn_score.py
