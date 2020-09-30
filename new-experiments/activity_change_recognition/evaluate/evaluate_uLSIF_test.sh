#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder ulSIF --offset 1 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder ulSIF --offset 5 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder ulSIF --offset 10 --train_or_test test

python3 activity_change_recognition_detection_delay_eval_summary.py --folder uLSIF --train_or_test test