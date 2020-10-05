#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder SEP --offset 1 --train_or_test $1
python3 activity_change_recognition_eval_summary.py --folder SEP --offset 5 --train_or_test $1
python3 activity_change_recognition_eval_summary.py --folder SEP --offset 10 --train_or_test $1

python3 activity_change_recognition_detection_delay_eval_summary.py --folder SEP --train_or_test $1