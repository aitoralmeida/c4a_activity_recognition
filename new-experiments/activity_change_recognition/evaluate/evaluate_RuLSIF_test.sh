#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder RulSIF --offset 1 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder RulSIF --offset 5 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder RulSIF --offset 10 --train_or_test test

python3 activity_change_recognition_detection_delay_eval_summary.py --folder RuLSIF --train_or_test test