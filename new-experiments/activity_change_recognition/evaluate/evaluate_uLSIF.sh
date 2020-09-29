#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder ulSIF --offset 1
python3 activity_change_recognition_eval_summary.py --folder ulSIF --offset 5
python3 activity_change_recognition_eval_summary.py --folder ulSIF --offset 10

python3 activity_change_recognition_detection_delay_eval_summary.py --folder uLSIF