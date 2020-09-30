#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder word2vec/window_${1}_iterations_${2}_embedding_size_${3} --offset 1 --train_or_test $4
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_${1}_iterations_${2}_embedding_size_${3} --offset 5 --train_or_test $4
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_${1}_iterations_${2}_embedding_size_${3} --offset 10 --train_or_test $4
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec/window_${1}_iterations_${2}_embedding_size_${3} --train_or_test $4