#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder word2vec_context_retrofitted_activity/context_window_${1}_window_${2}_iterations_${3}_embedding_size_${4} --offset 1 --train_or_test $5
python3 activity_change_recognition_eval_summary.py --folder word2vec_context_retrofitted_activity/context_window_${1}_window_${2}_iterations_${3}_embedding_size_${4} --offset 5 --train_or_test $5
python3 activity_change_recognition_eval_summary.py --folder word2vec_context_retrofitted_activity/context_window_${1}_window_${2}_iterations_${3}_embedding_size_${4} --offset 10 --train_or_test $5
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec_context_retrofitted_activity/context_window_${1}_window_${2}_iterations_${3}_embedding_size_${4} --train_or_test $5