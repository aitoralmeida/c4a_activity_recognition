#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder word2vec/window_1_iterations_5_embedding_size_50 --offset 1 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_2_iterations_5_embedding_size_50 --offset 1 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_3_iterations_5_embedding_size_50 --offset 1 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_4_iterations_5_embedding_size_50 --offset 1 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_5_iterations_5_embedding_size_50 --offset 1 --train_or_test test

python3 activity_change_recognition_eval_summary.py --folder word2vec/window_1_iterations_5_embedding_size_50 --offset 5 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_2_iterations_5_embedding_size_50 --offset 5 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_3_iterations_5_embedding_size_50 --offset 5 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_4_iterations_5_embedding_size_50 --offset 5 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_5_iterations_5_embedding_size_50 --offset 5 --train_or_test test

python3 activity_change_recognition_eval_summary.py --folder word2vec/window_1_iterations_5_embedding_size_50 --offset 10 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_2_iterations_5_embedding_size_50 --offset 10 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_3_iterations_5_embedding_size_50 --offset 10 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_4_iterations_5_embedding_size_50 --offset 10 --train_or_test test
python3 activity_change_recognition_eval_summary.py --folder word2vec/window_5_iterations_5_embedding_size_50 --offset 10 --train_or_test test

python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec/window_1_iterations_5_embedding_size_50 --train_or_test test
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec/window_2_iterations_5_embedding_size_50 --train_or_test test
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec/window_3_iterations_5_embedding_size_50 --train_or_test test
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec/window_4_iterations_5_embedding_size_50 --train_or_test test
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec/window_5_iterations_5_embedding_size_50 --train_or_test test