#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_1_window_1_iterations_5_embedding_size_50 --offset 1
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_2_window_1_iterations_5_embedding_size_50 --offset 1
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_3_window_1_iterations_5_embedding_size_50 --offset 1
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_4_window_1_iterations_5_embedding_size_50 --offset 1
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_5_window_1_iterations_5_embedding_size_50 --offset 1

python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_1_window_1_iterations_5_embedding_size_50 --offset 5
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_2_window_1_iterations_5_embedding_size_50 --offset 5
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_3_window_1_iterations_5_embedding_size_50 --offset 5
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_4_window_1_iterations_5_embedding_size_50 --offset 5
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_5_window_1_iterations_5_embedding_size_50 --offset 5

python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_1_window_1_iterations_5_embedding_size_50 --offset 10
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_2_window_1_iterations_5_embedding_size_50 --offset 10
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_3_window_1_iterations_5_embedding_size_50 --offset 10
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_4_window_1_iterations_5_embedding_size_50 --offset 10
python3 activity_change_recognition_eval_summary.py --folder word2vec_context/context_window_5_window_1_iterations_5_embedding_size_50 --offset 10

python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec_context/context_window_1_window_1_iterations_5_embedding_size_50
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec_context/context_window_2_window_1_iterations_5_embedding_size_50
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec_context/context_window_3_window_1_iterations_5_embedding_size_50
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec_context/context_window_4_window_1_iterations_5_embedding_size_50
python3 activity_change_recognition_detection_delay_eval_summary.py --folder word2vec_context/context_window_5_window_1_iterations_5_embedding_size_50