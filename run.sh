#!/bin/bash

python ./HW1/inference.py --para_model_dir ./para-lert-vera --span_model_dir ./span-lert-verc --context_file ${1} --test_file ${2} --output_file ${3}
