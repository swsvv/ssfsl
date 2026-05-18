#!/bin/bash

python feat_viz.py /path/to/ckpt/<exp_name>/config.yaml --mode=test --classifier=LogisticRegression --num_episode=5 --n_way=5 --num_qry=15 --num_spt=85 --device=cuda:0
