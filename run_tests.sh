#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 100000 -psequence_width=8 -psequence_max_length=20 -pnum_batches=1000000 --checkpoint-path ./results/priority-sort/

python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=8 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=12 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=16 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=20 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=24 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=28 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 42 --task priority-sort --checkpoint-interval 10000 -psequence_width=1 -psequence_max_length=30 -pnum_batches=100000 --checkpoint-path ./results/priority-sort/
