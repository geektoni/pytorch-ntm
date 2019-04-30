#!/usr/bin/env bash

python3 ./train.py --seed 1 --task priority-sort --checkpoint-interval 5 -pnum_batches=1000 -pbatch_size=1 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 10 --task priority-sort --checkpoint-interval 5 -pnum_batches=1000 -pbatch_size=1 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 100 --task priority-sort --checkpoint-interval 5 -pnum_batches=1000 -pbatch_size=1 --checkpoint-path ./results/priority-sort/
python3 ./train.py --seed 1000 --task priority-sort --checkpoint-interval 5 -pnum_batches=1000 -pbatch_size=1 --checkpoint-path ./results/priority-sort/
