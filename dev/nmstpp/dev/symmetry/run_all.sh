#!/bin/sh
# 3 arms x 3 seeds on the full Wyscout split; the arms of one seed run in parallel.
PY=python
for seed in 0 1 2; do
  for arm in baseline augment frameavg; do
    $PY -u train_symmetry.py --config configs/train_NMSTPP_reproduction.yaml --arm $arm --seed $seed \
      > dev/symmetry/logs/${arm}_seed${seed}.log 2>&1 &
  done
  wait
done
echo ALL DONE
