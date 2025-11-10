#! /bin/bash

SEED=(0 1 2 3 4)
years=(2009 2010 2011 2012 2013 2014 2015 2016 2017)
# years=(2016)

for year in "${years[@]}" ;do
    downstream_year=$((year+1))
    for seed in "${SEED[@]}"; do
        echo "Running: year=$year, seed=$seed"
        # python run_graph.py --pretrain_dataset $year --test_dataset $downstream_year --seed $seed
        CUDA_VISIBLE_DEVICES=1 python graphcontrol.py --dataset MyPygDataset --epochs 300 --lr 0.5 --optimizer adamw --weight_decay 5e-4 --threshold 0.17 --walk_steps 256 --restart 0.8 --seeds $seed --year $year
    done
done
