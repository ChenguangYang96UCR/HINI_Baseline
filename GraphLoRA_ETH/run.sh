#! /bin/bash

SEED=(0 1 2 3 4)
# years=(2009 2010 2011 2012 2013 2014 2015 2016)
years=(2015 2016)

for year in "${years[@]}" ;do
    downstream_year=$((year+1))
    for seed in "${SEED[@]}"; do
        echo "Running: year=$year → downstream_year=$downstream_year, seed=$seed"
        python run_graph.py --pretrain_dataset $year --test_dataset $downstream_year --seed $seed
    done
done
