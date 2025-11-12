#! /bin/bash

SEED=(0 1 2 3 4)
# years=(2010 2011 2012 2013 2014 2015)
years=(2009 2016)

for year in "${years[@]}" ;do
    downstream_year=$((year+1))
    for seed in "${SEED[@]}"; do
        echo "Running: year=$year → downstream_year=$downstream_year, seed=$seed"
        python main.py --pretrain_dataset $year --test_dataset $downstream_year --seed $seed
    done
done

SEED=(0 10 42 1234 4321)
eth_years=("2018_jan" "2018_feb" "2018_mar" "2018_apr" "2018_may" "2018_jun" "2018_jul" "2018_aug" "2018_sep" "2018_oct" "2018_nov" "2018_dec" "2019_jan")
# for ((i=0; i<${#eth_years[@]}-1; i++)); do
#     current="${eth_years[$i]}"
#     next="${eth_years[$((i+1))]}"
#     for seed in "${SEED[@]}"; do
#         echo "Running ETH: year=$current → downstream_year=$next, seed=$seed"
#         python main.py --dataset eth --pretrain_dataset $current --test_dataset $next --seed $seed --epochs 100
#     done
# done