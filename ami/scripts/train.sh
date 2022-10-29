#!/bin/bash

script_dir="./scripts"
# check if the script directory exists
if [ ! -d "$script_dir" ]; then
    echo "Error: script directory not found"
    exit 1
fi

# array of scripts for training
declare -a scripts=(
    "train_small_ctc_char.sh"
)
declare -a info=(
    "13m param CTC conformer with vocab size of 29"
)

#check if the scripts exist
for i in "${scripts[@]}"
do
    if [ ! -f "$script_dir/$i" ]; then
        echo "Error: script $i not found"
        exit 1
    fi
done

for i in "${!scripts[@]}"
do
    num=$((i+1))
    echo "${num}. ${scripts[$i]} <--- ${info[$i]}"
    echo
done

# get user input for which script to run
read -p "Enter the number of the script you want to run: " script_num

if [ $script_num -gt ${#scripts[@]} ] || [ $script_num -lt 1 ]
then
    echo "Invalid script number"
    exit 1
fi

script_num=$(($script_num - 1))
echo "You chose ${scripts[$script_num]}"
echo 

# run the script
bash ${script_dir}/${scripts[$script_num]}
