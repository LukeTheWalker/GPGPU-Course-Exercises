#!/usr/bin/bash

# Take first argument as input number
nels=$1

# List of programs to run
declare -a programs=("vecsmooth" "vecsmooth_vec" "vecsmooth_lmem" "vecsmooth_lmem_vec")


# Make each file and run it
for program in "${programs[@]}"
do
    echo "--------------------- Running $program "
    # Make the program
    make $program

    # Run the program
    ./$program $nels
done