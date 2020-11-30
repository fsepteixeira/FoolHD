#!/bin/bash
 
METHODS=(untargeted targeted)

for method in "${METHODS[@]}"
do
	echo $method	
	python3 -W ignore score.py --method=$method 
done




