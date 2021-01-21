#!/bin/bash
DATA_DIR="/home/alireza/HOME/RecSys-master/ensemble_hypergraph" && \
python3 -m ensemble \
	--input_dataset ${DATA_DIR}/datasets/aotm.csv
