#!/bin/bash
DATA_DIR=".../ensemble_hypergraph" && \
python -m ensemble \
	--input_dataset ${DATA_DIR}/user_ratedmovies.dat
