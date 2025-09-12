#!/usr/bin/env bash
PROJECT_NAME="test-nohup-runs-Sep12-2025_0"
DIR="." # It will put a "wandb/" directory in whatever DIR is set to.
# If running on your local machine, you can set DIR="." or DIR="$HOME/wandb_runs"
# If on a cluster, you might want to set DIR to a path to a directory with plenty of storage, e.g.
# DIR="/data/levinema/cd_dynamax"

# Activate conda env
eval "$(conda shell.bash hook)" # essentially runs "conda init" in a bash script
conda activate cd_dynamax-2025

# Take wandb online
wandb online

### ----- L63 DICT (regular timesteps) ------- ###
#### L = 0.01 (small diffusion) requires covariance inflation to track true state
# dt=0.001 (small dt)
RUN_NAME="L63-LaplaceDict-L1e-2-dt1e-3"
nohup python l63_dict_laplace.py --init_lr 0.05 --dt 0.001 --num_epochs 5000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

# dt=0.01
RUN_NAME="L63-LaplaceDict-L1e-2-dt1e-2"
nohup python l63_dict_laplace.py --init_lr 0.05 --dt 0.01 --num_epochs 5000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

# dt=0.1 (large dt)
# this one needs some optimization tuning! 20000 epochs wasn't really enough. 
RUN_NAME="L63-LaplaceDict-L1e-2-dt1e-1"
nohup python l63_dict_laplace.py --init_lr 0.005 --dt 0.1 --num_epochs 10000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

#### L= 1.0 (medium diffusion) ####
# small dt=0.001 (takes the longest)
RUN_NAME="L63-LaplaceDict-L1-dt1e-3"
nohup python l63_dict_laplace.py --init_lr 0.1 --dt 0.001 --num_epochs 5000  --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

# moderate dt=0.01
RUN_NAME="L63-LaplaceDict-L1-dt1e-2"
nohup python l63_dict_laplace.py --init_lr 0.1 --dt 0.01 --num_epochs 5000 --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

# larger dt=0.1
RUN_NAME="L63-LaplaceDict-L1-dt1e-1"
nohup python l63_dict_laplace.py --init_lr 0.01 --dt 0.1 --num_epochs 10000  --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

#### ------- L63 Neural Net (regular timesteps) ------- ###
RUN_NAME="L63-NN-FullObs"
nohup python l63_nn.py --init_lr 0.01 --num_epochs 1000 --dt 0.01 --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

##### ------ Partially observed L63 Neural Net (regular timesteps) ------- ###
RUN_NAME="L63-NN-PartialObs"
nohup python l63_nn.py --num_epochs 5000 --init_lr 0.01 --use_lr_scheduler 1 --emission_dim 1 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

#### ------- Van Der Pol GP-expansion (regular timesteps) ------- ###
RUN_NAME="VDP-HSGP-Matern-m10"
nohup python vdp_hsgp_matern.py --num_epochs 10000 --m 10 --init_lr 0.005 --seed 0 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

#### -------- L96 DICT (regular timesteps dt=0.01) ---------- ###
# 5 states
RUN_NAME="L96-LaplaceDict-dx5"
nohup python l96_dict_laplace_init_zero.py --init_lr 0.01 --num_epochs 2000 --state_dim 5 --N_particles 25 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1

# 10 states
RUN_NAME="L96-LaplaceDict-dx10"
nohup python l96_dict_laplace_init_zero.py --init_lr 0.01 --num_epochs 1000 --state_dim 10 --N_particles 100 --project $PROJECT_NAME --run_name $RUN_NAME --dir $DIR & > $DIR/$PROJECT_NAME/$RUN_NAME.out 2>&1
