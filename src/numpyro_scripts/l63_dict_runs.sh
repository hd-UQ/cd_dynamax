### L63 DICT (regular timesteps)###
PROJECT_NAME="l63_dict_laplace"

#### L = 0.01 (small diffusion) requires covariance inflation to track true state
# dt=0.001 (small dt)
nohup python l63_dict_laplace.py --init_lr 0.05 --dt 0.001 --num_epochs 5000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 0 --project $PROJECT_NAME &
# nohup python l63_dict_laplace.py --init_lr 0.05 --dt 0.001 --num_epochs 5000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 1 --project $PROJECT_NAME &

# dt=0.01
nohup python l63_dict_laplace.py --init_lr 0.05 --dt 0.01 --num_epochs 5000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 0 --project $PROJECT_NAME &
# nohup python l63_dict_laplace.py --init_lr 0.05 --dt 0.01 --num_epochs 5000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 1 --project $PROJECT_NAME &

# dt=0.1 (large dt)
nohup python l63_dict_laplace.py --init_lr 0.005 --dt 0.1 --num_epochs 20000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 0 --project $PROJECT_NAME &
# nohup python l63_dict_laplace.py --init_lr 0.005 --dt 0.1 --num_epochs 20000 --diffusion_coeff 0.01 --inflation_delta 0.5 --seed 1 --project $PROJECT_NAME &


#### L= 1.0 (medium diffusion) ####

# small dt=0.001 (takes the longest)
nohup python l63_dict_laplace.py --init_lr 0.1 --dt 0.001 --num_epochs 5000  --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME &
# nohup python l63_dict_laplace.py --init_lr 0.1 --dt 0.001 --num_epochs 5000  --diffusion_coeff 1.0 --seed 1 --project $PROJECT_NAME &

# moderate dt=0.01
nohup python l63_dict_laplace.py --init_lr 0.1 --dt 0.01 --num_epochs 5000 --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME &
# nohup python l63_dict_laplace.py --init_lr 0.1 --dt 0.01 --num_epochs 5000 --diffusion_coeff 1.0 --seed 1 --project $PROJECT_NAME &

# larger dt=0.1
nohup python l63_dict_laplace.py --init_lr 0.01 --dt 0.1 --num_epochs 10000  --diffusion_coeff 1.0 --seed 0 --project $PROJECT_NAME &
# nohup python l63_dict_laplace.py --init_lr 0.01 --dt 0.1 --num_epochs 10000  --diffusion_coeff 1.0 --seed 1 --project $PROJECT_NAME &


#### L96 DICT (regular timesteps dt=0.01)###
PROJECT_NAME="l96_dict_laplace"

# 5 states
nohup python l96_dict_laplace_init_zero.py --init_lr 0.01 --num_epochs 2000 --state_dim 5 --N_particles 25 &

# 10 states
nohup python l96_dict_laplace_init_zero.py --init_lr 0.01 --num_epochs 1000 --state_dim 10 --N_particles 100 &