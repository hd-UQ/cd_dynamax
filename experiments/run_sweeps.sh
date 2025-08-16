python run_filtering_then_forecast_sweepDt_L.py --Ls 0.01,0.1,1,5,10,40 --dts 1e-3,1e-2,0.1,0.2,0.5,1.0 --num_reps 10 --model_config_file configs/model/true_l63_mech_x1_initCov400 --output_root results/filter_then_forecast_SWEEP_inflation_initCov400_T100_irregular --t1 100 --irregular_samples 1
python run_filtering_then_forecast_sweepDt_L.py --Ls 0.01,0.1,1,5,10,40 --dts 1e-3,1e-2,0.1,0.2,0.5,1.0 --num_reps 10 --model_config_file configs/model/true_l63_mech_x1_initCov400 --output_root results/filter_then_forecast_SWEEP_inflation_initCov400_T100_regular --t1 100 --irregular_samples 0

python run_filtering_then_forecast_sweepDt_L.py --Ls 0.01,0.1,1,5,10,40 --dts 1e-3,1e-2,0.1,0.2,0.5,1.0 --num_reps 10 --model_config_file configs/model/true_l63_mech_x1_initCov400 --output_root results/filter_then_forecast_SWEEP_inflation_initCov400_T10_irregular --t1 3 --irregular_samples 1
python run_filtering_then_forecast_sweepDt_L.py --Ls 0.01,0.1,1,5,10,40 --dts 1e-3,1e-2,0.1,0.2,0.5,1.0 --num_reps 10 --model_config_file configs/model/true_l63_mech_x1_initCov400 --output_root results/filter_then_forecast_SWEEP_inflation_initCov400_T10_regular --t1 3 --irregular_samples 0

python run_filtering_then_forecast_sweepDt_L.py --Ls 0.01,0.1,1,5,10,40 --dts 1e-3,1e-2,0.1,0.2,0.5,1.0 --num_reps 10 --model_config_file configs/model/true_l63_mech_x1_initCov400 --output_root results/filter_then_forecast_SWEEP_inflation_initCov400_T3_irregular --t1 3 --irregular_samples 1
python run_filtering_then_forecast_sweepDt_L.py --Ls 0.01,0.1,1,5,10,40 --dts 1e-3,1e-2,0.1,0.2,0.5,1.0 --num_reps 10 --model_config_file configs/model/true_l63_mech_x1_initCov400 --output_root results/filter_then_forecast_SWEEP_inflation_initCov400_T3_regular --t1 3 --irregular_samples 0

