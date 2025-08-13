## Bash loop over data_config files:
# configs/data/l63_data_x1
# configs/data/l63_data_x1_regular
# configs/data/l63_data_x1_regular_lowFreq
# configs/data/l63_data_x1_lowFreq

## Bash loop over model_config files:
# configs/model/true_l63_mech_x1_initCov400_diffusionCoeff1e-2_emiCov1
# configs/model/true_l63_mech_x1_initCov100_diffusionCoeff1e-2_emiCov20
# configs/model/true_l63_mech_x1_initCov100_diffusionCoeff1e-2_emiCov1
# configs/model/true_l63_mech_x1_initCov100_diffusionCoeff1e-1_emiCov1
# configs/model/true_l63_mech_x1_initCov400_diffusionCoeff1e-1_emiCov20

# Call form is:
# python filter_then_forecast.py --data_config_file <> --model_config_file <>

#!/bin/bash
# Loop over data configurations
for data_config in configs/data/l63_data_x1_lowFreq; do
                    #configs/data/l63_data_x1 configs/data/l63_data_x1_regular \
                    #configs/data/l63_data_x1_regular_lowFreq \
    # Loop over model configurations
    for model_config in configs/model/true_l63_mech_x1_initCov400_diffusionCoeff1e-1_emiCov20 \
                        configs/model/true_l63_mech_x1_initCov400_diffusionCoeff1e-2_emiCov1 ; do
                        #configs/model/true_l63_mech_x1_initCov100_diffusionCoeff1e-2_emiCov20 \
                        #configs/model/true_l63_mech_x1_initCov100_diffusionCoeff1e-2_emiCov1 \
                        #configs/model/true_l63_mech_x1_initCov100_diffusionCoeff1e-1_emiCov1 \
        # Run the filter then forecast script
        python run_filtering_then_forecast_experiment.py \
            --data_config_file $data_config \
            --model_config_file $model_config \
            --do_run 1 \
            --do_eval 1 \
            --enforce_twin_experiment 1
        echo "Completed filtering and forecasting for data: $data_config, model: $model_config"
    done
done