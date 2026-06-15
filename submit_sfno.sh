#!/bin/bash
#SBATCH --account=gfdlport
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=SPEAR_SFNO_EMULATOR-1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uriel.ramirez@noaa.gov
#SBATCH --output=logs/SPEAR_SFNO_EMULATOR-1.%j

module load rdhpcs-conda/25.11.0
conda activate /scratch4/GFDL/gfdlscr/Uriel.Ramirez/conda/envs/WUT4

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

cd /scratch4/GFDL/gfdlscr/Uriel.Ramirez/DEV/LIGHTING
echo $LD_LIBRARY_PATH

WORKING_DIR="/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/architecture_comparison"
YAML_NAME="config_sfno-1.yaml"
LABEL="sfno-candidate-1"

mkdir -p "${WORKING_DIR}"
cat <<EOF > "${WORKING_DIR}/${YAML_NAME}"
verbose: True
seed: 42
model_type: "sfno"
nepochs: 10

sfno:
  embed_dim: 128
  num_layers: 6
  filter_type: "linear"
  operator_type: "dhconv"
  scale_factor: 1
  spectral_layers: 4
  mlp: 2.0
  drop_path_rate: 0.1

paths:
  input_dir: "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/ANEMOI-DATASETS/Set3"
  training: "train_dataset.zarr"
  validating: "val_dataset.zarr"
  testing: "test_dataset.zarr"

batch_size: 8
precision: "bf16-mixed"

optimizer:
  name: "Adam"
  learning_rate: 0.0001
  weight_decay: 0.01

data_config:
  type: "residual"     # "default" or "residual"
  method:
    name: "autoregressive"      # "lags" or "autoregressive"
    nsteps: 3

use_coordinates: False

variables:
  air_temperature_at_two_meters:
    variable_type: "prognostic"
    target_weight: 1.0
    normalization: "z-score"

  eastward_wind:
    variable_type: "prognostic"
    target_weight: 1.0
    normalization: "z-score"
    vertical_levels: [25, 96, 203, 345, 517, 695, 847, 963]

  northward_wind:
    variable_type: "prognostic"
    target_weight: 1.0
    normalization: "z-score"
    vertical_levels: [25, 96, 203, 345, 517, 695, 847, 963]

  specific_humidity:
    variable_type: "prognostic"
    target_weight: 1.0
    normalization: "z-score"
    vertical_levels: [25, 96, 203, 345, 517, 695, 847, 963]

  air_temperature:
    variable_type: "prognostic"
    target_weight: 1.0
    normalization: "z-score"
    vertical_levels: [25, 96, 203, 345, 517, 695, 847, 963]

  DLWRFsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  DLWRFsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  PRATEsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  PRESsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  SHTFLsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  ULWRFsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  ULWRFtoa:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  USWRFsfc:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  USWRFtoa:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  surface_evaporation_rate:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  surface_temperature:
    variable_type: "diagnostic"
    target_weight: 1.0
    normalization: "z-score"

  DSWRFtoa:
    variable_type: "forcing"
    normalization: "z-score"

  HGTsfc:
    variable_type: "forcing-static"
    normalization: "z-score"

  land_fraction:
    variable_type: "forcing-static"
    normalization: "none"
EOF

echo "Training the model: ${LABEL} ... "
echo "Using config: ${YAML_NAME}"
echo "Working directory: ${WORKING_DIR}"
echo "---------------------------------------------"

python train_driver.py \
    --working_dir "${WORKING_DIR}" \
    --yaml_name "${YAML_NAME}" \
    --label "${LABEL}"
