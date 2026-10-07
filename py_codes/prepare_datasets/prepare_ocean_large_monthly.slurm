#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=prepare_ocean_large
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_monthly_prepare_%j.log

set -e
cd /home/x-zjian1/unsz
./unsz_env/bin/python py_codes/prepare_datasets/prepare_ocean_large.py \
    -monthly-source-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/mpaso \
    -mesh-source-path /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/source/MPAS-O_V6.0_RRS18to6/oRRS18to6v3.171116.nc \
    -output-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
