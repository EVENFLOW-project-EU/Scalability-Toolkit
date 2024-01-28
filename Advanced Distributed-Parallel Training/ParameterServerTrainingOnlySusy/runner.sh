#! /bin/bash

# Preprocessing step
python evenflow/data/preprocessing.py \
    --chunk_size 32 \
    --num_chunks 1 \
    --input_file_path 'SUSY.csv.gz' \
    --train_topic 'train' \
    --test_topic 'test' \
    --num_partitions 16 \
    --train_test_ratio 0.8

# Sync algorithm
python evenflow/driver.py \
    --world_size 3 \
    --mode sync \
    --num_training_batches 3 \
    --num_test_batches 3 \
    --batch_size 16 \
    --lr 0.001 \
    --num_worker_threads 32 \
    --rpc_timeout 0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --ds_source  local\
    --ds_location SUSY.csv.gz \
    --log_root demo/demo_sync \
    --total_rounds 3 \
    --world_size 3 \
    --master_address localhost \
    --master_port 29500 \
    --test_batches 5 \
    --log_root logs/ \
    --gm_threshold 0.1 \
    --max_min_on_sphere_impl torch_standard \
    --fft_coeffs 3
