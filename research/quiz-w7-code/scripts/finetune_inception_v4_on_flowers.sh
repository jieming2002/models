#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV4 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v4_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=../tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=../tmp/flowers-models/inception_v4

# Where the dataset is saved to.
DATASET_DIR=../tmp/flowers

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
# 模型：slim框架下的Inception_v4模型
# Inception_v4的Checkpoint：http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
# 数据集：google的flower数据集http://download.tensorflow.org/example_images/flower_photos.tgz 5种类别的花
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
  tar -xvf inception_v4_2016_09_09.tar.gz
  mv inception_v4.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt
  rm inception_v4_2016_09_09.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
# python train_image_classifier.py \
#   --train_dir=${TRAIN_DIR} \
#   --dataset_name=flowers \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4 \
#   --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
#   --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
#   --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
#   --max_number_of_steps=1000 \
#   --batch_size=32 \
#   --learning_rate=0.01 \
#   --learning_rate_decay_type=fixed \
#   --save_interval_secs=60 \
#   --save_summaries_secs=60 \
#   --log_every_n_steps=100 \
#   --optimizer=rmsprop \
#   --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4

# Fine-tune all the new layers for 500 steps.
# python train_image_classifier.py \
#   --train_dir=${TRAIN_DIR}/all \
#   --dataset_name=flowers \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v4 \
#   --checkpoint_path=${TRAIN_DIR} \
#   --max_number_of_steps=500 \
#   --batch_size=32 \
#   --learning_rate=0.0001 \
#   --learning_rate_decay_type=fixed \
#   --save_interval_secs=60 \
#   --save_summaries_secs=60 \
#   --log_every_n_steps=10 \
#   --optimizer=rmsprop \
#   --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4
