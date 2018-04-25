# 简介
本代码为 TinyMind 第一届汉字书法识别挑战赛。
http://www.tinymind.cn/competitions/41#ranking

### 数据集
本数据集拥有 100个汉字，每个汉字 500张图片，共计5W 张图片，其中 4W张作为训练集，1W张图片作为验证集。

### 训练模型
使用的网络是 inception_v4 ,所以这里我们使用 tensorflow 提供的预训练的 inception_v4模型作为输入。

### 模型
模型代码来自：
https://github.com/tensorflow/models/tree/master/research/slim
这里为了适应本挑战赛提供的数据集，稍作修改，添加了一个 tmd001 数据集以及一个训练并验证的脚本

# 参考内容
本地运行 slim 框架所用命令行：

训练：
python3 train_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --checkpoint_path=/path/to/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/path/to/train_ckpt --learning_rate=0.001 --optimizer=rmsprop  --batch_size=32

train集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=train --model_name=inception_v4 --checkpoint_path=/path/to/train_ckpt --eval_dir=/path/to/train_eval --batch_size=32 --max_num_batches=128

validation集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=validation --model_name=inception_v4 --checkpoint_path=/path/to/train_ckpt --eval_dir=/path/to/validation_eval --batch_size=32 --max_num_batches=128

统一脚本：
python3 train_eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --checkpoint_path=/path/to/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --optimizer=rmsprop --train_dir=/path/to/log/train_ckpt --learning_rate=0.001 --dataset_split_name=validation --eval_dir=/path/to/eval --max_num_batches=128

