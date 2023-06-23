# Code for Two-Stage Focus Measurement Network with Joint Boundary Refinement for Multi-Focus Image Fusion

## 环境要求 / Environmental Requirements  
  
```
conda create -n PIF python=3.7  
conda activate PIF  
conda install pytorch=1.6.0 torchvision -c pytorch  
pip install tqdm pyyaml tensorboardX opencv-python  
```
  
## 参数设置 / Parameter Setting  
  
```python
# 此文件为./config/config.yaml

PROJECT: # 项目参数
  name: 'MF_NeXt' # 项目名称
  save_path: './work_dirs/' # 项目保存路径，训练模型会保存至此路径下的项目名称文件夹中

TRAIN_DATASET: # 训练数据集参数
  root_dir: './datasets/TNO_crop/' # 训练数据集根目录
  sensors: [ 'Far', 'Near' ] # 训练数据集包含的数据类型
  channels: 3 # 训练数据中图片的通道数
  input_size: 256 # 训练数据中图片的尺寸
  mean: [ 0.485, 0.456, 0.406 ] # 训练数据中图片的归一化均值（暂时用不到）
  std: [ 0.229, 0.224, 0.225 ] # 训练数据中图片的归一化标准差（暂时用不到）

TRAIN: # 训练参数
  batch_size: 32 # 训练批次大小
  max_epoch: 200 # 训练最大代数
  lr: 0.01 # 训练学习率
  gamma: 0.01 # 训练学习率衰减系数
  milestones: [ 100, 150, 175 ] # 训练学习率衰减的里程碑
  opt: Adam # 训练优化器
  loss_func: ['Loss_MF'] # 训练使用的损失函数
  val_interval: 1 # 训练每过多少代数后保存权重
  debug_interval: 100 # 训练每过多少批次后进行可视化，结果可视化在tensorboard中
  resume: None # 训练停止后继续训练加载权重路径
  loss_weights: [ 1000, 1 ] # 对VIF_Net的两个损失的权值

TEST_DATASET: # 测试数据集参数
  root_dir: './datasets/Lytro/' # 测试数据集根目录
  sensors: [ 'Far', 'Near' ] # 测试数据集包含的数据类型
  channels: 3 # 测试数据中图片的通道数
  input_size: 512 # 测试数据中图片的尺寸
  mean: [ 0.485, 0.456, 0.406 ] # 测试数据中图片的归一化均值（暂时用不到）
  std: [ 0.229, 0.224, 0.225 ] # 测试数据中图片的归一化标准差（暂时用不到）

TEST: # 测试参数
  batch_size: 1 # 测试批次大小
  weight_path: './work_dirs/model_50.pth' # 测试加载的权重路径
  save_path: './results/' # 测试结果保存路径

MODEL: # 模型参数
  model_name: 'MF_NeXt' # 模型名称
  input_channels: 3 # 模型输入通道数
  out_channels: 1 # 模型每一层输出的通道数
  input_sensors: [ 'Far', 'Near' ] # 模型输入数据类型

```  

## 训练与测试 / Training And Testing  
  
### 训练 / Training  
&emsp;&emsp;运行  ` python run.py --train `  进行训练。训练的模型权重会保存再指定的路径下。  

#### tensorboardX进行训练可视化  
&emsp;&emsp;运行  ` tensorboard --logdir= XXX `  进行训练可视化。将  ` XXX `  替换为模型储存的路径。
  
### 测试 / Testing  
&emsp;&emsp;运行  ` python run.py --test `  进行测试。结果会批量保存至指定路径下。  
