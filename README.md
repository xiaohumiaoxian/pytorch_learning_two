# pytorch_learning_two
## PyTorch深度学习模块

### 基本配置

#### 导包

常见的包有

****

| **`os`**    | operate system |
| ----------- | -------------- |
| **`numpy`** | 科学计算       |

等，此外还需要调用PyTorch自身一些模块便于灵活使用，比如

| `pandas`                          | 表格操作，数据存储文件交互   |
| --------------------------------- | ---------------------------- |
| **`torch`**                       | pytorch                      |
| **`torch.nn`**                    | net work                     |
| **`torch.utils.data.Dataset`**    | 数据读取，数据预处理方式     |
| **`torch.utils.data.DataLoade`r** | 数据加载器（可快速迭代数据） |
| **`torch.optimizer`**             | 优化器                       |
| **`matplotlib、seaborn`**         | 可视化                       |
| **`sklearn`**                     | 下游分析和指标计算           |

等等。

```python
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer
```

#### 配置训练过程的超参数

- batch size

- 初始学习率（初始）（learning rate）

- 训练次数（max_epochs）

- 读入数据线程（num_workers）

  ```python
  batch_size = 256
  num_workers=4
  # 批次的大小
  lr = 1e-4
  # 优化器的学习率
  max_epochs = 100
  ```

- GPU配置

  ```python
  # 方案一：使用os.environ，这种情况如果使用GPU不需要设置
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  
  # 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  ```

### 数据读入

定义自己的Dataset类来实现灵活的数据读取，主要包含三个函数：

- `__init__`: 传入外部参数，定义样本集
- `__getitem__`: 逐个读取样本集，可进行一定变换，并返回所需的数据
- `__len__`: 返回数据集样本数

使用DataLoader来按批次读入数据：

- batch_size：每次读入的样本数
- num_workers：有多少个进程用于读取数据
- shuffle：是否将读入的数据打乱
- drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练
- pin_memory：表示要将load进来的数据是否要拷贝到pin_memory区中，其表示生成的Tensor数据是属于内存中的锁页内存区，这样将Tensor数据转移到GPU中速度就会快一些，默认为False。

> 下载并使用PyTorch提供的内置数据集（适合常见的数据集，适用于快速测试的方法）
>
> 从网站下载csv格式存储的数据，读入并转成预期的格式（需自己构建Dataset，对数据进行必要变换，eg：统一图片大小，数据格式转为tensor类等
>
> 可用torchvision---官方图像处理工具库

```python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import  torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from  torchvision import datasets

os.environ['CUDA_VISIBLE_DEVICES']='0'
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size=256
num_workers=0
lr=1e-4
epochs=20
#设置数据变换
"""与模型匹配"""
image_size=28
data_transform=transforms.Compose([
   """内置PL库，取决于后续数据读取方式，如果使用内置数据集则不需要"""
    transforms.ToPILImage(),
    """设置大小"""
    transforms.Resize(image_size),
    """变为tensor的形式"""
    transforms.ToTensor()
])
# #读取方式一：使用torchvision自带数据集,train=False代表测试集
# tran_data=datasets.FashionMNIST(root='./',train=True,download=True,transform=data_transform)
# test_data=datasets.FashionMNIST(root='./',train=False,download=True,transform=data_transform)

#读取方式二：读入csv格式的数据，自行构建Dataset类
#继承Dataset子类
class FMDataset(Dataset):
    def __init__(self,df,transform=None):
        #初始化
        self.df=df
        self.transform=transform
        #读图，从第二列开始--第一列是标签label，提取为uint8
        self.images=df.iloc[:,1:].values.astype(np.uint8)
        self.labels=df.iloc[:,0].values
#数据集长度
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #所有image第index（个）行---变化形状，28*28，1---单色通道
        image=self.images[idx].reshape(28,28,1)
        #将标签变为整型
        label=int(self.labels[idx])
        if self.transform is not None:
            image=self.transform(image)
        else:
            #image变为tensor的形式，图像每一个数是0-255除法后变为0-1方便模型处理
            image=torch.tensor(image/255.,dtype=torch.float)
        label=torch.tensor(label,dtype=torch.long)
        return image,label
#实例化
train_df=pd.read_csv("./FashionMNIST/fashion-mnist_train.csv")
test_df=pd.read_csv("./FashionMNIST/fashion-mnist_test.csv")
train_data=FMDataset(train_df,data_transform)
test_data=FMDataset(test_df,data_transform)
#定义DataLoader类，以便在训练和测试时加载数据
#从哪个dataset load data数据，每批加载的数据，是否打乱顺序，用多少线程读数据，是否将最后没达到批次样本继续参与训练
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
test_load=DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)

#next表迭代（1次），shape形状，matplotlib进行可视化，展示第0个数据
image,label=next(iter(train_loader))
print(image.shape,label.shape)
plt.imshow(image[100][0],cmap="gray")
```

```python
torch.Size([256, 1, 28, 28]) torch.Size([256])#batchsize为256，通道为1，28为图片大小
```

### 模型构建

PyTorch中神经网络构造一般是基于 Module 类的模型来完成的（nn.Module)

- 重载`__init__`和forward函数

- “层定义+层顺序”方式构建
- 神经网络常见层nn.Conv2d,nn.MaxPool2d,nn.Linear,nn.ReLU,……

#### 模型设计

```python
import torch
from torch import nn

#继承nn.Module类
class Net(nn.Module):
    #完成关于层的定义
    def __int__(self):
        #supper初始化，层定义
        super(Net,self).__init__()
        #序贯模型
        self.conv=nn.Sequential(
        #二维卷积
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_2_t,
            nn.Conv2d(1,32,5),
            #激活函数
            nn.ReLU(),
            #池化
            nn.MaxPool2d(2,stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Dropout(0.3)
        )
        #全连接层，从64*4*4神经元变成512，再从512变为结果10（类）
        self.fc=nn.Sequential(
            nn.Linear(64*4*4,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    #层顺序的排列（前馈）
    def forward(self, x):
        #数据x走conv卷积流程
        x=self.conv(x)
        #数据拉平，便于全连接层的操作
        x=x.view(-1,64*4*4)
        #变换数据维度
        x=self.fc(x)
        return x
#实例化
model=Net()
model=model.cuda()
```

### 损失函数

- BCELoss 二分类损失函数
- BCEWithLogitsLoss
- 负对数似然损失函数 NLLLoss
- 交叉熵损失函数 CrossEntropyLoss
- L1损失函数
- L2 损失函数
- SmoothL1Loss 损失函数

一般通过torch.nn调用, 损失函数常用操作：backward()
torch.Size([256, 1, 28, 28]) torch.Size([256])
EPOCH:1 	Training Loss: 0.680543
Epoch: 1 	Validation Loss: 0.441949, Accuracy: 0.842200
EPOCH:2 	Training Loss: 0.431540
Epoch: 2 	Validation Loss: 0.345487, Accuracy: 0.879300
EPOCH:3 	Training Loss: 0.366046
Epoch: 3 	Validation Loss: 0.316142, Accuracy: 0.887900
EPOCH:4 	Training Loss: 0.336380
Epoch: 4 	Validation Loss: 0.285833, Accuracy: 0.896500
EPOCH:5 	Training Loss: 0.311207
Epoch: 5 	Validation Loss: 0.267263, Accuracy: 0.902200
EPOCH:6 	Training Loss: 0.290970
Epoch: 6 	Validation Loss: 0.257569, Accuracy: 0.905200
EPOCH:7 	Training Loss: 0.275216
Epoch: 7 	Validation Loss: 0.257715, Accuracy: 0.905700
EPOCH:8 	Training Loss: 0.263491
Epoch: 8 	Validation Loss: 0.247774, Accuracy: 0.910000
EPOCH:9 	Training Loss: 0.253206
Epoch: 9 	Validation Loss: 0.257099, Accuracy: 0.902700
EPOCH:10 	Training Loss: 0.243676
Epoch: 10 	Validation Loss: 0.236460, Accuracy: 0.914600
EPOCH:11 	Training Loss: 0.233477
Epoch: 11 	Validation Loss: 0.227580, Accuracy: 0.917400
EPOCH:12 	Training Loss: 0.227029
Epoch: 12 	Validation Loss: 0.217505, Accuracy: 0.919000
EPOCH:13 	Training Loss: 0.219373
Epoch: 13 	Validation Loss: 0.218901, Accuracy: 0.918400
EPOCH:14 	Training Loss: 0.214295
Epoch: 14 	Validation Loss: 0.221296, Accuracy: 0.918900
EPOCH:15 	Training Loss: 0.204996
Epoch: 15 	Validation Loss: 0.217973, Accuracy: 0.920500
EPOCH:16 	Training Loss: 0.197011
Epoch: 16 	Validation Loss: 0.212263, Accuracy: 0.919200
EPOCH:17 	Training Loss: 0.191312
Epoch: 17 	Validation Loss: 0.207198, Accuracy: 0.925000
EPOCH:18 	Training Loss: 0.187683
Epoch: 18 	Validation Loss: 0.202116, Accuracy: 0.925500
EPOCH:19 	Training Loss: 0.181617
Epoch: 19 	Validation Loss: 0.199740, Accuracy: 0.926000
EPOCH:20 	Training Loss: 0.179331
Epoch: 20 	Validation Loss: 0.202072, Accuracy: 0.925200
