# MyFlow:一个tensorflow1.x风格的迷你机器学习框架
---
本项目仿照tensorflow1.x版本的静态图模式API，使用python实现了一个基本的机器学习框架Myflow，并用该框架给出了一个在猫狗大战(Kaggle Cats and Dogs)数据集上的神经网络分类模型的例子。

## 环境配置：
python3.5 以上

numpy==1.17.4
matplotlib==3.0.3
Pillow==6.2.1
scikit_learn==0.22

---
## 实现的主要功能：
- 计算图的搭建和执行
- 自动求梯度
- 提供了参数优化器的实现
- 提供了一组常用的运算操作
---
## 基本用例：
```python
import myflow as mf
import numpy as np
x = mf.constant(np.asarray([[1.0, 2.5], [3.0, 4.0]]), 'x')
y = mf.placeholder([2, 3], 'y')
z = mf.matmul(x, y, 'z')
with mf.Session() as sess:
    x_val, y_val, z_val = sess.run([x, y, z], feed_dict={y: np.random.random([2, 3])})
    print("x_val:", x_val)
    print("y_val:", y_val)
    print("z_val:", z_val)
```
---
## 搭建模型训练Kaggle Cats and Dogs数据集
搭建了一个二分类的多层全连接网络，分类数据集中的猫和狗的图片。代码见cats_and_dogs.py。

数据集下载地址：<https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip>

1. 下载数据集并解压到Myflow项目的根目录下，将数据集文件夹重命名为kagglecatsanddogs，如下图:
![mulu.png](https://github.com/heiwushi/MyFlow/blob/master/docimgs/mulu.png)

2. 进入Myflow根目录，安装需要的python 依赖： pip install -r requirements.txt

3. 运行python cats_and_dogs.py

当训练进行10000次迭代时，验证集的准确率达到97%，训练过程中准确率变化的曲线如下图:
![train.png](https://github.com/heiwushi/MyFlow/blob/master/docimgs/train.png)