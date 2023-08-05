### 脚本介绍

1. <span style='color:green'>clear.py</span>，该脚本作用为，删除多余的 XML 文件以及 JPG 图片，根据 XML 生成 TXT 文件。

2. <span style='color:green'>prepare.py</span>，修改测试比例，路径以及添加标签在 71 行处还有路径需要修改。该脚本作用为，生成 val，trainval，test，train 等 TXT 文件。
![图片描述](D:\yolo\algorithm\oneStage\Yolov5-6\picture\prepare.py.png)


3. <span style='color:green'>Leo_F.yaml</span>，修改路径，目标数量以及添加标签。
![图片描述](D:\yolo\algorithm\oneStage\Yolov5-6\picture\Leo.F.yaml.png)


4. <span style='color:green'>yolov5s.yaml</span>，打开修改其中 nc 数量即可，如果想手动修改卷积可以在下面操作。
![图片描述](D:\yolo\algorithm\oneStage\Yolov5-6\picture\yolov5s.yaml.png)

5. <span style='color:green'>train.py</span>，修改 weights，data，batch-size，device，project，name 等配置等。

6. <span style='color:green'>detect.py</span>，修改 weights，source，img-size，conf-thres，iou-thres，device 等配置等。

7. <span style='color:green'>yuzhi.py</span>，修改path，阈值范围等。

# Data

数据存放地址 <span style='color:orange'>Yolov5-6\data_set</span>  
本文采用 VOC 数据集格式，

在目录下新建文件夹 A（存放以 labelimg 生成的 XML 文件）。 

新建文件夹 I（存放以 JPG 结尾的图片）。
![图片描述](D:\yolo\algorithm\oneStage\Yolov5-6\picture\new.png))





### 数据集预处理
1. 在Yolov5-6/data_set下删除labels文件夹。


2. 运行Yolov5-6/data_set下的clear.py，删除多余的XML文件以及JPG图片，根据XML生成TXT文件。


3. 运行Yolov5-6/data_set下的prepare.py，生成<span style='color:red'>val，trainval，test，train</span>等TXT文件。


4. 至此，数据集准备工作完成。
  
### 训练
1. 在Yolov5-6/data/下打开<span style='color:green'>Leo_F.yaml</span>,修改路径，以及类别数量。
2. 在Yolov5-6/models/下打开<span style='color:green'>yolov5s.yaml</span>，修改nc数量即可，如果想手动修改卷积可以在下面操作。
   3. 在Yolov5-6下打开<span style='color:green'>train.py</span>，<span style='color:red'>修改weights，data，
   batch-size，device，project，name</span>等配置等。
       ```python
       parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
       parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')    #选择网络结构，提供了l，m，n，s，x五款
       parser.add_argument('--data', type=str, default=ROOT / 'data/me.yaml', help='dataset.yaml path')  #修改自己的yaml
       parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
       parser.add_argument('--epochs', type=int, default=50)                                               #轮数设置
       parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')        #输入数据大小
       parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')     #输入图片大小
       parser.add_argument('--rect', action='store_true', help='rectangular training')
       parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
       parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
       parser.add_argument('--noval', action='store_true', help='only validate final epoch')
       parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
       parser.add_argument('--noplots', action='store_true', help='save no plot files')
       parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
       parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
       parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
       parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
       parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
       parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
       parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
       parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
       parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
       parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
       parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
       parser.add_argument('--name', default='exp', help='save to project/name')                                       #log生成名称
       parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
       parser.add_argument('--quad', action='store_true', help='quad dataloader')
       parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
       parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
       parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
       parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
       parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
       parser.add_argument('--seed', type=int, default=0, help='Global training seed')
       parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
       parser.add_argument('--entity', default=None, help='W&B: Entity')
       parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
       parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
       parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
- 当然我也提供了一个简单的方法，做了一个shell的集成 first.sh，运行即可。
- ```shell
  # -*- coding: utf-8 -*-
  # @Time : 2023/7/28 11:17
  # @Author : Leo_F
  # @Email : Fshaolei.F@gmail.com
  # @File : first.sh
  # @Software ：shell
  rm -rf path/yolov5-6/data_set/labels/ && python clear.py && python prepare.py  && nohup python train.py
  ```
4. 运行<span style='color:green'>train.py</span>，开始训练。
5. 生成的log文件存放在<span style='color:orange'>runs/train/exp</span>下。
### 压测
1. 训练完成的pt文件存放在<span style='color:orange'>runs/train/exp/weights/</span>下。当然这些位置可以在代码中进行手动修改。
2. 在Yolov5-6下打开<span style='color:green'>detect.py</span>，<span style='color:red'>修改weights，source，imgsz，device，project，name</span>等配置等。
   - ```python
     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'path/last.pt', help='model path(s)')
       parser.add_argument('--source', type=str, default=ROOT / 'data所在位置' ,help='file/dir/URL/glob, 0 for webcam')
       parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
       parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
       parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
       parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
       parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
       parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
       parser.add_argument('--view-img', action='store_true', help='show results')
       parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
       parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
       parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
       parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
       parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
       parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
       parser.add_argument('--augment', action='store_true', help='augmented inference')
       parser.add_argument('--visualize', action='store_true', help='visualize features')
       parser.add_argument('--update', action='store_true', help='update all models')
       parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
       parser.add_argument('--name', default='exp', help='save results to project/name')
       parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
       parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
       parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
       parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
       parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
       parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
3. 运行<span style='color:green'>detect.py</span>时想要查看运行结果图片和阈值。
4. 生成log文件存放在<span style='color:orange'>runs/detect/exp</span>下。当然这些位置可以在代码中进行手动修改。
5. 生成的txt文件存放在<span style='color:orange'>runs/detect/exp/labels</span>下。当然这些位置可以在代码中进行手动修改。

### 评估
在这里写了一个筛选阈值的脚本，可以根据阈值筛选出最优的阈值。
1. 在Yolov5-6下打开<span style='color:green'>yuzhi.py</span>，修改其中路径，阈值范围等。
- ```python
   path = "压测生成图片路径"
   path_result2 = "压测生成txt路径"
   path_result3 ="保存结果路径"
   .......
   if data.split(" ")[0] == "0" and float(data.split(" ")[5])>=0.8:
2. 生成的图片路径储存在特定路径下。
![图片描述](D:\yolo\algorithm\oneStage\Yolov5-6\picture\yuzhi.py.png))
3. 一般就是用压测的图片总量除以特定阈值筛选的图片数量乘以特定阈值。


### 转换
1. 在Yolov5-6下打开<span style='color:green'>export.py</span>，修改其中路径，阈值范围等。
2. 运行<span style='color:green'>export.py</span>，生成的onnx文件存放在<span style='color:orange'>runs/train/exp/weights</span>下。
3. 可以转换成其他格式的文件，具体参考<span style='color:green'>export.py</span>中的注释。 

### 优化 剪枝&&量化（待完成）
相关原理：

Learning Efficient Convolutional Networks Through Network Slimming（https://arxiv.org/abs/1708.06519）

Pruning Filters for Efficient ConvNets（https://arxiv.org/abs/1608.08710）

相关原理见https://blog.csdn.net/IEEE_FELLOW/article/details/117236025

这里实验了三种剪枝方式

## 剪枝方法1

基于BN层系数gamma剪枝。

在一个卷积-BN-激活模块中，BN层可以实现通道的缩放。如下：
<p align="center">
<img src="picture/Screenshot from 2021-05-25 00-26-23.png">
</p>

BN层的具体操作有两部分：

<p align="center">
<img src="picture/Screenshot from 2021-05-25 00-28-15.png">
</p>

在归一化后会进行线性变换，那么当系数gamma很小时候，对应的激活（Zout）会相应很小。这些响应很小的输出可以裁剪掉，这样就实现了bn层的通道剪枝。

通过在loss函数中添加gamma的L1正则约束，可以实现gamma的稀疏化。

<p align="center">
<img src="picture/Screenshot from 2021-05-25 00-28-52.png">
</p>



上面损失函数L右边第一项是原始的损失函数，第二项是约束，其中g(s) = |s|，λ是正则系数，根据数据集调整

实际训练的时候，就是在优化L最小，依据梯度下降算法：

​														𝐿′=∑𝑙′+𝜆∑𝑔′(𝛾)=∑𝑙′+𝜆∑|𝛾|′=∑𝑙′+𝜆∑𝛾∗𝑠𝑖𝑔𝑛(𝛾)

所以只需要在BP传播时候，在BN层权重乘以权重的符号函数输出和系数即可，对应添加如下代码:

```python
            # Backward
            loss.backward()
            # scaler.scale(loss).backward()
            # # ============================= sparsity training ========================== #
            srtmp = opt.sr*(1 - 0.9*epoch/epochs)
            if opt.st:
                ignore_bn_list = []
                for k, m in model.named_modules():
                    if isinstance(m, Bottleneck):
                        if m.add:
                            ignore_bn_list.append(k.rsplit(".", 2)[0] + ".cv1.bn")
                            ignore_bn_list.append(k + '.cv1.bn')
                            ignore_bn_list.append(k + '.cv2.bn')
                    if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
                        m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))  # L1
                        m.bias.grad.data.add_(opt.sr*10 * torch.sign(m.bias.data))  # L1
            # # ============================= sparsity training ========================== #

            optimizer.step()
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
            optimizer.zero_grad()
```

这里并未对所有BN层gamma进行约束，详情见yolov5s每个模块 https://blog.csdn.net/IEEE_FELLOW/article/details/117536808
分析，这里对C3结构中的Bottleneck结构中有shortcut的层不进行剪枝，主要是为了保持tensor维度可以加：

<p align="center">
<img src="picture/Screenshot from 2021-05-27 22-20-33.png">
</p>

实际上，在yolov5中，只有backbone中的Bottleneck是有shortcut的，Head中全部没有shortcut.

如果不加L1正则约束，训练结束后的BN层gamma分布近似正太分布：

<p align="center">
<img src="picture/Screenshot from 2021-05-23 20-19-08.png">
</p>

是无法进行剪枝的。

稀疏训练后的分布：

<p align="center">
<img src="picture/Screenshot from 2021-05-23 20-19-30.png">
</p>

可以看到，随着训练epoch进行，越来越多的gamma逼近0.

训练完成后可以进行剪枝，一个基本的原则是阈值不能大于任何通道bn的最大gamma。然后根据设定的裁剪比例剪枝。

剪掉一个BN层，需要将对应上一层的卷积核裁剪掉，同时将下一层卷积核对应的通道减掉。

这里在某个数据集上实验。

