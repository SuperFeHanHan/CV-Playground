
# DETR的结构
![](/imgs/2-DETR/structure.png)

- resnet50结构：使用的时候去除最顶上的fc和avgpool那层，即取layer4的输出结果。