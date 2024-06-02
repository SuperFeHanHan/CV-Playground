> [参考](https://github.com/jungomi/math-formula-recognition)

TODO:
- [x] 数据
- [ ] Crop White Space around formula
- [ ] 图片数据增强
- [ ] VisionEncoderDecoder的训练代码

# 0. 数据来源

## 0.1 CHROME
- 原始数据集：[CROHME](https://www.kaggle.com/datasets/3a6d371ee185e037fbc0d4eba075053aeaa54077bda6de3ed1b8f988891fa9ea)
- 利用自己写的脚本(latex_to_img.py)将正确的latex公式变为了png，并做了一些处理:
    - 去处了一些没有实质内容的图片：
        - test/2014/RIT_2014_216.png
        - test/2014/RIT_2014_191.png 
    - 修正了一些公式
        - test/2013/rit_4275_4.png
        - test/2013/rit_42185_1.png
        - test/2014/RIT_2014_309.png
    - 将原始的train和test合并为一个train，对应的latex的公式放在latex_train。
    - 对应的数据放在groundtruth.csv和groundtruth_latex.csv中
        - 其中latex中合并了一些公式重复的项目，因为手写的是不一样的但是latex公式图片是一样的。主要集中于MfrDB,expressmatch,MathBrush,KAIST
        - 公式部分经过了一些替换，比如\sin变为\mathrm{sin}
- 修正完后的数据统计：
```text
groundtruth_latex.csv, 一共7725条数据
2013             670
2014             973
2016            1145
HAMEX           2478
KAIST             49
MathBrush       1880
MfrDB            470
expressmatch      36
extension         24

groundtruth.csv, 一共11637条数据
class
2013             671
2014             984
2016            1147
HAMEX           2484
KAIST           1051
MathBrush       2989
MfrDB           1666
expressmatch     620
extension         25
Name: path, dtype: int64
```
# 1. VisionEncoderDecoder模型构建
- 尝试利用VisionEncoderDecoder模型进行构建
