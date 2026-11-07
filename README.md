# 图像处理课程作业

李星翰　20245624

## 1 Github链接

仓库：https://github.com/lee11-star/MachineLearningHomework

完整报告见 [report/1.报告.docx](report/1.报告.docx)，实验代码、课程数据和已训练模型见 [task4](task4)。

## 2 任务描述

从彩色眼底图像中提取血管，输出逐像素二值分割结果。本项目使用视野内归一化RGB和绿色通道局部对比度构成四通道输入，由小型U-Net预测血管概率，再完成阈值化、512×512转换和RLE编码。

实验以现有基线为起点，对比扩大网络宽度及训练切块、调整训练配置与增加推理方向后的结果。

## 3 数据描述

课程提供20张有标注图像和20张无公开标注的测试图像，原图宽565、高584像素。开发阶段按整图划分为16张训练图和4张验证图，验证图编号固定为8、10、15、16。配置选定后，使用全部20张有标注图像重新训练最终模型，再生成测试结果。

训练数据、人工标注和课程提供的编码脚本保存在task4/segmentation目录中。血管在标注和提交二值图中均以白色表示。

## 4 代码修改描述

核心代码及两套实际配置均保存在task4中：

| 内容 | 基线 | 改进版 |
|---|---|---|
| U-Net起始宽度 | 16 | 24 |
| 训练切块 | 128×128 | 192×192 |
| 批量大小 | 16 | 12 |
| 开发训练上限 | 80轮 | 120轮 |
| 验证选中轮数 | 45 | 65 |
| 推理方向 | 4 | 8 |
| 选中阈值 | 0.475 | 0.400 |

输入构造、损失和输出规则在两套配置间共用。`vessel/data.py`实现视野提取、四通道输入和切块增强；`vessel/model.py`实现U-Net和方向还原；`train.py`负责训练及权重选择；`compare_models.py`比较候选；`predict.py`和`vessel/submission.py`生成预测及提交编码。

## 5 结果对比

主要指标为按课程编码脚本的转换规则处理后，四张验证图Dice的算术平均：先在原尺寸阈值化，再双线性缩放到512×512，并以非零像素作为前景。预测和人工标注执行同一转换。

| 策略 | 转换后平均Dice | 原尺寸平均Dice | 阈值 |
|---|---:|---:|---:|
| 基线四方向 | 0.848591 | 0.813110 | 0.475 |
| 基线八方向 | 0.848868 | 0.814359 | 0.525 |
| 改进八方向 | 0.851209 | 0.815958 | 0.400 |
| 两模型等权平均 | 0.850760 | 0.816138 | 0.450 |

改进八方向相对基线四方向，主要Dice提高0.002618，约0.262个百分点。原尺寸下四图汇总误分像素从25,561降至25,110，漏分从16,761降至16,502。固定共同阈值0.475时，基线四方向与改进八方向的平均Dice分别为0.848591和0.850920。

这些分数来自同一组四张开发验证图，该组图也用于选择轮数、阈值和方案。本次比较的是联合配置；没有将收益拆成单项改动的贡献。测试图未提供公开真值，不报告测试Dice。报告另列逐图变化、边界带召回、训练开销和预测图。

## 运行项目

下载仓库后进入task4，安装Python依赖，再使用附带权重生成测试结果，无需重新训练：

```bash
cd task4
python -m pip install -r requirements.txt
python predict.py --ensemble outputs/improved/ensemble_selection.json --output outputs/reproduced
python verify_submission.py --output outputs/reproduced
```

运行关键测试及复算报告对比：

```bash
python -m unittest discover -s tests -v
python compare_models.py --output outputs/recheck_selection
python analyze_saved_results.py --output outputs/report_analysis.json
```

完整训练命令、环境和文件说明见 [task4/README.md](task4/README.md)。模型、训练日志和验证预测均已附带。仓库不包含虚拟环境、临时缓存或重复生成的测试图片目录。
