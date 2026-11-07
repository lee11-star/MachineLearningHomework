# 眼底血管分割课程项目

李星翰　20245624

项目包含课程数据、训练和推理程序、基线与改进模型，以及报告对比所用的验证预测。
运行命令以本README所在的task4目录为工作目录。

## 环境

原实验环境为Windows、Python 3.12、PyTorch 2.7.0+cu126和RTX 4060 Laptop GPU。
依赖版本见requirements.txt。先安装依赖，再在task4目录运行下列命令：

```powershell
python -m pip install -r requirements.txt
$taskPython = 'python'
```

如果使用conda或虚拟环境，请先激活环境；也可以把`$taskPython`设为相应解释器路径。
程序自动选择CUDA或CPU，推理使用已附权重，不需重新训练。

## 生成测试结果

```powershell
& $taskPython predict.py --ensemble outputs/improved/ensemble_selection.json --output outputs/reproduced
& $taskPython verify_submission.py --output outputs/reproduced
```

第一条命令生成20张测试图的概率、原尺寸及512×512二值图、反色图和CSV；第二条对照官方编码器检查结果。
已保存的改进版CSV为outputs/improved/submission.csv。新的输出写在outputs/reproduced中。

## 复算报告中的对比

```powershell
& $taskPython compare_models.py --output outputs/recheck_selection
& $taskPython analyze_saved_results.py --output outputs/report_analysis.json
& $taskPython -m unittest discover -s tests -v
```

compare_models.py比较基线四方向、基线八方向、改进八方向和两模型平均。
analyze_saved_results.py从标注、保存概率及训练日志复算像素错误、阈值曲线、数据分布和训练开销。
验证图固定为8、10、15、16，主要指标为模拟官方转换后的四图平均Dice。

## 从头训练

```powershell
# 基线配置
& $taskPython train.py --epochs 80 --steps-per-epoch 50 --seed 42 --split-seed 42 --output outputs/retrain_baseline

# 改进配置
& $taskPython train.py --width 24 --patch-size 192 --batch-size 12 --epochs 120 --steps-per-epoch 50 --seed 42 --split-seed 42 --tta d4 --output outputs/retrain_context

# 采用新训练选定的轮数，用全部20张标注图重训
& $taskPython train.py --all-data --selection outputs/retrain_context/summary.json --output outputs/retrain_context_full
& $taskPython predict.py --checkpoint outputs/retrain_context_full/model.pt --output outputs/retrain_predictions
& $taskPython verify_submission.py --output outputs/retrain_predictions
```

标注和用于提交的二值图均以白色表示血管。反色图供显示使用，不能直接作为官方编码器的前景输入。
报告分数来自开发阶段的固定四张验证图，测试图没有公开真值。原始数据与官方segmentation_to_csv.py均保留。
