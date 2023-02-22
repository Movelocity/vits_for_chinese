### 中文语音合成

Copied from this repo: https://github.com/jaywalnut310/vits

### 2023-02-22 更新日志
```
1.train_single_card.py 可以只使用单张显卡训练，降低配置难度

2.取消 eval_loader, 直接从验证集读取前四条并测试合成效果，可以在 tensorboard 查看

3.取消单 speaker 和多 speaker 的训练区分，统一使用train.py或train_single_card.py，便于后续分享模型主干，分离说话者编码。

4.只有一个speaker时，默认speaker id=0
```

### 2023-01-27 更新日志
```
增加了中文编码方法，使用pingying库以style3将中文汉字转为拼音，在拼音的基础上划分声母和韵母。
```
**语音数据集制作方法** [notebooks/make_dataset.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/make_dataset.ipynb)

**训练方法详见** [notebooks/train.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/train.ipynb)

**使用方法详见** [notebooks/infer.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/infer.ipynb)

**windows平台调用vgmstream的脚本** [notebooks/convert.bat.txt](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/convert.bat.txt) (使用时去掉txt后缀)

视频持续更新。。。

项目结构修改中，使用前建议Fork一份到自己的仓库。

如遇到 Bug 请提交 Issue 或联系UP主。