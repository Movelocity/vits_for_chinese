### 中文语音合成

Copied from this repo: https://github.com/jaywalnut310/vits

- **语音数据集制作方法** [notebooks/make_dataset.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/make_dataset.ipynb)

- 可以用 whisper 识别语音: https://github.com/openai/whisper

- **训练方法详见** [notebooks/train.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/train.ipynb)

- **使用方法详见** [notebooks/infer.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/infer.ipynb)

- **windows平台调用vgmstream的脚本** [notebooks/convert.bat.txt](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/convert.bat.txt) (使用时去掉txt后缀)

### 2034-02-28 更新日志
```
1.加入frp内网穿透并且可以自动打开 tensorboard 
在本地、colab、featurize等支持查看tensorboard的地方不需要这个函数，主要在kaggle上使用

import utils
utils.frp_for_online_tensorboard(
    server_ip='xxx.xxx.xxx.xxx',  # 你的公网IP
    server_port='7000', # 服务器上运行的frp的 bind_port
    local_port='7860',  # 在训练主机上打开的网络端口
    remote_port='7860'  # 在服务器上运行的frp的 vhost_http_port
)
详见：https://www.kaggle.com/code/hollway/train-vits
```

### 2023-02-25 更新日志
```
1.加入 python 版本检查，只能使用 python3，防止出现意料之外的错误

2.检查多个依赖包并自动安装

3.不用手动编译 monotonic_align，现在由程序自动编译，具体代码在 utils.py

4.合并了 train_single_card.py 和 trian.py，统一用 train.py
```

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


### pytorch版本注意事项

建议去官方页面根据自己的环境选择合适的版本:
https://pytorch.org/get-started/locally/

比如要在linux上安装 pytorch1.13.1 以支持混合精度训练(训练速度可以翻倍)，在官方页面选择后得到以下命令:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
其中 torchvision 可以不装。但一定要安装 torch 和 torchaudio，直接使用官方链接可以避免版本差异导致的报错。

- 如果你的云GPU环境出现类似 ***google::protobuf::FatalException*** 的错误，网上对这个问题是没有答案的，目前猜测是 tensorflow 版本过低导致的问题，我的建议是直接卸载 tensorflow。

如果想在kaggle训练模型，可以使用 **frp** (fast reverse proxy) 暴露 kaggle 端的网络端口，然后就可以在其它电脑上用浏览器查看 tensorboard 页面了。(需要有公网IP，自己租个云服务器或者你的Wifi就有公网IP，不然暂时没有办法在kaggle上实时查看训练效果)
https://www.kaggle.com/code/hollway/train-vits


视频持续更新。。。

项目结构修改中，使用前建议Fork一份到自己的仓库。

如遇到 Bug 请提交 Issue 或联系UP主。

音频读取避坑
```
1.部分标贝数据集的音频无法用 scipy.io.wavfile.read 读取，会报错 local variable 'fs' referenced before assignment

2.使用 scipy.io.wavfile.read 读取的音频波形值域在 0~32768
一般需要手动归一化到 0~1 区间: wave = wave/32768

3.使用 torchaudio.load 读取的音频波形值域在 0~1
如果继续用上面的代码归一化, 值域将变为 0 ~ 1/32768。这种情况可以根据频谱亮度看出来。
```

写在最后。后续可以考虑移除说话者编码，纯粹用lora来调整和分享声线，而不用调整主干。也就是一个模型通用，然后声线可以额外训练并保存到一个小文件中。好处是微调收敛快，特征文件小，主干模型可以全社区共享。