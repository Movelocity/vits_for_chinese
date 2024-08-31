### 中文语音合成

> [基于 VITS 项目修改](https://github.com/jaywalnut310/vits)

VITS 是一个用于文本合成语音的项目，虽然截至 2024-08-31 社区已经出现了很多更好用的语音合成模型，比如 BertVITS2, GPT-SoVITS 等，效果非常好，但模型推理普遍对算力要求高。而本项目基于原始的 VITS 进行修改并加入汉语拼音的支持，模型文件大小不会超过170MB。训练出来的模型也适合在低算力（CPU）场景下合成中文语音，同时也适合入门深度学习的朋友学习和修改代码。

如果你看懂了 VITS 的代码，再去学习其它和 VITS 相关的项目，就会相对容易看懂。

### 使用方式
- **语音数据集制作方法** [notebooks/make_dataset.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/make_dataset.ipynb)
- **可以用 whisper 识别语音**: https://github.com/openai/whisper
- **训练方法详见** [notebooks/train.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/train.ipynb)
- **使用方法详见** [notebooks/infer.ipynb](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/infer.ipynb)
- **windows平台调用vgmstream的脚本** [notebooks/vgm_decode.py](https://github.com/Movelocity/vits_for_chinese/blob/main/notebooks/vgm_decode.py) 。不推荐使用 `.bat` 的版本，使用 `.py` 的版本易读性更高
  - 打开方式：命令行运行 `python vgm_decode.py`
  - 或者右键打开文件，使用 python 打开

> ### pytorch版本注意事项
> 去官方页面根据自己的环境选择合适的版本: https://pytorch.org/get-started/locally/
>
> 比如要在linux上安装 pytorch1.13.1 以支持混合精度训练(训练速度翻倍)，在官方页面选择后得到以下命令:
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
> ```
> torchvision 可以不装。但必须安装 torch 和 torchaudio，直接使用官方链接可以避免版本差异导致的报错。

### 常见问题
1. 云GPU环境出现类似 ***google::protobuf::FatalException*** 的错误

    猜测是 tensorflow 版本过低导致的问题，建议卸载当前环境的 tensorflow 后重新尝试运行。

2. kaggle上训练模型，怎么看 tensorboard?

    使用 **frp** (fast reverse proxy) 暴露 kaggle 端的网络端口，然后就可以在其它电脑上用浏览器查看 tensorboard 页面了。(需要有公网IP，自己租个云服务器或者自己的主机就有公网IP，才能让frp找到你的主机并且转发tensorbord的url)。可以参考这个kaggle笔记本: https://www.kaggle.com/code/hollway/train-vits

3. local variable 'fs' referenced before assignment

    部分标贝数据集的音频无法用 scipy.io.wavfile.read 读取，会报错 

4. tensorboard 看到的频谱图很暗

    - `scipy.io.wavfile.read` 读取的音频，数值区间为 0~32768，需要手动归一化到 0~1 区间: `wave = wave/32768`
    - `torchaudio` 读取音频，波形值域在 0~1 区间，不用重新归一化

5. 编译 monotonic_align 模块可能会失败

    - windows x64/amd 可使用预编译的文件

6. 为什么要新开一个仓库？
    1. 取消多卡训练，降低配置难度，本地训练以及kaggle训练更友好。如果把多卡换成可选的代码块来执行
    2. 增加内网穿透功能，使得你在colab, kaggle等云平台上面也能正常查看tensorboard日志

后续可以考虑移除说话者编码，纯粹用lora来调整和分享声线，而不用调整主干。也就是一个模型通用，然后声线可以额外训练并保存到一个小文件中。好处是微调收敛快，特征文件小，主干模型可以全社区共享。

项目结构修改中，使用前建议Fork一份到自己的仓库。如遇到 Bug 请提交 Issue 或联系UP主。

**2023-01-27 更新日志**
```
增加了中文编码方法，使用pingying库以style3将中文汉字转为拼音，在拼音的基础上划分声母和韵母。
```

**2023-02-22 更新日志**
```
1.train_single_card.py 可以只使用单张显卡训练，降低配置难度
2.取消 eval_loader, 直接从验证集读取前四条并测试合成效果，可以在 tensorboard 查看
3.取消单 speaker 和多 speaker 的训练区分，统一使用train.py或train_single_card.py，便于后续分享模型主干，分离说话者编码。
4.只有一个speaker时，默认speaker id=0
```

**2023-02-25 更新日志**
```
1.加入 python 版本检查，只能使用 python3，防止出现意料之外的错误
2.检查多个依赖包并自动安装
3.不用手动编译 monotonic_align，现在由程序自动编译，具体代码在 utils.py
4.合并了 train_single_card.py 和 trian.py，统一用 train.py
```

**2023-02-28 更新日志**
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
