import os
import glob
import sys
import logging
import json
import numpy as np
import shutil
import importlib
import subprocess
import time
import text

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

def check_py_version():
    if sys.version_info.major != 3:
        print("""
请使用 Python3
如果使用 sudo python train.py ... 运行程序, 大概率用的是python2, 试一下 sudo python3 train.py
其它情况请在命令行使用 python --version 检查版本
""")

# Copied from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/launch.py
def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    """执行命令"""
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")
        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

python = sys.executable
def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)

index_url = os.environ.get('INDEX_URL', "")
def run_pip(args, desc=None):
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', 
        desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")

def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False
    return spec is not None

def frp_for_online_tensorboard(server_ip, server_port, local_port, remote_port):
    if not is_installed("tensorboard"):  # 保证 tensorboard 能用
        run_pip(f'install protobuf=<3.20.0')
        run_pip(f"install tensorboard==2.3.0", "tensorboard")

    import platform
    if platform.system() == "Linux" and not os.path.exists('./frp_0.37.0_linux_amd64.tar.gz'):
        run(
            'wget -nc https://github.com/fatedier/frp/releases/download/v0.37.0/frp_0.37.0_linux_amd64.tar.gz'
            '&& tar -zxvf frp_0.37.0_linux_amd64.tar.gz'
            '&& mv frp_0.37.0_linux_amd64 frp37',
            desc="正在安装frp(代理)"
        )
        client_config = \
"""
[common]
server_addr = {0}
server_port = {1}

[web]
type = http
local_port = {2}
remote_port = {3}
custom_domains = {0}
""".format(server_ip, server_port, local_port, remote_port)

        with open('./frp37/frpc.ini', 'w') as f:
            f.write(client_config)
            print('配置已写入frpc.ini:')
            print(client_config)

        run('./frp37/frpc -c ./frp37/frpc.ini > ./frp37/output.txt 2>&1 &')
        time.sleep(5)
        with open('./frp37/output.txt', 'r') as f:
            print(f.read())
        tb_link = f'http://{server_ip}:{server_port}'
    else:
        print('非Linux系统, 默认本地使用, 不用安装frp')
        tb_link = f'http://localhost:{local_port}'
    run(f'tensorboard --logdir ./logs --host 0.0.0.0 --port {local_port} &')
    print(f'已启动TensorBoard, 训练产生记录后后再打开{tb_link}')

def prepare_env():
    if not is_installed("pypinyin"):
        run_pip(f"install pypinyin", "pypinyin")
    if not is_installed("matplotlib"):
        run_pip(f"install matplotlib", "matplotlib")
    if not is_installed("Cython"):
        run_pip(f"install Cython==0.29.21", "Cython")
    if not is_installed("librosa"):
        run_pip(f"install librosa==0.6.0", "librosa")
    
    try:
        import monotonic_align.maximum_path
    except:
        print('正在编译monotonic_align模块')
        run(f"cd monotonic_align; {python} setup.py build_ext --inplace; cd ..", live=True)

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)

try:   # 简单检查一下 PyTorch 有没有安装，没有的话就自动安装咯
    import torch
except:
    torch_command = "pip install torch==1.13.1+cu117 torchaudio==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
    run(f'"{python}" -m {torch_command}', "Installing torch and torchaudio", "Couldn't install torch", live=True)
    
def load_model(model, saved_state_dict):
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():  # 如果配置文件比原来的模型增加了模块，就提醒一下
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info(f"{k} 预训练权重和当前配置不匹配，使用默认权重.")
            new_state_dict[k] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

# TODO: 下载共享的预训练权重
def from_pretrained(model, link):
    name = link.split('/')[-1]
    try:
        run(f'wget -nc -O ./{name} {link}',
                desc='正在加载预训练权重', errdesc='预训练权重未加载')
    except RuntimeError:
        return
    # 考虑使用基于Transformer的预训练Tokenizer，但是可能覆盖不了每个词的读音，暂时搁置
    
    ld_ckpt = torch.load(f'./{name}', map_location='cpu')
    if isinstance(ld_ckpt, dict) and 'model' in ld_ckpt.keys():
        saved_state_dict = ld_ckpt['model']
    else:
        saved_state_dict = ld_ckpt

    state_dict = model.state_dict()
    init_speaker_emb, init_vocab_emb = False, False

    try:
        if model.state_dict()['enc_p.emb.weight'].shape != saved_state_dict['enc_p.emb.weight'].shape:
            init_vocab_emb = True
        if model.state_dict()['emb_g.weight'].shape != saved_state_dict['emb_g.weight'].shape:
            init_speaker_emb = True
    except:
        init_speaker_emb = True

    new_state_dict= {}
    for k, v in state_dict.items():  # 如果配置文件比原来的模型增加了模块，就提醒一下
        if k=='emb_g.weight' and init_speaker_emb:
            new_state_dict[k] = torch.randn(hps.data.n_speakers, hps.model.gin_channels)
            torch.nn.init.normal_(new_state_dict[k], 0.0, hps.model.gin_channels**-0.5)
            print('Randomly init speaker embeddings.')
        elif k=='enc_p.emb.weight' and init_vocab_emb:
            new_state_dict[k] = torch.randn(len(text.symbols), hps.model.hidden_channels)
            torch.nn.init.normal_(new_state_dict[k], 0.0, hps.model.hidden_channels**-0.5)
            print('Randomly init vocab embeddings.')
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

def load_checkpoint(net_g, optim_g, net_d, optim_d, hps):
    model_dir = hps.model_dir
    folders = glob.glob(os.path.join(model_dir, 'epoch_*'))
    if len(folders) == 0:
        # 测试链接来自仓库: https://github.com/SayaSS/vits-finetuning，若侵权请联系我删除
        from_pretrained(net_g, 'https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth')
        from_pretrained(net_d, 'https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth')
        return hps.train.learning_rate, 1
    else:
        folders.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        ckpt_folder = folders[-1]

    load_model(net_g, os.path.join(ckpt_folder, 'generator.ckpt'))
    load_model(net_d, os.path.join(ckpt_folder, 'discriminator.ckpt'))

    try:
        optim_g.load_state_dict(torch.load(os.path.join(ckpt_folder, 'optim_g')))
        optim_d.load_state_dict(torch.load(os.path.join(ckpt_folder, 'optim_d')))
    except ValueError:
        print('优化器加载失败，使用随机初始化...')

    info = torch.load(os.path.join(ckpt_folder, 'info.pt'))
    learning_rate, epoch = info['learning_rate'], info['epoch']
    logger.info("Loaded checkpoint '{}' (epoch {})" .format(ckpt_folder, epoch))
    return learning_rate, epoch


"""
> ls -hl logs/zh/epoch_7
/bin/bash: /opt/conda/lib/libtinfo.so.6: no version information available (required by /bin/bash)
total 991M
-rw-r--r-- 1 root root 179M Feb 25 01:11 discriminator.ckpt
-rw-r--r-- 1 root root 152M Feb 25 01:11 generator.ckpt
-rw-r--r-- 1 root root  431 Feb 25 01:11 info.pt
-rw-r--r-- 1 root root 357M Feb 25 01:11 optim_d
-rw-r--r-- 1 root root 304M Feb 25 01:11 optim_g
"""

def save_checkpoint(net_g, optim_g, net_d, optim_d, learning_rate, epoch, model_dir):
    """
    保存训练状态, 默认保留最新的两个epoch文件夹, 旧的直接删去
    
    保留两个是为了防止模型进入过拟合状态，
    
    感觉模型过拟合的时候，可以手动删掉过拟合的文件夹，然后重新启动训练
    """
    checkpoint_folder = os.path.join(model_dir, f'epoch_{epoch}')

    logger.info("Saving ckpt to {}".format(checkpoint_folder))

    g_state_dict = net_g.module.state_dict() if hasattr(net_g, 'module') else net_g.state_dict()
    d_state_dict = net_d.module.state_dict() if hasattr(net_d, 'module') else net_d.state_dict()

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    savelist = {
        'generator.ckpt': g_state_dict, 
        'discriminator.ckpt': d_state_dict,
        'optim_g': optim_g.state_dict(),
        'optim_d': optim_d.state_dict()
    }
    for k, v in savelist.items():
        torch.save(v, os.path.join(checkpoint_folder, k))

    torch.save({'learning_rate': learning_rate, 'epoch': epoch}, os.path.join(checkpoint_folder, 'info.pt'))

    # 删除旧的检查点
    folders = glob.glob(os.path.join(model_dir, 'epoch_*'))
    folders.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if len(folders) > 2:
        shutil.rmtree(folders[0])    #递归删除文件夹
        logger.info(f'remove old ckpts: {folders[0]}')


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)

def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

import torchaudio
def load_wav_to_torch(full_path):
    wav, sr = torchaudio.load(full_path)
    return torch.FloatTensor(wav[0]), sr

def get_hparams(args):
    model_dir = os.path.join("./logs", args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
  
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams

def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams

def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.INFO)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.INFO)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
