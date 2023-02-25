import os
import glob
import sys
import logging
import json
import numpy as np
import torch
import shutil


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
    if skip_install: return
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', 
        desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")

def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False
    return spec is not None

def prepare_env():
    if not is_installed("torch") or not is_installed("torchaudio"):
        torch_command = "pip install torch==1.13.1+cu117 torchaudio==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117"
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
    if not is_installed("pypinyin"):
        run_pip(f"install pypinyin", "pypinyin")
    if not is_installed("Cython"):
        run_pip(f"install Cython==0.29.21", "Cython")
    if not is_installed("librosa"):
        run_pip(f"install librosa==0.6.0", "librosa")
    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)

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
    pass

def load_checkpoint(net_g, optim_g, net_d, optim_d, hps):
    model_dir = hps.model_dir
    folders = glob.glob(os.path.join(model_dir, 'epoch_*'))
    if len(folders) == 0:
        return hps.train.learning_rate, 1
        # TODO: 下载预训练权重
        ckpt_folder = os.path.join(model_dir, 'epoch_0')
        os.mkdir(ckpt_folder)
    else:
        folders.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        ckpt_folder = folders[-1]

    load_model(net_g, os.path.join(ckpt_folder, 'generator.ckpt'))
    load_model(net_d, os.path.join(ckpt_folder, 'discriminator.ckpt'))
    
    # TODO: 加上try except, 如果读取不到就不读, 使用原来的随机初始化版本
    optim_g.load_state_dict(torch.load(os.path.join(ckpt_folder, 'optim_g')))
    optim_d.load_state_dict(torch.load(os.path.join(ckpt_folder, 'optim_d')))

    info = torch.load(os.path.join(ckpt_folder, 'info.pt'))
    learning_rate, epoch = info['learning_rate'], info['epoch']
    logger.info("Loaded checkpoint '{}' (epoch {})" .format(ckpt_folder, epoch))
    return lr, epoch


"""
> ls -hl logs/zh/epoch_10
/bin/bash: /opt/conda/lib/libtinfo.so.6: no version information available (required by /bin/bash)
total 1.3G
-rw-r--r-- 1 root root 179M Feb 25 01:00 discriminator.ckpt
-rw-r--r-- 1 root root 152M Feb 25 01:00 generator.ckpt
-rw-r--r-- 1 root root  431 Feb 25 01:00 info.pt
-rw-r--r-- 1 root root 536M Feb 25 01:00 optim_d
-rw-r--r-- 1 root root 456M Feb 25 01:00 optim_g
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
