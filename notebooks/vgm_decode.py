import os
import subprocess
from pathlib import Path
from tqdm import tqdm


# 使用方式，直接使用 python 打开本文件
# 如果提示缺少tqdm，就运行 pip install tqdm
""" 使用示例
E:\repos\GPT-SoVITS> python vgm_decode.py
vgm stream 便捷脚本，用于批量解码游戏语音。
输入(拖入)待解码文件夹/文件路径(wav文件)
>>> E:\dataset\audio_starrail\ext1
100%|████████████████████████████████████████████████████████████████████████████████| 923/923 [00:17<00:00, 53.77it/s]
处理完毕!
"""

def validate_exe(path_to_exe):
    try:
        result = subprocess.run([path_to_exe], capture_output=True, text=True)
        if result.stderr and "missing input file" in result.stderr:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def process_files(path_to_exe, path_to_files):
    # 解码原音频文件，生成 _done.wav 后缀的文件后删除原文件
    wav_files = []
    if os.path.isdir(path_to_files):
        wav_files = list(Path(path_to_files).rglob("*.wav"))
    elif os.path.isfile(path_to_files) and path_to_files.endswith('.wav'):
        wav_files.append(Path(path_to_files))
    else:
        print(f"出错啦: {path_to_files}")
        return
    
    wav_files = [f for f in wav_files if not f.name.endswith('_done.wav')]

    for wav_file in tqdm(wav_files):
        new_name = str(wav_file).replace('.wav', '_done.wav')
        subprocess.run([path_to_exe, str(wav_file), "-o", new_name], capture_output=True)
        os.remove(wav_file)

def main():
    print('vgm stream 便捷脚本，用于批量解码游戏语音。')
    path_to_exe = "F:/Tools/vgmstream/vgmstream-cli.exe"  # or "test.exe", 使用你自己的实际目录
    if not validate_exe(path_to_exe):
        path_to_exe = input(f"{path_to_exe} 未找到vgm工具文件，请手动指定其路径\n>>> ")
        if not validate_exe(path_to_exe):
            print("请输入 test.exe 或 vgmstream-cli.exe 的路径")
            return

    path_to_files = input("输入(拖入)待解码文件夹/文件路径(wav文件)\n>>> ")
    process_files(path_to_exe, path_to_files)

    print("处理完毕!")

if __name__ == "__main__":
    main()
