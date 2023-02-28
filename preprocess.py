import argparse
import text
from data_utils import load_filepaths_and_text
import random

# python preprocess.py --mix True --filelists filelists/elysia_collect.txt.cleaned filelists/paimon_collect.txt.cleaned
def mix(filelists, output_name='filelists/mixed.txt.cleaned'):
    """合并多个语音标签集"""
    total = []
    for file in filelists:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')
        for d in data:
            if '|' in d:
                total.append(d)
    with open(output_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(total))

def validate_tokens(filelist):
    """验证标签集是否存在不能识别的字符"""
    with open(filelist, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')

    err_list = []
    for dat in data:
        if '|' not in dat: continue
        row = dat.split('|')
        try:
            text.tokens2ids(row[-1])
        except:
            err_list.append(dat)
    msg = '需要修改的有:\n'+'\n'.join(err_list) if len(err_list)>0 else f'file {filelist} is alright.'
    return msg

def clean_tokens(filelist):
    """直接去除存在不能识别的字符的数据"""
    save_list = []
    with open(filelist, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
    for dat in data:
        if '|' not in dat: continue
        row = dat.split('|')
        try:
            text.tokens2ids(row[-1])
            save_list.append(dat)
        except: pass
    err_cnt = len(data) - len(save_list) - 1
    if err_cnt == 0: return 'Tokens alright'
    import shutil
    shutil.move(filelist, filelist+'.bak')
    with open(filelist, 'w', encoding='utf-8') as f:
        f.write('\n'.join(save_list))
    return 'bak file saved: %s, %d cleared.'%(filelist+'.bak', err_cnt)


def preprocess(filelist, out_extension='cleaned'):
    """
    filelist 指定文件,需要包含路径信息,

    text_index 在单 speaker 模式是 1 ,在多 speaker 模式是 2
    
    处理完毕后, 就会在源文件路径生成以 .cleaned 结尾的文件, 供训练时加载语音标签
    
    最后在configs里的你配置文件中指定 "training_files":"filelists/XXXX.txt.cleaned",
    """
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
        original_text = filepaths_and_text[i][-1]
        cleaned_text = text.pypinyin_g2p_phone(original_text)
        filepaths_and_text[i][-1] = cleaned_text
        filepaths_and_text[i][0] = "./wave_data/"+filepaths_and_text[i][0]

    new_filelist = filelist + "." + out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])


# python preprocess.py --text_index 2 --filelists filelists/elysia_collect.txt filelists/paimon_collect.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False)
    parser.add_argument("--mix", default=False)
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1,type=int)  # 数据集标签文件中文本所在的列(从0开始数)
    parser.add_argument("--filelists", nargs="+", default=[
                        "filelists/paimon_train.txt", "filelists/paimon_valid.txt"])

    args = parser.parse_args()
    print(args)

    if args.test:
        for filelist in args.filelists:
            validate_tokens(filelist, args.text_index)
        exit()

    if args.mix:
        mix(args.filelists)
        exit()

    for filelist in args.filelists:
        preprocess(filelist)
