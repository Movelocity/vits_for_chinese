from pypinyin import pinyin, Style
import re

_pad = list("_")

_punctuation = list("!'(),.:;?-")

# ["，", "。", "？", "！",'2','1']
# "[", "]" 先保留，后面可以用来插入其它语言注释

# 韵母
_initials = [
    'b','p','m','f',
    'd','t','n','l',
    'g','k','h','j','q','x',
    'w','y',
    "zh","ch","sh","z","c","s","r"
] # 单字母的要往后放，让多字母先匹配

# 声母
_phonetic_symbols = [
    'a','o','e','i','u','v','ê',
    'ai','ei','ui','ao','ou','iu','ie','ve','er',
    'an','en','in','un','vn','on',
    'ang','eng','ing','ong',
    'ia','iao','ian','iang','iong',
    'uo','uai','uei','ua','ue','uan','uen','uang','ueng'
]
_additional_pho_symbols = [
    "n1", "n2", "n3", "n4", "m1", "m2", "m3", "m4"
]
_tones = ["", "1", "2", "3", "4"]  # 声调

symbols = _pad + \
          _punctuation + \
          _initials + [i + j for i in _phonetic_symbols for j in _tones] +\
          _additional_pho_symbols

_letters_map = {
    "A": "ei-1",
    "B": "b-i-1",
    "C": "x-i-1",
    "D": "d-i-1",
    "E": "y-i-1",
    "F": "ei-4-f-u-3",
    'G': "g-v-1",
    "H": "ei-4-ch-i-3",
    "I": "ai-1",
    "J": "zh-ei-1",
    "K": "k-ei-1",
    "L": "e-1-l",
    "M": "ei-m",
    "N": "ei-n",
    "O": "o-1",
    "P": "p-i-1",
    "Q": "k-i-1-y-u-1",
    "R": "a-1",
    "S": "ei-4-s-i-3",
    "T": "t-i-1",
    "U": "y-u-1",
    "V": "w-ei-1",
    "W": "d-a-3-b-u-4-l",
    "X": "ei-4-k-e-3-x-i-3",
    "Y": "w-ai-1",
    "Z": "z-e-4"
}

_punct_map = {
    "、": ",",
    "，": ",",
    "。": ".",  # 朗读代码的话把 '.' 换成 '点'
    "！": "!",
    "？": "?",
    "（": "(",
    "）": ")",
    "：": ":",
    "；": ";",
}

_num_map = {
    "0": "l-ing2",
    "1": "y-i1",
    "2": "er4",
    "3": "s-an1",
    "4": "s-i4",
    "5": "w-u3",
    "6": "l-iu4",
    "7": "q-i1",
    "8": "b-a1",
    "9": "j-iu3",
}

_special_map = {
    "+": "加",
    "-": "减",
    "*": "乘",
    "/": "斜杠",
    "\\": "反斜杠",
    "=": "等于",
    "<": "小于",
    ">": "大于",
    "@": "艾特",
}

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
# _id_to_symbol = {i: s for i, s in enumerate(symbols)}

def clean_text(text, pad_token='_'):
    """
    当pinyin模块遇到处理不了的字符块时, 这个会被调用, 处理字符串
    input: str
    output: str
    1. {} 中的内容会被完整保留
    2. {} 外的字母全部变成大写，然后根据 _letters_map 映射到对应的子串
    3. {} 外，如果有和 _punct_map.keys() 符合的字符，也会被该字典对应的值取代
    3. 结果中，不包含花括号，各个元素用-分隔
    >>> clean_text('{n i 3 h ao 3}， aBC{p iao 4 l iang 4}') # 输入中不能有"-"
    "n-i3-h-ao-3-,-ai-bii-cij-p-iao-4-l-iang-4"
    TODO: 根据上面的要求完成这个函数
    """
    result = []
    si = 0
    in_brackets = False
    for i, char in enumerate(text):
        if char == '{':
            in_brackets = True
            si = i
        elif char == '}' and in_brackets:
            in_brackets = False
            result.extend(text[si+1:i].split())
        elif in_brackets == False:
            if char in _letters_map: # 字母
                result.append(_letters_map[char.upper()])
            elif char in _punct_map: # 标点
                result.append(_punct_map[char])
            elif char in _num_map:   # 数字
                result.extend(_num_map[char].split('-'))
            elif char != ' ':        # 其它
                result.append(pad_token)

    return '-'.join(result)

def special_token_cleaner(text):
    for k, v in _special_map.items():
        text = text.replace(k, v)
    return text

def phonemizer(text):
    """
    其中，{}内的合理内容完整保留，{}外的内容经过一系列替换，
    1."你"替换成"我", 
    2.使用 _punct_map 把全部全角标点替换成英文标点。
    3.阿拉伯数字替换成汉字
    4.替换后的内容相对原来的位置不变
    # """
    text = special_token_cleaner(text)
    pre_phoneme = pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=False, errors=clean_text)
    phoneme = []
    for p in pre_phoneme:
        p0 = p[0]      # 选择匹配的第一个结果
        if '-' in p0:  # 匹配中存在 clean 处理得到的列表
            phoneme.extend(p0.split('-'))
            continue

        match = False
        for y in _initials:
            if p0.startswith(y):
                if len(p0)>len(y) and p0[len(y)] not in ['1','2','3','4']:
                    phoneme.extend([y, p0[len(y):]])
                    match = True
                    break
        if not match:
            phoneme.append(p0)
    return phoneme


def pypinyin_g2p(text):
    """使用声调风格3, 在各个拼音之后用数字 [1-4] 表示声调。
    例如：你好世界 => ni3 hao3 shi4 jie4"""
    return " ".join(phonemizer(text))


def tokens2ids(cleaned_text):
    '''将一组用空格相隔的韵母声母字符串转为一系列整数'''
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text.split()]
    return sequence


# def ids2tokens(sequence):
#     '''将一系列整数转为对应的韵母或声母，一般用不到'''
#     result = ''
#     for symbol_id in sequence:
#         s = _id_to_symbol[symbol_id]
#         result += s
#     return result



