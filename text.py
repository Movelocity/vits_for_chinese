from pypinyin import Style, pinyin

# 定义标点符号集，若修改的话需要重新训练模型
_signs = ["，", "。", "？", "！",'2','1']

# 韵母
_initials = [
    'b','p','m','f',
    'd','t','n','l',
    'g','k','h','j','q','x',
    'w','y',
    "zh","ch","sh","z","c","s","r"
] # 单字母的要往后放，让多字母先匹配

# 声调
_tones = ["", "1", "2", "3", "4"]

# 声母
_finals = [
    'a','o','e','i','u','v',
    'ai','ei','ui','ao','ou','iu','ie','ve','er',
    'an','en','in','un','vn','on',
    'ang','eng','ing','ong',
    'ia','iao','ian','iang','iong',
    'uo','uai','uei','ua','ue','uan','uen','uang','ueng',
    'van','ag','og','eg','ig','iag'
]

symbols = _signs + _initials + [i + j for i in _finals for j in _tones]

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# TODO: 完善这些cleaner的功能，提高g2p流程的抗干扰性。
# 现阶段还是以test检测加手动定位纠正为主

# TODO: 英文字母转汉字
def letters_cleaner(phonemes):
    return phonemes

_anno_map = {
    "、": "，",
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？"
}
def multianno_cleaner(phonemes):
    for i, p in enumerate(phonemes):
        if p[0] in _anno_map.keys():
            phonemes[i] = _anno_map[p[0]] # 替换英文标点
        for s in _anno_map.values():
            if s in phonemes[i] and len(phonemes[i])>1:
                # 多个符号相邻，取第一个
                phonemes[i] = s
    return phonemes

# TODO: 阿拉伯数字转汉字。现阶段需手动修改数据集
def number_cleaner(phonemes):
    return phonemes

def pypinyin_g2p_phone(text):
    """使用声调风格3, 在各个拼音之后用数字 [1-4] 表示声调。
    例如：你好世界 => ni3 hao3 shi4 jie4"""
    phonemes = pinyin(text, style=Style.TONE3, errors='default')  # TONE3 = 8
    result = []
    for phone in phonemes:
        matched = False
        for yunmu in _initials:
            if phone[0].startswith(yunmu):
                result.append(yunmu)
                result.append(phone[0].replace(yunmu, ""))
                matched = True
                break
        if not matched:
            result.append(phone[0])  # 单一个声母的情况

    for cleaner in [letters_cleaner, multianno_cleaner, number_cleaner]:
        result = cleaner(result)
    return " ".join(result)


def tokens2ids(cleaned_text):
  '''将一组用空格相隔的韵母声母字符串转为一系列整数'''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text.split()]
  return sequence


def ids2tokens(sequence):
  '''将一系列整数转为对应的韵母或声母，一般用不到'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


