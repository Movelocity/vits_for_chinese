import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
import text

from speechbrain.pretrained import EncoderClassifier  # 增加依赖链有风险

import torchaudio
import torchaudio.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_filenames_and_text(textfile, split="|"):
    with open(textfile, encoding='utf-8') as f:
        fnames_and_text = []
        for line in f:
            if split not in line: continue
            path_text = line.strip().split(split)
            fnames_and_text.append(path_text)
    return fnames_and_text

def prepare_data(hparams, skip=False):
    """如果要重新识别语音，可以手动删除原有记录"""
    import whisper
    import glob
    print('正在加载 whisper 语音识别模型...')
    model = whisper.load_model("base").to(device)
    print('加载成功')

    print("正在加载 ecapa 声纹编码模型...")
    speaker_classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    print('加载成功')

    audio_files = glob.glob("dataset/wave_data/*.wav")
    names = [f.split('/')[-1] for f in audio_files]

    ok_list = []
    with open('dataset/phonemes.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        if len(lines) > 2:
            for line in lines:
                ok_list.append(line.strip().split("|")[0])
    
    text_file = open('dataset/text.txt', 'a', encoding='utf-8')
    phoneme_file = open('dataset/phonemes.txt', 'a', encoding='utf-8')
    for name in names:
        if name in ok_list: continue
        # 要重采样到16000
        result_dict = model.transcribe('dataset/wave_data/'+audio_file)  # 内置语音转文字就不用另外写cleaner了
        result = '#'+('#'.join([t['text'] for t in result_dict['segments']]))+'#'

        phonemes = text.pypinyin_g2p(result)
        phoneme_data.append(phonemes)  # 这个地方应该很容易报错，要多加关注

        text_file.write(name + '|' + result + '\n')  
        phoneme_file.write(name + '|' + phonemes + '\n')  # 保存临时结果

    text_file.close()
    phoneme_file.close()
    print('数据集语音识别结果已保存在 dataset/text.txt')

    print('尝试生成声纹编码和频谱图')
    audio, sr = torchaudio.load(name)
    resampler = T.Resample(sr, new_freq=16000, dtype=audio.dtype)  # 数据最好是同样采样率的，不然会出错
    for name in names:
        emebed_file = 'dataset/embed/'+name.replace('.wav', '.emb.pt')
        spec_file = 'dataset/spec/'+name.replace('.wav', '.spec.pt')
        # 可以指定 skip=True 来跳过已经处理过的文件
        if skip and os.path.exists(emebed_file) and os.path.exists(spec_file): continue

        audio, sr = torchaudio.load(name)
        audio_16k = resampler(audio) if sr != 16000 else audio
        embedding = speaker_classifier.encode_batch(audio_16k)[0, 0]
        embedding = embedding / torch.norm(embedding)
        torch.save(embedding, emebed_file)

        spectrogram = spectrogram_torch(audio, n_fft=hparams.filter_length, sampling_rate=hparams.sampling_rate,
                        hop_size=hparams.hop_length, win_size=hparams.win_length, center=False)[0]
        torch.save(spectrogram, spec_file)


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, hparams):
        self.audiopaths_sid_text = load_filenames_and_text('dataset/phonemes.txt')
        # self.max_wav_value = 32768.0
        self.sr = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length    = hparams.hop_length
        self.win_length    = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.add_blank = False
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        # random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, embed, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        audio, spec, embed = self.get_audio_spec_emb(audiopath)
        return (text, spec, audio, embed)

    def get_audio_spec_emb(self, name):
        audio, sr = torchaudio.load('dataset/wave_data/'+name)
        if sr != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sr, self.sr))
        emebed_file = 'dataset/embed/'+name.replace('.wav', '.emb.pt')
        spec_file = 'dataset/spec/'+name.replace('.wav', '.spec.pt')
        spec =  torch.load(spec_file)
        embed =  torch.load(emebed_file)
        return audio, spec, embed

    def get_text(self, text):
        text_norm = text.tokens2ids(text)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Help improve training efficiency and reduce memory usage by reducing the amount of padding required.
    使得一个batch里的数据差不多长, 减少padding操作, 节约算力和内存

    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
