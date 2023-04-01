
import os
import json
import argparse
import itertools
import math

import utils
import commons
import numpy as np
import monotonic_align

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)
from models import (
    VITS_Model,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import text

torch.backends.cudnn.benchmark = True

def negative_cross_entropy(m_p, logs_p, z_p):
    # negative cross-entropy between speech prior (s_p) and word prior (z_p)
    # -sum(x * log(y) + (1 - x) * log(1 - y))
    # [b, d, t] meaning sp times squre root of e
    s_p_sq_r = torch.exp(-2 * logs_p)  # (e^ln(s_p))^(-2) = 1 / s_p^2
    neg_cent1 = torch.sum(-0.5*math.log(2*math.pi) -logs_p, [1], keepdim=True)  # [b, 1, t_s]
    # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
    neg_cent2 = torch.matmul(-0.5*(z_p**2).transpose(1, 2), s_p_sq_r)
    # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
    neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
    neg_cent4 = torch.sum(-0.5*(m_p**2) * s_p_sq_r,[1], keepdim=True)  # [b, 1, t_s]
    neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
    return neg_cent

@torch.no_grad()
def calc_attn(x_mask, y_mask, m_p, logs_p, z_p):
    neg_cent = negative_cross_entropy(m_p, logs_p, z_p)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
    return attn

class Trainer:
    def __init__(self, hps, use_tensorboard=True):
        # 加载DataLoader
        # 自动寻找模型检查点，没有就创建新模型
        self.train_config = hps.train
        self.data_config = hps.data
        self.model_config = hps.model

        self.log_interval = self.train_config.log_interval
        self.eval_interval = self.train_config.eval_interval
        self.segment_size = self.train_config.segment_size // self.data_config.hop_length
        self.fp16_run = self.train_config.fp16_run
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.logger = utils.get_logger("logs/")
            # logger.info(hps)
            # utils.check_git_hash(hps.model_dir)
            self.writer = SummaryWriter(log_dir="logs/train")
            self.writer_eval = SummaryWriter(log_dir="logs/eval")
        else:
            print("Run without tensorboard")

        self.train_dataset = TextAudioSpeakerLoader(hparams=self.data_config)
        train_sampler = DistributedBucketSampler(
            self.train_dataset,
            self.train_config.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=1,
            rank=0,
            shuffle=True
        )

        collate_fn = TextAudioSpeakerCollate()
        self.train_loader = DataLoader(self.train_dataset, num_workers=2, shuffle=False, pin_memory=True,
                                collate_fn=collate_fn, batch_sampler=train_sampler)

        self.model = VITS_Model(
            n_vocab=len(text.symbols),
            spec_channels=self.data_config.filter_length//2 + 1,
            **hps.model).cuda()

        self.net_d = MultiPeriodDiscriminator(self.model_config.use_spectral_norm).cuda()

        self.optim_g = torch.optim.AdamW(
            self.model.parameters(),
            self.train_config.learning_rate,
            betas=self.train_config.betas,
            eps= self.train_config.eps)
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.train_config.learning_rate,
            betas=self.train_config.betas,
            eps=self.train_config.eps)

        self.lr, self.epoch_start = utils.load_checkpoint(self.model, self.optim_g, self.net_d, self.optim_d, init_lr=self.train_config.learning_rate)

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.train_config.lr_decay, last_epoch=self.epoch_start-2)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.train_config.lr_decay, last_epoch=self.epoch_start-2)

        self.scaler = GradScaler(enabled=self.fp16_run)
        self.scalar_dict = None

    def train(self, epochs=1000):
        for epoch in range(self.epoch_start, epochs+1):
            print('====> Epoch: {}'.format(epoch))
            self.train_loader.batch_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            
            if epoch % self.log_interval == 0:
                self.log(epoch)
            if self.epoch % self.eval_interval == 1:
                self.evaluate(size=5, epoch=epoch)
                utils.save_checkpoint(self.model, self.optim_g, self.net_d, self.optim_d, 
                    self.train_config.learning_rate, epoch, "logs/model")

    def train_epoch(self, epoch):
        
        for (x, x_lengths, spec, spec_lengths, y, y_lengths, speaker_embs) in self.train_loader:
            x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
            spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
            y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)
            speaker_embs = speaker_embs.cuda(non_blocking=True)

            self.model.train()
            self.net_d.train()

            # Train the Discriminator
            with autocast(enabled=self.fp16_run):
                 # x: 文本编码；y: 语音频谱
                x, m_p, logs_p, x_mask = self.model.enc_p(x, x_lengths)
                z, m_q, logs_q, y_mask = self.model.enc_q(spec, spec_lengths, embed=speaker_embs)
                z_p = self.model.flow(z, y_mask, embed=speaker_embs)  # 具体形状有待调试

                attn = calc_attn(x_mask, y_mask, m_p, logs_p, z_p)
                w = attn.sum(2)

                # calculate duration loss
                l_length = self.model.dp(x, x_mask, w, embed=speaker_embs, training=True) / torch.sum(x_mask)
                # sum(x_mask): 每行mask都代表样本在batch中的有效长度

                # 按照预测的时长扩展文本编码
                m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
                logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

                # 批次里每条语音只取一个片段用于计算loss，这里指定片段的偏移量，便于从ground truth里截取同样的片段
                z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
                y_hat = self.model.dec(z_slice, embed=speaker_embs)

                # 这里曾经是原仓库的分界线
                y = commons.slice_segments(y, ids_slice * self.data_config.hop_length, self.train_config.segment_size)
                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())  # 训练D的阶段，detech可防止梯度传到G
                with autocast(enabled=False):  # 不知道要不要去掉这个，能不能直接取消缩进退出cast的代码块
                    loss_disc_all, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

            self.optim_d.zero_grad()
            self.scaler.scale(loss_disc_all).backward()
            self.scaler.unscale_(self.optim_d)
            grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), clip_value=None) # 裁剪值None，仅计算和记录, 不产生其它效应
            self.scaler.step(self.optim_d)

            # Train the Generator
            with autocast(enabled=self.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)

                mel = spec_to_mel_torch(spec, config=self.data_config)  # 由频谱获得梅尔频谱，用于计算损失
                y_mel = commons.slice_segments(mel, ids_slice, self.segment_size)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), config=self.data_config)
                with autocast(enabled=False):  # 可不可以取消这个？
                    loss_dur = torch.sum(l_length.float())  # panelty on the total time of the result
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.train_config.c_mel  # 梅尔频谱图之间计算损失
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * self.train_config.c_kl  # 文本编码对齐语音VAE的中间值

                    loss_fm = feature_loss(fmap_r, fmap_g)  # 真假数据在判别器模块内的feature map应尽量靠近
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)  # 每个值尽量靠近1
                    loss_gen_all = loss_gen + loss_fm + loss_kl + loss_mel + loss_dur

            self.optim_g.zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            grad_norm_g = commons.clip_grad_value_(self.model.parameters(), clip_value=None)  # 裁剪值None，仅计算和记录, 不产生其它效应
            self.scaler.step(self.optim_g)
            self.scaler.update()

        self.scheduler_g.step()
        self.scheduler_d.step()

        self.scalar_dict = {
            "info/grad_norm_d": grad_norm_d,
            "info/grad_norm_g": grad_norm_g,
            "info/learning_rate": self.optim_g.param_groups[0]['lr'], 
            "loss/loss_gen_all": loss_gen_all, 
            "loss/loss_disc_all": loss_disc_all,  # 记录判别器损失，可以知道训练有没有崩掉
            "loss/g/dur": loss_dur,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            "loss/g/fm": loss_fm, 
        }
        self.scalar_dict.update({"loss/gen/{}".format(i): v for i, v in enumerate(losses_gen)})
        self.scalar_dict.update({"loss/disc_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        self.scalar_dict.update({"loss/disc_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

        self.image_dict = {
            "sliced/mel_orgin": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "sliced/mel_generated": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "complete/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "complete/alignment": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
        }

    def log(self, epoch):
        if epoch % self.log_interval != 0: return
        if not self.use_tensorboard: return
        # 将训练记录写入tensorboard
        # 若服务器无法开启查看端口，训练完后可将记录下载到本地查看
        # 写入tensorboard日志
        utils.summarize(
            writer=self.writer, 
            global_step=self.epoch, 
            images=self.image_dict, 
            scalars=self.scalar_dict
        )
        
    @torch.no_grad()
    def evaluate(self, size=4, epoch=1):
        self.model.eval()
        eval_data = [self.train_dataset[idx] for idx in np.random.randint(0, len(self.train_dataset), size=size)]
        # (token_ids, spec, audio, embed)
        audio_dict = {}
        image_dict = {}
        for i, (token_ids, spec, audio, embed) in enumerate(eval_data):
            phonemes = data[-1]
            input_ids = token_ids.unsqueeze(0).cuda()
            input_lengths = torch.LongTensor([input_ids.size(1)]).cuda()
            embed = embed.unsqueeze(0).cuda()
            y_hat = self.model.infer(input_ids, input_lengths, embed=embed)[0]

            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), config=self.data_config)
            audio_dict.update({str(i): y_hat[0, :, :]})
            image_dict.update({f"gen/mel/{i}": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())})

        utils.summarize(
            writer=self.writer_eval,
            global_step=epoch,
            images=image_dict,
            audios=audio_dict,
            audio_sampling_rate=self.data_config.sampling_rate
        )
        self.model.train()

if __name__ == "__main__":
    # Assume Single Node Multi GPUs Training Only
    assert torch.cuda.is_available(), "训练模型要用显卡哦."

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorboard', type=bool, default=True, help='whether to use tensorboard or not')
    args = parser.parse_args()

    hps = utils.get_hparams()  # 已创建logs文件夹
    print("-------- running ---------")
    trainer = Trainer(hps, use_tensorboard=args.tensorboard)

    trainer.train()

# def train(hps):
#     logger = utils.get_logger(hps.model_dir)
#     logger.info(hps)
#     # utils.check_git_hash(hps.model_dir)
#     writer = SummaryWriter(log_dir=os.path.join(hps.model_dir, "train"))
#     writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

#     train_dataset = TextAudioSpeakerLoader(hparams=hps.data)
#     train_sampler = DistributedBucketSampler(
#         train_dataset,
#         hps.train.batch_size,
#         [32, 300, 400, 500, 600, 700, 800, 900, 1000],
#         num_replicas=1,
#         rank=0,
#         shuffle=True)

#     collate_fn = TextAudioSpeakerCollate()
#     train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
#                               collate_fn=collate_fn, batch_sampler=train_sampler)

#     net_g = SynthesizerTrn(
#         n_vocab=len(text.symbols),
#         spec_channels=hps.data.filter_length // 2 + 1,
#         segment_size=hps.train.segment_size // hps.data.hop_length,  # 这个参数应该只有训练时要用，没必要存储在模型内部
#         **hps.model).cuda()

#     net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
#     optim_g = torch.optim.AdamW(
#         net_g.parameters(),
#         hps.train.learning_rate,
#         betas=hps.train.betas,
#         eps=hps.train.eps)
#     optim_d = torch.optim.AdamW(
#         net_d.parameters(),
#         hps.train.learning_rate,
#         betas=hps.train.betas,
#         eps=hps.train.eps)

#     lr, epoch_start = utils.load_checkpoint(net_g, optim_g, net_d, optim_d, hps)

#     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
#         optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_start-2)
#     scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
#         optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_start-2)

#     scaler = GradScaler(enabled=hps.train.fp16_run)

#     for epoch in range(epoch_start, hps.train.epochs + 1):
#         train_and_evaluate(0, epoch, hps, [net_g, net_d], 
#                 [optim_g, optim_d], [scheduler_g, scheduler_d], 
#                 scaler, train_loader, logger, [writer, writer_eval])
#         scheduler_g.step()
#         scheduler_d.step()

# def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
#     net_g, net_d = nets
#     optim_g, optim_d = optims
#     scheduler_g, scheduler_d = schedulers
#     train_loader = loaders
#     if writers is not None: writer, writer_eval = writers

#     train_loader.batch_sampler.set_epoch(epoch)
    
#     net_g.train()
#     net_d.train()
#     for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speaker_embs) in enumerate(train_loader):
#         x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
#         spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
#         y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
#         speaker_embs = speaker_embs.cuda(rank, non_blocking=True)

#         with autocast(enabled=hps.train.fp16_run):
#             y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
#                 (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speaker_embs)

#             mel = spec_to_mel_torch(spec, config=hps.data)  # 由频谱获得梅尔频谱，用于计算损失
#             y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size//hps.data.hop_length)
#             y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), config=hps.data)

#             y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
#             # Discriminator
#             y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
#             with autocast(enabled=False):
#                 loss_disc_all, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

#         optim_d.zero_grad()
#         scaler.scale(loss_disc_all).backward()
#         scaler.unscale_(optim_d)
#         grad_norm_d = commons.clip_grad_value_(net_d.parameters(), clip_value=None) # 裁剪值None，仅计算和记录, 不产生其它效应
#         scaler.step(optim_d)

#         with autocast(enabled=hps.train.fp16_run):
#             # Generator
#             y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
#             with autocast(enabled=False):
#                 loss_dur = torch.sum(l_length.float())  # panelty on the total time of the result
#                 loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel  # 频谱图之间计算损失
#                 loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl  # 文本编码对齐语音VAE的中间值

#                 loss_fm = feature_loss(fmap_r, fmap_g)  # 真假数据在判别器模块内的feature map应尽量靠近
#                 loss_gen, losses_gen = generator_loss(y_d_hat_g)  # 每个值尽量靠近1
#                 loss_gen_all = loss_gen + loss_fm + loss_kl + loss_mel + loss_dur
#         optim_g.zero_grad()
#         scaler.scale(loss_gen_all).backward()
#         scaler.unscale_(optim_g)
#         grad_norm_g = commons.clip_grad_value_(net_g.parameters(), clip_value=None)  # 裁剪值None，仅计算和记录, 不产生其它效应
#         scaler.step(optim_g)
#         scaler.update()

#     if epoch % hps.train.log_interval == 0:
#         # 将训练记录写入tensorboard，
#         # 若服务器无法开启查看端口，训练完后可将记录下载到本地查看
#         lr = optim_g.param_groups[0]['lr']
#         scalar_dict = {
#             "info/grad_norm_d": grad_norm_d, 
#             "info/grad_norm_g": grad_norm_g,
#             "info/learning_rate": lr, 
#             "loss/loss_gen_all": loss_gen_all, 
#             "loss/loss_disc_all": loss_disc_all,  # 记录判别器损失，可以知道训练有没有崩掉
#             "loss/g/dur": loss_dur,
#             "loss/g/mel": loss_mel,
#             "loss/g/kl": loss_kl,
#             "loss/g/fm": loss_fm, 
#         }
#         scalar_dict.update(
#             {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
#         scalar_dict.update(
#             {"loss/disc_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
#         scalar_dict.update(
#             {"loss/disc_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

#         image_dict = {
#             "sliced/mel_orgin": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
#             "sliced/mel_generated": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
#             "complete/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
#             "complete/alignment": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
#         }

#         # 写入tensorboard日志
#         utils.summarize(writer=writer, global_step=epoch, images=image_dict, scalars=scalar_dict)
#         print('====> Epoch: {}'.format(epoch))

#     if epoch % hps.train.eval_interval == 1:
#         evaluate(hps, net_g, writer_eval, epoch)
#         utils.save_checkpoint(net_g, optim_g, net_d, optim_d, hps.train.learning_rate, epoch, hps.model_dir)

# @torch.no_grad()
# def evaluate(hps, generator, writer_eval, epoch):
#     generator.eval()
#     eval_data = load_filepaths_and_text(hps.data.validation_files)[:4]

#     audio_dict = {}
#     image_dict = {}
#     for i, data in enumerate(eval_data):
#         phonemes = data[-1]
#         input_ids = torch.LongTensor(text.tokens2ids(phonemes)).unsqueeze(0).cuda()
#         input_lengths = torch.LongTensor([input_ids.size(1)]).cuda()
#         sid = torch.LongTensor([int(data[1])]).cuda()
#         y_hat = generator.infer(input_ids, input_lengths, sid=sid)[0]

#         y_hat_mel = mel_spectrogram_torch(
#             y_hat.squeeze(1).float(),
#             hps.data.filter_length,
#             hps.data.n_mel_channels,
#             hps.data.sampling_rate,
#             hps.data.hop_length,
#             hps.data.win_length,
#             hps.data.mel_fmin,
#             hps.data.mel_fmax
#         )
#         audio_dict.update({str(i): y_hat[0, :, :]})
#         image_dict.update({f"gen/mel/{i}": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())})

#     utils.summarize(
#         writer=writer_eval,
#         global_step=epoch,
#         images=image_dict,
#         audios=audio_dict,
#         audio_sampling_rate=hps.data.sampling_rate
#     )
#     generator.train()



