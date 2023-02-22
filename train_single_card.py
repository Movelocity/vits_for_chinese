
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
    load_filepaths_and_text
)
from models import (
    SynthesizerTrn,
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

def train(hps):
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    # utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=1,
        rank=0,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    
    net_g = SynthesizerTrn(
        len(text.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    except:
        epoch_str = 1

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(0, epoch, hps, [net_g, net_d], [optim_g, optim_d], [
                scheduler_g, scheduler_d], scaler, train_loader, logger, [writer, writer_eval])
        scheduler_g.step()
        scheduler_d.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    
    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    x, x_lengths, spec, spec_lengths, speakers)

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                  z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

    if epoch % hps.train.log_interval == 0:  # 记录模型损失值
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]

        scalar_dict = {
            "loss/loss_gen_all": loss_gen_all, 
            "loss/loss_disc_all": loss_disc_all,
            "learning_rate": lr, 
            "grad_norm_d": grad_norm_d, 
            "grad_norm_g": grad_norm_g,
            "loss/g/fm": loss_fm, 
            "loss/g/mel": loss_mel,
            "loss/g/dur": loss_dur, 
            "loss/g/kl": loss_kl
        }

        scalar_dict.update(
            {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update(
            {"loss/disc_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update(
            {"loss/disc_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

        image_dict = {
            "sliced/mel_orgin": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "sliced/mel_generated": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "complete/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "complete/alignment": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
        }

        utils.summarize(
            writer=writer,
            global_step=epoch,
            images=image_dict,
            scalars=scalar_dict)

    if epoch % hps.train.eval_interval == 0:
        evaluate(hps, net_g, writer_eval, epoch)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(
            hps.model_dir, "G_{}.pth".format(epoch)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(
            hps.model_dir, "D_{}.pth".format(epoch)))

    print('====> Epoch: {}'.format(epoch))

@torch.no_grad()
def evaluate(hps, generator, writer_eval, epoch):
    generator.eval()
    eval_data = load_filepaths_and_text(hps.data.validation_files)[:4]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    audio_dict = {}
    for data in eval_data:
        phonemes = text.pypinyin_g2p_phone(data[-1])
        input_ids = torch.LongTensor(text.tokens2ids(phonemes)).unsqueeze(0).to(device)
        input_lengths = torch.LongTensor([input_ids.size(1)]).to(device)
        sid = torch.LongTensor([int(data[1])]).to(device)
        audio = net_g.infer(input_ids, input_lengths, sid=sid)[0][0,0].data.cpu().float().numpy()
        audio_dict.update({data[-1]: audio})

    utils.summarize(
        writer=writer_eval,
        global_step=epoch,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    # Assume Single Node Multi GPUs Training Only
    assert torch.cuda.is_available(), "CPU training is not allowed."

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                      help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, default="model",
                      help='Model name')
    args = parser.parse_args()
    hps = utils.get_hparams(args)  # 已创建logs文件夹

    print("-------- running ---------")
    train(hps)
