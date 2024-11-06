import os, json, argparse, warnings, logging, sys
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
sys.path.append('./src')
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
global_step = 0

def main(model: str, config: dict):
    MODEL_NAME = model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    train_dataset = TextAudioLoader(config['data']['training_files'], config['data'])
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True, pin_memory=True,
        batch_size=config['train']['batch_size'], collate_fn=collate_fn)
    logger.info(f'TrainLoader Length: {len(train_loader)}')
    net_g = SynthesizerTrn(
        len(symbols),
        config['data']['filter_length'] // 2 + 1,
        config['train']['segment_size'] // config['data']['hop_length'],
        **config['model']).to(DEVICE)
    net_d = MultiPeriodDiscriminator(config['model']['use_spectral_norm']).to(DEVICE)
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        config['train']['learning_rate'], 
        betas=config['train']['betas'], 
        eps=config['train']['eps'])
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config['train']['learning_rate'], 
        betas=config['train']['betas'], 
        eps=config['train']['eps'])
    try:
        g_checkpoint_path = f'./outputs/{MODEL_NAME}/ckpt/G_.pth'
        g_checkpoint = torch.load(g_checkpoint_path, map_location=DEVICE)
        net_g.load_state_dict(g_checkpoint['model'])
        optim_g.load_state_dict(g_checkpoint['optimizer'])
        d_checkpoint_path = f'./outputs/{MODEL_NAME}/ckpt/D_.pth'
        d_checkpoint = torch.load(d_checkpoint_path, map_location=DEVICE)
        net_d.load_state_dict(d_checkpoint['model'])
        optim_d.load_state_dict(d_checkpoint['optimizer'])
        epoch_str = g_checkpoint['iteration']
        global global_step
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config['train']['lr_decay'], last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config['train']['lr_decay'], last_epoch=epoch_str-2)
    scaler = GradScaler(enabled=config['train']['fp16_run'])
    logger.info('Start Training')
    for epoch in range(epoch_str, config['train']['epochs'] + 1):
        train(MODEL_NAME, epoch, config, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, train_loader, DEVICE)
        scheduler_g.step()
        scheduler_d.step()

def train(model_name, epoch, config, nets, optimizers, schedulers, scaler, train_loader, device):
    MODEL_NAME = model_name
    net_g, net_d = nets
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = schedulers
    global global_step
    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        with autocast(enabled=config['train']['fp16_run']):
            y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)
            mel = spec_to_mel_torch(
                spec, 
                config['data']['filter_length'], 
                config['data']['n_mel_channels'], 
                config['data']['sampling_rate'],
                config['data']['mel_fmin'], 
                config['data']['mel_fmax'])
            y_mel = commons.slice_segments(mel, ids_slice, config['train']['segment_size'] // config['data']['hop_length'])
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), 
                config['data']['filter_length'], 
                config['data']['n_mel_channels'], 
                config['data']['sampling_rate'], 
                config['data']['hop_length'], 
                config['data']['win_length'], 
                config['data']['mel_fmin'], 
                config['data']['mel_fmax']
            )
            y = commons.slice_segments(y, ids_slice * config['data']['hop_length'], config['train']['segment_size'])
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        with autocast(enabled=config['train']['fp16_run']):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * config['train']['c_mel']
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config['train']['c_kl']
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        if global_step % config['train']['log_interval'] == 0:
            lr = optim_g.param_groups[0]['lr']
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
            logger.info('Train Epoch: {} [{:.0f}%]'.format(
                epoch, 100. * batch_idx / len(train_loader)))
            logger.debug([x.item() for x in losses] + [global_step, lr])
            scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
            scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})
            scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
            scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})
        if global_step % 1000 == 0:
            if hasattr(net_g, 'module'): state_dict = net_g.module.state_dict()
            else: state_dict = net_g.state_dict()
            torch.save({'model': state_dict,
                'iteration': epoch,
                'optimizer': optim_g.state_dict(),
                'learning_rate': config['train']['learning_rate']}, f'./outputs/{MODEL_NAME}/ckpt/G_{global_step}.pth')
            if hasattr(net_d, 'module'): state_dict = net_d.module.state_dict()
            else: state_dict = net_d.state_dict()
            torch.save({'model': state_dict,
                'iteration': epoch,
                'optimizer': optim_d.state_dict(),
                'learning_rate': config['train']['learning_rate']}, f'./outputs/{MODEL_NAME}/ckpt/D_{global_step}.pth')
            logger.info(f'Save Checkpoint, Path: G_{global_step}.pth, D_{global_step}.pth')
        global_step += 1
    logger.info(f'====> Epoch: {epoch}')

def get_logger(modelname: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./outputs/{modelname}/train.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()
    logger = get_logger(args.model)
    os.mkdir(f'./outputs/{args.model}/ckpt')
    with open(f'./outputs/{args.model}/config.json', 'r') as f:
        config = json.load(f)
    main(args.model, config)