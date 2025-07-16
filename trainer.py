import argparse
from omegaconf import OmegaConf

import sys
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset import HTRDataset

from models import HTRNet
from utils.transforms import aug_transforms

import torch.nn.functional as F
import math

from utils.metrics import CER, WER


class HTRTrainer(nn.Module):
    def __init__(self, config):
        super(HTRTrainer, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()

    def prepare_dataloaders(self):
        config = self.config

        dataset_folder = config.data.path
        fixed_size = (config.arch.img_size[0], config.arch.img_size[1])

        train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=aug_transforms)
        classes = train_set.character_classes
        print('# training lines ' + str(train_set.__len__()))

        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        print('# validation lines ' + str(val_set.__len__()))

        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        print('# testing lines ' + str(test_set.__len__()))

        train_loader = DataLoader(train_set, batch_size=config.train.batch_size,
                                  shuffle=True, num_workers=config.train.num_workers)
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=config.eval.batch_size,
                                    shuffle=False, num_workers=config.eval.num_workers)
        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,
                                 shuffle=False, num_workers=config.eval.num_workers)

        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        classes += ' '
        classes = np.unique(classes)

        np.save(os.path.join(dataset_folder, 'classes.npy'), classes)

        cdict = {c: (i + 1) for i, c in enumerate(classes)}
        icdict = {(i + 1): c for i, c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }

    def prepare_net(self):
        config = self.config
        device = config.device

        print('Preparing Net - Architectural elements:')
        print(OmegaConf.to_yaml(config.arch))

        classes = self.classes['classes']
        net = HTRNet(config.arch, len(classes) + 1)

        if hasattr(config, 'resume') and config.resume is not None:
            print('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume, map_location=device)
            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)

        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='mean', zero_infinity=True)(F.log_softmax(y, dim=2),
                                                                                              t, ly, lt)

    def prepare_optimizers(self):
        config = self.config
        optimizer = torch.optim.AdamW(self.net.parameters(), config.train.lr, weight_decay=0.00005)
        self.optimizer = optimizer

        max_epochs = config.train.num_epochs

        # SỬA LỖI: Triển khai scheduler với warmup và cosine decay
        warmup_epochs = config.train.get('warmup_epochs', 1)  # Lấy từ config, mặc định là 1
        num_training_steps = max_epochs * len(self.loaders['train'])
        num_warmup_steps = warmup_epochs * len(self.loaders['train'])

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def decode(self, tdec, blank_id=0):
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([self.classes['i2c'][t] for t in tt if t != blank_id])
        return dec_transcr

    def sample_decoding(self):
        img, transcr = self.loaders['val'].dataset[np.random.randint(0, len(self.loaders['val'].dataset))]
        img = img.unsqueeze(0).to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            output_logits, _ = self.net(img)

        self.net.train()

        tdec = output_logits.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        dec_transcr = self.decode(tdec)

        print('orig:: ' + transcr.strip())
        print('pred:: ' + dec_transcr.strip())

    def train(self, epoch):
        config = self.config
        device = config.device
        self.net.train()

        t = tqdm.tqdm(self.loaders['train'])
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, (img, transcr) in enumerate(t):
            self.optimizer.zero_grad()
            img = img.to(device)

            output_for_ctc, act_lens = self.net(img)

            labels = torch.IntTensor([self.classes['c2i'][c] for c in ''.join(transcr)]).to(device)
            label_lens = torch.IntTensor([len(t) for t in transcr]).to(device)

            try:
                loss_val = self.ctc_loss(output_for_ctc, labels, act_lens, label_lens)
            except Exception as e:
                print("ERROR in CTC Loss calculation:", e)
                print(f"Logits shape: {output_for_ctc.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Act Lens shape: {act_lens.shape}, content: {act_lens}")
                print(f"Label Lens shape: {label_lens.shape}, content: {label_lens}")
                raise e

            tloss_val = loss_val.item()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            # SỬA LỖI: scheduler.step() được gọi sau mỗi batch
            self.scheduler.step()

            t.set_postfix(values='loss : {:.4f}'.format(tloss_val))

        self.sample_decoding()

    def test(self, epoch, tset='test'):
        config = self.config
        device = config.device
        self.net.eval()

        loader = self.loaders.get(tset)
        if loader is None:
            print(f"Set '{tset}' not recognized in test function")
            return

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))

        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        for (imgs, transcrs) in tqdm.tqdm(loader):
            imgs = imgs.to(device)
            with torch.no_grad():
                output_logits, _ = self.net(imgs)

            tdecs = output_logits.argmax(2).permute(1, 0).cpu().numpy()

            for i in range(tdecs.shape[0]):
                tdec = tdecs[i]
                transcr = transcrs[i].strip()
                dec_transcr = self.decode(tdec).strip()

                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)

        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        self.net.train()

    def save(self, epoch):
        model_dir = self.config.model_dir
        print(f'####################### Saving model at epoch {epoch} #######################')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        save_path = os.path.join(model_dir, f'htrnet_{epoch}.pt')
        torch.save(self.net.cpu().state_dict(), save_path)
        print(f"Model saved to {save_path}")
        self.net.to(self.config.device)


def parse_args():
    parser = argparse.ArgumentParser(description="HTR Trainer")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args, unknown = parser.parse_known_args()

    conf = OmegaConf.load(args.config)

    cli_conf = OmegaConf.from_cli(unknown)
    conf = OmegaConf.merge(conf, cli_conf)

    print("--- Loaded Configuration ---")
    print(OmegaConf.to_yaml(conf))
    print("--------------------------")

    return conf


if __name__ == '__main__':
    config = parse_args()
    max_epochs = config.train.num_epochs

    htr_trainer = HTRTrainer(config)

    print('Training Started!')
    htr_trainer.test(0, 'test')
    for epoch in range(1, max_epochs + 1):
        htr_trainer.train(epoch)
        # SỬA LỖI: Không gọi scheduler.step() ở đây nữa vì nó được gọi sau mỗi batch
        # htr_trainer.scheduler.step()

        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            htr_trainer.test(epoch, 'val')
            htr_trainer.test(epoch, 'test')

    final_save_path = os.path.join(config.model_dir, config.save)
    print(f"Saving final model to {final_save_path}")
    torch.save(htr_trainer.net.cpu().state_dict(), final_save_path)

