import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class TrainCVAE:
    def __init__(self,
                 model,
                 train_gen,
                 val_gen,
                 optimizer,
                 loss,
                 epoch,
                 batch_size,
                 steps_per_epoch,
                 steps_per_val,
                 writer,
                 model_name,
                 info_interval):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.opt = optimizer
        self.loss = loss
        self.epoch = epoch
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_val = steps_per_val
        self.writer = writer
        self.model_name = model_name
        self.info_interval = info_interval

        self.scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        total = 0

        for i in tqdm(range(self.steps_per_epoch)):
            x, y_true = self.train_gen.data_generation(i)

            self.opt.zero_grad()

            recon_batch, mu, logvar = self.model(x, y_true)
            loss = self.loss.cvae_loss_function(recon_batch, x, mu, logvar)
            loss.backward()

            self.opt.step()

            running_loss += loss.item()

            total += x.size(0)

            if i % self.info_interval == (self.info_interval - 1):
                last_loss = running_loss / total

                print('\n')
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                print('\n')

                tb_x = epoch_index * self.steps_per_epoch + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def training(self):
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(self.epoch):
            print('\n\n ####################### EPOCH {} ####################### \n'.format(epoch_number + 1))

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, self.writer)

            running_vloss = 0.0
            total = 0
            self.model.eval()

            with torch.no_grad():
                for i in range(self.steps_per_val):
                    xv, yv_true = self.val_gen.data_generation(i)

                    recon_batch, mu, logvar = self.model(xv, yv_true)
                    vloss = self.loss.cvae_loss_function(recon_batch, xv, mu, logvar)

                    running_vloss += vloss

                    total += xv.size(0)

            avg_vloss = running_vloss / total

            print('LOSS train {}, validation {}'.format(avg_loss, avg_vloss))

            self.writer.add_scalars('Training vs. Validation loss',
                                    {'Training_loss': avg_loss, 'Validation_loss': avg_vloss},
                                    epoch_number + 1)
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), self.model_name)

            if self.scheduler:
                self.scheduler.step(avg_vloss)

            epoch_number += 1
