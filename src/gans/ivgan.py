import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm_notebook
from src.utils import *
import tensorflow as tf
from numpy.random import uniform
import matplotlib.pyplot as plt

class IVGan:
    def __init__(self, critic, generator):
        self.critic = critic
        self.generator = generator
       
    def _set_lr(self, lr):
        if lr is not None:
            self.optimizer = RMSprop(lr=lr)

    def _clip_critic_weights(self,clip):
        wgt = [tf.clip_by_value(w, -clip, clip) for w in self.critic.trainable_weights]
        self.critic.set_weights(wgt)

    def show_hist(self,x_train,x_true):
        iv = self.generator.predict_iv(x_train)
        plt.hist((x_train[:,-1] - x_train[:,0])/np.sqrt(iv),alpha=0.6,bins=50, density=True,label="Generated")
        plt.hist(x_true, bins=50, alpha=0.6, density=True,label="True")
        plt.legend()
        plt.show()

    def get_RMSE(self,vol_model):
        total_timesteps = 60*60*7.5
        n_timesteps = 60
        spots_test, vols_test = vol_model.generate(10000, total_timesteps , n_timesteps)
        iv = self.generator.predict_iv(spots_test)
        true_iv = np.sum(vols_test**2/total_timesteps, axis=1)
        return tf.math.sqrt(tf.keras.losses.MSE(iv,true_iv))


    def train_critic(self, true_batch, fake_batch, clip):
        fake_iv = tf.squeeze(self.generator(fake_batch))
        fake_iv = np.where(fake_iv ==0 , 0.000001, fake_iv)
        fake_sample = (fake_batch[:,-1] - fake_batch[:,0])/np.sqrt(fake_iv)
        fake_sample = np.reshape(fake_sample,(fake_sample.shape[0],1))
        true_sample = np.reshape(true_batch,(true_batch.shape[0],1))
        with tf.GradientTape() as tape:
            #we want to maximise so we need to put "-"
            critic_loss = tf.reduce_mean(self.critic(fake_sample) -self.critic(true_sample))
            epsilon = tf.random.uniform([1], 0,1)
            x_hat = epsilon * fake_sample + (1-epsilon)*true_sample
            with tf.GradientTape() as gp_tape:
                #GP stands for Gradient Penalty
                gp_tape.watch(x_hat)
                y_hat = self.critic(x_hat)
            gradient_norm = tf.abs(gp_tape.gradient(y_hat,x_hat))
            critic_loss += clip*tf.reduce_mean(tf.math.square((gradient_norm-1)))
        grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        #self._clip_critic_weights(clip)
        return critic_loss

    def train_generator(self, x_batch):
        with tf.GradientTape() as tape:
            iv = tf.cast(tf.squeeze(self.generator(x_batch)),tf.float32)
            iv = tf.where(iv ==0 , 0.000001, iv)
            gen_z = (x_batch[:,-1] - x_batch[:,0])/tf.math.sqrt(iv)
            gen_z = tf.reshape(gen_z,(gen_z.shape[0],1))
            generator_loss = -tf.reduce_mean(self.critic(gen_z))
        if tf.math.is_nan(generator_loss):
            print("NaN Loss")
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return generator_loss

    def train(self, x_train, true_sample, num_epochs, batch_size, clip, num_critic=5, lr=None, true_int_var=None, 
                show_loss = True, show_hist=False, show_rmse = False, vol_model = None):
        self._set_lr(lr)
        sample_size = x_train.shape[0] 
        n_steps = sample_size // batch_size
        x_fake = create_train_dataset(x_train, batch_size,drop_remainder = True).repeat().__iter__()
        x_true = create_train_dataset(true_sample, batch_size,drop_remainder = True).repeat().__iter__()
        critic_losses = []
        generator_losses = []
        test_rmse = []
        for epoch in range(num_epochs):
            with tqdm_notebook(total=n_steps, desc=f'Epoch {epoch+1} of {num_epochs}') as progress:
                for steps in range(n_steps):
                    progress.update()
                    for critic_step in range(num_critic):
                        loss_val = self.train_critic(next(x_true),next(x_fake),clip)
                    critic_losses.append(loss_val.numpy())
                    loss_val = self.train_generator(next(x_fake))
                    generator_losses.append(loss_val.numpy())
            if show_hist:
                self.show_hist(x_train,true_sample)
            if show_rmse and vol_model:
                test_rmse.append(self.get_RMSE(vol_model))
                plt.plot(test_rmse)
                plt.grid()
                plt.xlabel("Epochs")
                plt.ylabel("RMSE")
                plt.yscale("log")
                plt.show()
        self.history = {'critic_loss': critic_losses, 'generator_loss': generator_losses,
                        'RMSE':test_rmse}
        return self.history

    def predict_iv(self, x):
        iv = self.generator(x).numpy()
        return iv.squeeze()

    def predict_z(self, x):
        iv = self.predict_iv(x)
        z = (x[:,-1] - x[:,0])/np.sqrt(iv)
        return z