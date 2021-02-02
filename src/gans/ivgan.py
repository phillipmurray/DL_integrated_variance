import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm_notebook
from src.utils import *
import tensorflow as tf
from numpy.random import uniform

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

    def train_critic(self, true_batch, fake_batch, clip):
        fake_iv = tf.squeeze(self.generator(fake_batch))
        fake_iv = np.abs(np.where(fake_iv ==0 , 0.000001, fake_iv))
        fake_sample = (fake_batch[:,-1] - fake_batch[:,0])/np.sqrt(fake_iv)
        fake_sample = np.reshape(fake_sample,(fake_sample.shape[0],1))
        true_sample = np.reshape(true_batch,(true_batch.shape[0],1))
        with tf.GradientTape() as tape:
            #we want to maximise so we need to put "-"
            mean_fake = tf.reduce_mean(self.critic(fake_sample))
            mean_true = tf.reduce_mean(self.critic(true_sample))
            critic_loss = tf.reduce_mean(self.critic(fake_sample) -self.critic(true_sample))
        grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        self._clip_critic_weights(clip)
        return critic_loss

    def train_generator(self, x_batch):
        with tf.GradientTape() as tape:
            iv = tf.cast(tf.squeeze(self.generator(x_batch)),tf.float32)
            iv = tf.math.abs(tf.where(iv ==0 , 0.000001, iv))
            gen_z = (x_batch[:,-1] - x_batch[:,0])/tf.math.sqrt(iv)
            gen_z = tf.reshape(gen_z,(gen_z.shape[0],1))
            generator_loss = -tf.reduce_mean(self.critic(gen_z))
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return generator_loss

    def train(self, x_train, true_sample, num_epochs, batch_size, clip, num_critic=5, lr=None, true_int_var=None, show_loss = True, show_hist=False):
        self._set_lr(lr)
        sample_size = x_train.shape[0] 
        n_steps = sample_size // batch_size
        x_fake = create_train_dataset(x_train, batch_size,drop_remainder = True).repeat().__iter__()
        x_true = create_train_dataset(true_sample, batch_size,drop_remainder = True).repeat().__iter__()
        critic_losses = []
        generator_losses = []
        for epoch in range(num_epochs):
            with tqdm_notebook(total=n_steps, desc=f'Epoch {epoch+1} of {num_epochs}') as progress:
                for steps in range(n_steps):
                    progress.update()
                    for critic_step in range(num_critic):
                        loss_val = self.train_critic(next(x_true),next(x_fake),clip)
                    critic_losses.append(loss_val.numpy())
                    loss_val = self.train_generator(next(x_fake))
                    generator_losses.append(loss_val.numpy())

        self.history = {'critic_loss': critic_losses, 'generator_loss': generator_losses}
        return self.history

    def predict_iv(self, x):
        iv = self.generator(x).numpy()
        return iv.squeeze()

    def predict_z(self, x):
        iv = self.predict_iv(x)
        z = (x[:,-1] - x[:,0])/np.sqrt(iv)
        return z