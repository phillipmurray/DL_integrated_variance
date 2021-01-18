from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm_notebook
import numpy as np


@tf.function
def create_train_dataset(x_train, batch_size):
    """Creates a batched dataset from a given numpy array
    """
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset



class CNNetwork(Model):

    def __init__(self,n_timesteps, n_features, n_outputs,loss=None):
        super().__init__()
        self.timesteps = n_timesteps
        self.features = n_features
        self.outputs = n_outputs
        layers = []
        layers.append(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        layers.append(Conv1D(filters=64, kernel_size=3, activation='relu'))
        layers.append(Dropout(0.5))
        layers.append(MaxPooling1D(pool_size=2))
        layers.append(Flatten())
        layers.append(Dense(100, activation='relu'))
        layers.append(Dense(n_outputs, activation='softmax')) 
        self.h_layers = layers
        self.loss = loss
        self.optimizer = Adam(lr=0.001)
        self.compile(loss = loss)

    def call(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return x

    def train_step(self, x_batch):
        total_increment = x_batch[:, -1] - x_batch[:, 0]
        with tf.GradientTape() as tape:
            int_var = self(np.reshape(x_batch, (x_batch.shape[0],x_batch.shape[1], 1)))
            int_var = tf.squeeze(int_var)
            loss_value = self.loss(total_increment, int_var)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value

    def train(self, x_train, num_epochs, batch_size):
        n_steps = x_train.shape[0] // batch_size
        x_train = create_train_dataset(x_train, batch_size)
        losses = []
        mses = []
        for epoch in range(num_epochs):
            with tqdm_notebook(total=n_steps, desc=f'Epoch {epoch+1} of {num_epochs}') as progress:
                for step, x_batch in enumerate(x_train):
                    progress.update()
                    loss_val = self.train_step(x_batch)
                    losses.append(loss_val.numpy())

        self.history = {'loss': losses, 'mse': mses}
        return self.history    


        def predict_iv(self, x):
            iv = self(np.reshape(xbatch, (x.shape[0],x.shape[1], 1))).numpy()
            return iv.squeeze()