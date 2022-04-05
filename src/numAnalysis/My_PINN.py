import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

''' Istruzioni: devi inserire il tuo codice all'interno degli apici'''

class grid:
    def __init__(self, n):
        self.n_points = n
        self.x = tf.random.normal(shape=[self.n_points]).numpy()
        self.y = tf.random.normal(shape=[self.n_points]).numpy()
        self.xy = tf.stack((self.x,self.y), axis=1)


class NN:
    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam):

        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.dim = dim

        self.hidden_layers = [tf.keras.layers.Dense(n_neurons, activation=activation)] * n_layers
        self.layers = [tf.keras.layers.Dense(units=n_neurons, input_shape=(dim, ), activation=activation)] + self.hidden_layers + [tf.keras.layers.Dense(1)]

        self.model = tf.keras.Sequential(self.layers)
        self.last_loss_fit = tf.constant([0.0])
        self.learning_rate = learning_rate
        self.optimizer = opt(learning_rate)
        self.u_ex = u_ex


    def __call__(self,val):
        return self.model(val)

    def __repr__(self):
        return f"Number of layers: {self.n_layers}\n Number of neurons: {self.n_neurons}\n Activation function: {self.activation._tf_api_names[2]}\n Optimizer: {self.optimizer._keras_api_names[0]}\n Learning rate of the NN: {self.learning_rate}"

        

    def loss_fit(self,points): # aggiunta del tipo per utilizzare autocomplete
        x = tf.constant(points.x)
        y = tf.constant(points.y)
        u = self.model(tf.stack( (x, y), axis=1))
        u_ex = self.u_ex(points.x,points.y)
        self.last_loss_fit = tf.reduce_mean(tf.square(u_ex-tf.reshape(u,(x.shape[0],))))
        return self.last_loss_fit

    def fit(self, points, log, num_epochs=100):
        start = time.time()
        
        opt=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimization loop
        ts_ini = time.time()
        for i in range (num_epochs) :
            opt.minimize(lambda: self.loss_fit(points), self.model.variables)

        print(f'\n\n------ NN Model ------\n', file=log)
        print(f'elapsed time: {(time.time() - ts_ini):.2f} s', file=log)
        print(f'loss: {self.last_loss_fit.numpy()}', file=log)



class PINN(NN):

    def __init__(self, u_ex, n_layers = 3,
                       n_neurons = 4,
                       activation = tf.nn.tanh,
                       dim = 2,
                       learning_rate = 1e-3,
                       opt = tf.keras.optimizers.Adam,
                       mu = tf.Variable(1.0),
                       inverse = False):

        NN.__init__(self,u_ex,n_layers=n_layers, n_neurons=n_neurons, activation=activation, dim=dim, learning_rate=learning_rate, opt=opt)
        self.mu = mu
        self.last_loss_PDE = tf.constant([0.0])
        self.trainable_variables = [self.model.variables]
        if inverse:
          self.trainable_variables = self.trainable_variables + [self.mu]


    def loss_PDE(self, points):
        x = tf.constant(points.x)
        y = tf.constant(points.y)
        with tf.GradientTape (persistent = True) as tape:
            tape.watch(x)
            tape.watch(y)
            u = self.model(tf.stack( (x, y), axis=1))
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_xx = tape. gradient (u_x, x)
            u_yy = tape.gradient (u_y, y)
        self.last_loss_PDE = tf.reduce_mean(tf.square(-self.mu*(u_xx+u_yy)-tf.reshape(u,(x.shape[0],))))
        return self.last_loss_PDE

    def fit(self,points_int,points_pde,log,num_epochs=100):
        mu = self.mu
        opt=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        ts_ini = time.time()
        for i in range (num_epochs):
            opt.minimize(lambda: self.loss_fit(points_int) + self.loss_PDE(points_pde), self.trainable_variables)
            if (i%50 == 0):
                print (f'iter = {i}, mu = {mu.numpy()}, loss_fit = {self.last_loss_fit.numpy()}, loss_PDE = {self.last_loss_PDE.numpy()}')

        print(f'\n\n------ PINN Model ------\n', file=log)
        print(f'elapsed time: {(time.time() - ts_ini):.2f} s', file=log)
        print(f'loss fit: {self.last_loss_fit} s', file=log)
        print(f'loss pde: {self.last_loss_PDE} s', file=log)
