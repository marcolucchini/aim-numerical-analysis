import sys
import os
sys.path.append(os.getcwd())

print("cominciamo")

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

from numAnalysis.My_PINN import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################################################################
### Introduzione
############################################################################

# In questa challenge faremo uso della libreria Tensorflow per costruire
# Physics Informed Neural Networks (PINNs). Le PINNs, ossia reti neurali
# fisicamente informate, sono reti neurali (NN) grazie al quale possiamo
# risolvere equazioni differenziali.

# L'equazione che andremo a risolvere e' un equazione di Laplace con
# termine di reazione in R^2 ossia:
# -mu*Laplaciano(u) + u = 0

# Una rete neurale e', nel nostro caso, un modello di Machine Learning
# che permette di interpolare dei dati. I dati in questo caso coincidono
# con la soluzione della precedente equazione valutata in punti generici,
# creati aleatoriamente nel nostro codice.

# Di conseguenza, la rete neurale, si comporta come una funzione, che
# associa a punti di input un output. Il nostro obiettivo e' insegnarle
# a restituire dei dati in maniera piu' fedele possibile alla soluzione
# della equazione.

# Ovviamente, piu' dati della soluzione esatta diamo in pasto alla rete
# neurale, piu' essa sara' vicina alla soluzione che vogliamo.
# In molti casi pero' questi dati scarseggiano. E' qui che la PINN entra
# in gioco. Dando conoscenza alla rete neurale, non solo dei dati della
# soluzione, ma anche della equazione che essa risolve, possiamo fare si'
# che essa abbia bisogno di molti meno dati per poter approssimare la
# soluzione u esatta.

############################################################################
### Punto 0
############################################################################

f = open("output/logs.txt","w+")
f.close()
f = open("output/logs.txt", "a") # salveremo tutto l'output in un file .txt

# Generiamo aleatoriamente punti x,y in R per ottenere i dataset
# per l'interpolazione e l'equazione.

# ""input/NN_params.json" contiene i parametri della simulazione.
# "input/NN_params.json[0] contiene i parametri della rete, mentre
# "input/NN_params.json[1] il numero di punti di training e le iterazioni massime per l'ottimizzazione

# Modificate Questi Parametri a vostro piacimento cosi' da massimizzare
# le prestazioni della rete!
# Chi trovera' la rete che impieghera' piu' efficace / efficiente
# Avra' diritto a un premio. Potete modificare anche gli optimizer/funzioni
# di attivazione della NN se siete coraggiosi!

with open("input/My_NN_params.json", "r") as read_file:
    NN_params = json.load(read_file)
print("Input file parameters: ", NN_params, "\n\n", file=f)

# Inizializziamo seed per genarare puti casuali
np.random.seed(NN_params[1]["seed"])
tf.random.set_seed(NN_params[1]["seed"])

# mu e' un parametro della soluzione
mu_exact = 1.0

# soluzione esatta u = cos(x/sqrt(mu)) + sin(y/sqrt(mu))
u_ex = lambda x, y: np.cos(x/np.sqrt(mu_exact)) \
                  + np.sin(y/np.sqrt(mu_exact))

train_points = grid(NN_params[1]["num_train_points"]) # Punti di interpolazione
pde_points   = grid(NN_params[1]["num_pde_points"])   # Punti in qui valutiamo l'equazione
test_points  = grid(NN_params[1]["num_test_points"])  # Punti in cui verifichiamo la vicinanza tra PINN
u_test       = u_ex(test_points.x, test_points.y)     # e soluzoone esatta

############################################################################
### Punto 1
############################################################################

model_NN = NN(u_ex,**NN_params[0])

# Printate la rete neurale costruita, cosi' da vedere tutti i parametri scelti
print("Model\n", model_NN,  "\n\n")

# Grazie al metodo fit, allenate la rete neurale cosi' che interpoli i valori
# della soluzione esatta
model_NN.fit(train_points, f, num_epochs=NN_params[1]["num_epochs"])

# In un grafico, plottate la funzione costituita dalla rete neurale e la
# soluzione esatta
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(test_points.x, test_points.y, u_test)
ax.scatter(test_points.x, test_points.y, model_NN(test_points.xy).numpy())
ax.legend(('learned solution','exact solution'))
fig.savefig('output/interpolationProblem.png', bbox_inches='tight')

############################################################################
### Punto 2
############################################################################

# Create un oggetto model_NN di tipo PINN
model_PINN = PINN(u_ex,**NN_params[0])

# Allenate la PINN
model_PINN.fit(train_points, pde_points, f, num_epochs=NN_params[1]["num_epochs"])

# In un grafico, plottate la funzione costituita dalla rete neurale e la
# soluzione esatta
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(test_points.x, test_points.y, u_test)
ax.scatter(test_points.x, test_points.y, model_PINN(test_points.xy).numpy())
ax.legend(('learned solution','exact solution'))
fig.savefig('output/pinnProblem.png', bbox_inches='tight')

###########################################################################
### Punto 3
############################################################################

# Le PINN non sono solo metodi molto veloci per risolvere equazioni differenziali
# La loro utilità maggiore al giorno d'oggi, e' basata sulla abilità di risolvere
# i cosiddetti problemi inversi. Supponiamo infatti di conoscere a priori la
# soluzione dell'equazione. Tuttavia non conosciamo per esempio un parametro
# che è presente nella equazione stessa. Nel nostro caso parliamo di mu,
# il coeffiente del termine laplaciano della equazione.
# Grazie alla PINN possiamo stimare tale parametro, in quanto sarà quello per cui
# l'equazione è effettivamente risolta.
# Cio' è al giorno d'oggi una dei campi di ricerca principali nel calcolo scientifico
# in quanto permetterebbe di misurare parametri di fenomeni fisici basandosi su equazioni
# e non su misurazioni dirette, spesso complicate (o insalutari) da compiere.

mu_guess = 1.5
# Create un oggetto model_NN di tipo PINN con un parametro mu da identificare
model_PINN_inverse = PINN(u_ex, mu=tf.Variable(mu_guess), inverse=True, **NN_params[0])

# Allenate la PINN e stimate mu
model_PINN_inverse.fit(train_points, pde_points, f, num_epochs=NN_params[1]["num_epochs"])

# Mostrate il risultato il mu
print('estimated mu:   %f' % model_PINN_inverse.mu.numpy(), file=f)
print('relative error: %1.2e' % (abs(model_PINN_inverse.mu.numpy() - mu_exact)/mu_exact), file=f)
