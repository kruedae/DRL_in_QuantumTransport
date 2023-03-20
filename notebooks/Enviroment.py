#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kruedae/DRL_in_QuantumTransport/blob/master/Enviroment_construction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Tres puntos cuanticos

# ## Usando qutip

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import *


# In[2]:


r0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
ro = Qobj(r0)

# Definiendo nuestro hamiltoniano real
omegamax = 0.2
tmax = 50*np.pi/omegamax
def H1_coeff(t, args):
  sigma = tmax/8
  return omegamax*np.exp(-(t - ((tmax+sigma)/2))**2.0/(2*sigma**2.0))
def H2_coeff(t, args):
  sigma = tmax/8
  return omegamax*np.exp(-(t - ((tmax-sigma)/2))**2.0/(2*sigma**2.0))

H0 = np.array([[0,0,0],[0,0,0],[0,0,0]])
H0 = Qobj(H0)
H1 = np.array([[0,-1,0],[-1,0,0],[0,0,0]])
H1 = Qobj(H1)
H2 = np.array([[0,0,0],[0,0,-1],[0,-1,0]])
H2 = Qobj(H2)
H = [H0, [H1, H1_coeff], [H2, H2_coeff]]


#Dinamica del sistema
N = 1000
tlist = np.linspace(0,50,N)*np.pi/omegamax
result = mesolve(H, ro, tlist)

p11 = []
p22 = []
p33 = []

final_result = [result.states[i].full() for i in range(N)] 
for paso in final_result:
  densidad_matrix = np.reshape(paso,(9,))
  p11.append(densidad_matrix[0])
  p22.append(densidad_matrix[4])
  p33.append(densidad_matrix[8]) 

plt.plot(tlist, H1_coeff(tlist,[]), label= r'$\Omega_{1}$')
plt.plot(tlist, H2_coeff(tlist,[]), label=r'$\Omega_{2}$')
plt.title("Pulsos")
plt.legend()
plt.show()


plt.plot(tlist, np.real(p11), label='rho1')
plt.plot(tlist, np.real(p22), label='rho2')
plt.plot(tlist, np.real(p33), label='rho3')
plt.title("Densidades de probabilidad")
plt.legend()
plt.show()


# ## Usando odeintw

# In[3]:


get_ipython().system('pip install odeintw')
from odeintw import odeintw


# In[4]:


omegamax = 0.2
tmax = 50*np.pi/omegamax
sigma = tmax/8

def right_part(rho, t):
    omega1 = omegamax*np.exp(-(t - ((tmax+sigma)/2))**2.0/(2*sigma**2.0))
    omega2 = omegamax*np.exp(-(t - ((tmax-sigma)/2))**2.0/(2*sigma**2.0))

    hamiltonian = np.array(
        [[0     , -omega1,      0  ],
        [-omega1,    0.0 , -omega2 ],
        [ 0     , -omega2,      0.0]],
        dtype=np.complex128)
    return (np.dot(hamiltonian, rho) - np.dot(rho, hamiltonian)) / (1j)


psi_init = np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]], dtype=np.complex128)



# # detunning

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
get_ipython().system(' pip install qutip')
from qutip import *


# In[ ]:


r0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
ro = Qobj(r0)


# Definiendo nuestro hamiltoniano real
omegamax = 0.2
tmax = 50*np.pi/omegamax
def H1_coeff(t, args):
  sigma = tmax/8
  return omegamax*np.exp(-(t - ((tmax+sigma)/2))**2.0/(2*sigma**2.0))
def H2_coeff(t, args):
  sigma = tmax/8
  return omegamax*np.exp(-(t - ((tmax-sigma)/2))**2.0/(2*sigma**2.0))

#Agregamos terminos de tunning
H0 = np.array([[0,0,0],[0,0.1,0],[0,0,0.1]])
H0 = Qobj(H0)
H1 = np.array([[0,-1,0],[-1,0,0],[0,0,0]])
H1 = Qobj(H1)
H2 = np.array([[0,0,0],[0,0,-1],[0,-1,0]])
H2 = Qobj(H2)
H = [H0, [H1, H1_coeff], [H2, H2_coeff]]

#Dinamica del sistema 
N = 1000
tlist = np.linspace(0,50,N)*np.pi/omegamax
result = mesolve(H, ro, tlist)

p11 = []
p22 = []
p33 = []

final_result = [result.states[i].full() for i in range(N)] 
for paso in final_result:
  densidad_matrix = np.reshape(paso,(9,))
  p11.append(densidad_matrix[0])
  p22.append(densidad_matrix[4])
  p33.append(densidad_matrix[8])


plt.plot(tlist, np.real(p11), label='rho1')
plt.plot(tlist, np.real(p22), label='rho2')
plt.plot(tlist, np.real(p33), label='rho3')
plt.title("Densidades de probabilidad")
plt.legend()
plt.show()


# # 5 PUNTOS

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
get_ipython().system(' pip install qutip')
from qutip import *


# In[ ]:


# Matrix densidad inicial 
r0 = np.array([[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
ro = Qobj(r0)

# Definiendo nuestro hamiltoniano real
omegamax = 1
tmax = 100*np.pi/omegamax
def H1_coeff(t, args):
  sigma = tmax/8
  return omegamax*np.exp(-(t - ((tmax+sigma)/2))**2.0/(2*sigma**2.0))
def H2_coeff(t, args):
  sigma = tmax/8
  return omegamax*np.exp(-(t - ((tmax-sigma)/2))**2.0/(2*sigma**2.0))
def H3_coeff(t, args):
  sigma = tmax/8
  return 5.0*omegamax*np.exp(-(t - ((tmax)/2))**2.0/(2*sigma**2.0))


H0 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
H0 = Qobj(H0)
H1 = np.array([[0,-1,0,0,0],[-1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
H1 = Qobj(H1)
H2 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,-1],[0,0,0,-1,0]])
H2 = Qobj(H2)
H3 = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,0,-1,0],[0,0,-1,0,0],[0,0,0,0,0]])
H3 = Qobj(H3)
H = [H0, [H1, H1_coeff], [H2, H2_coeff], [H3, H3_coeff]]

# Dinamica del sistema
N = 1000
tlist = np.linspace(0,100,N)*np.pi/omegamax
result = mesolve(H, ro, tlist)


p11 = []
p22 = []
p33 = []
p44 = []
p55 = []

final_result = [result.states[i].full() for i in range(N)] 
for paso in final_result:
  densidad_matrix = np.reshape(paso,(25,))
  p11.append(densidad_matrix[0])
  p22.append(densidad_matrix[6])
  p33.append(densidad_matrix[12])
  p44.append(densidad_matrix[18])
  p55.append(densidad_matrix[24])



# # Construyendo el Entorno

# ###Probando la propiedad de Markov del proceso:
# 
# Queremos ver si podemos calcular ρ(t+1), usando la dinámica con ρ(t) como condicion inicial y recalculando todo.

# In[ ]:


from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().system('pip install odeintw')
from odeintw import odeintw


class PassageEnv(Env):
    def __init__(self):
        # Limites y espacio de las acciones [omega1, omega2]
        self.action_space = Box(low=np.array([0,0]), high=np.array([1,1]))
        #  Limites y espacio de los estados: [\rho, sigma1, sigma2]
        self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0]), 
                                high=np.array([1,1,1,1,1,1,1,1,1,1,1]))
        # Set start state
        self.state = np.array([1,0,0,0,0,0,0,0,0,0,0],dtype=np.complex128)

        self.length_trial = 0


    def refuerzo(self, p3, p2):
        return (-1 + p3 - p2) #+ A(p2)

    def step(self, action):
        # Aplicar accion
        omega1 = action[0]
        omega2 = action[1]

        # Cambiar el hamiltoniano
        def right_part(rho, t):
  
          hamiltonian = np.array([[0     , -omega1,      0  ],
                                  [-omega1,    0.0 , -omega2 ],
                                  [ 0     , -omega2,      0.0]],
                                  dtype=np.complex128)
          return (np.dot(hamiltonian, rho) - np.dot(rho, hamiltonian)) / (1j)

        # Evolucion temporal
        N = 100
        omegamax = 1
        dt = (50*np.pi/omegamax)/N
        #tmax = 10
        tpasos = np.linspace(0,dt,N)
        rho = np.array([[self.state[i] for i in range(0,3)],
                      [self.state[i] for i in range(3,6)],
                      [self.state[i] for i in range(6,9)]], dtype=np.complex128)

        result = odeintw(right_part, rho, tpasos)
        final_result = result[-1]
        final_result = np.reshape(final_result,(9,))

        # Nuevo estado a partir de la matriz densidad obtenida
        result = final_result
        #result = [np.mean(i) for i in final_result]
        #result = [i[-2] for i in final_result] # Testeando
        self.state = np.array([result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], omega1, omega2])

        # Calcular el refuerzo
        reward = self.refuerzo(self.state[6], self.state[8])

        done = False

        if self.state[4]>0.05:
          reward -= 100

        self.length_trial += 1
      
        if self.state[8] >= 0.98:
            done = True

        if self.length_trial >= 100:
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = np.array([1,0,0,0,0,0,0,0,0,0,0])
        self.length_trial = 0
        return self.state
