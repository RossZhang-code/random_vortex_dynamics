#!/usr/bin/env python
# coding: utf-8



from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2 , rfft2 , irfftn , rfftn
from numpy.fft import fftfreq
from numpy.fft import fftn, ifftn





# The assigned size of the mesh
M=7
# Actual number of nodes in each direction
N = 2**M
# Physical size of computational box
L = 2*pi
Nf = int(N/2+1)
# The mesh
X = mgrid[:N, :N, :N].astype(float)*L/N

kx = ky = fftfreq(N, 1./N)
kz = kx[:(int(N/2+1))].copy(); kz[-1] *= -1
K = array(meshgrid(kx, ky, kz, indexing='ij'), dtype=int)
K2 = sum(K*K, 0, dtype=int)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
kmax_dealias = 2./3.*(N/2+1)
dealias = array((abs(K[0]) < kmax_dealias)*(abs(K[1]) < kmax_dealias)* (abs(K[2]) < kmax_dealias), dtype=bool)

U = empty((3, N, N, N), dtype=float)
U[0] = 1*cos(X[0])*sin(X[1])*sin(X[2])+cos(X[0])*sin(X[1])*sin(2*X[2])
U[1] =-1*sin(X[0])*cos(X[1])*sin(X[2])+sin(X[0])*cos(X[1])*sin(2*X[2])
U[2] = -1*sin(X[0])*sin(X[1])*cos(2*X[2])
U_hat = empty((3, N, N, Nf), dtype=complex)
P = empty((N, N, N), dtype=float)
P_hat = empty((N, N, Nf), dtype=complex)
curl = empty((3, N, N, N), dtype=float)


# In[3]:


def fftn_mpi(u):

    fu = rfftn(u)
    return fu
def ifftn_mpi(u):
    fu = irfftn(u)
    return fu
# Usage
U_hat = rfftn(U,axes=(1,2,3))
U = irfftn(U_hat,axes=(1,2,3))
def Cross(a, b):
    c = empty((3, N,N, Nf), dtype=complex)
    c[0] =rfftn(a[1]*b[2] -a[2]*b[1])
    c[1] = rfftn(a[2]*b[0] -a[0]*b[2])
    c[2] = rfftn(a[0]*b[1] -a[1]*b[0])
    return c
def Curl(a):
    c = empty((3, N, N, N), dtype=complex)
    c[2] = irfftn(1j*(K[0]*a[1] -K[1]*a[0]))
    c[1] = irfftn(1j*(K[2]*a[0] -K[0]*a[2]))
    c[0] = irfftn(1j*(K[1]*a[2] -K[2]*a[1]))
    return c
nu=1/1600
dU = empty((3, N, N, Nf), dtype=complex)
dt = 0.01 # Time step nu = 0.001 # Viscosity
def computeRHS(dU):
    #global curl
# Compute convective term
    curl = Curl(U_hat)
    dU = Cross(U, curl)
    #print(dU)
    # Compute pressure
    P_hat[:] = sum(dU*K_over_K2 , 0)
    # Subtract pressure gradient
    dU -= P_hat*K
    # Subtract viscous term
    dU -= nu*K2*U_hat
    #print(dU[:,0,0,0])
    return dU
t = 0 # Physical time
T = 1 # End time while t <= T:
U_list = []
while t <= T:
    t += dt
    U_hat += computeRHS(dU)*dt
    for i in range(3):
        U[i] = irfftn(U_hat[i])
    U_list.append(U.copy())




import matplotlib.pyplot as plt


discrete_points=X[:,32:96:4,32:96:4,[2]] 
discrete_vel = U_list[-1][:,32:96:4,32:96:4,[2]] #U_list[n], n represents the time is at 0.01n

u,v = discrete_vel[0],discrete_vel[1]
x,y = discrete_points[0],discrete_points[1]
fig_vel,ax = plt.subplots()
qui = ax.quiver(x.reshape(-1),y.reshape(-1),u.reshape(-1),v.reshape(-1),scale=5)

plt.show()
