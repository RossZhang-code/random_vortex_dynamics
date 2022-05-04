#!/usr/bin/env python
# coding: utf-8




import numpy as np
from scipy import integrate

from ttictoc import tic, toc

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def Kernel(x):
    r = np.maximum(np.sqrt(np.sum(x**2,axis=1)),0.1)
    r = r.reshape(-1,1)

    out = -1/(4*np.pi)*x/(r**3+(0.1**16))
    return out
K = Kernel
def K_derivative(x,y):
    #Given an n*3 x, given an n*3 y, return n*3*3 matrix
    r = np.maximum(np.sqrt(np.sum(x**2,axis=1)),0.1)
    coef = 1/(r**5)*3/(8*np.pi)
    #print(coef.shape)
    coef = coef.reshape(-1,1,1)
    temp = np.cross(x,y)
    x = np.expand_dims(x,2)
    temp = np.expand_dims(temp,1)
    result = np.einsum('ijk,ikl->ijl', x, temp)
    result +=np.transpose(result,(0,2,1))
    result=result*coef


    return result


#n is the number of copies
n =100
x_start=np.tile(np.mgrid[0:0:1j, 0:0:1j, -10:10:41j].reshape(3,-1).T,(n,1))
w_start=np.tile(np.array([[0.0],[0.0],[0.5]]),(41*n,1,1))/n
num = x_start.shape[0]


#initialize the time and noise
t_interval= np.linspace(0,0.2,10,endpoint=False)
delta_t = t_interval[1]-t_interval[0]
noise_list = []
for i in range(n):
    noise_list.append(np.tile(np.random.randn(len(t_interval),3),(int(num/n),1,1)))
white_noise = np.concatenate(noise_list,axis=0)


from joblib import Parallel, delayed

result_X = [x_start]
result_G = [np.tile(np.eye(3),(num,1,1))]
result_Gpath = []
X = result_X[-1]
G = result_G[-1]

def Gpath_calc(k):
    matrix_ = np.sum(K_derivative(X[k]-X,np.squeeze((np.matmul(G,w_start)))),axis = 0)
    return 0.5*matrix_+0.5*matrix_.T
c = np.array(Parallel(n_jobs=200)(delayed(Gpath_calc)(_) for _ in range(num)))
result_Gpath.append(c.copy())


def evolution():
    for p in range(len(t_interval)):
        #Next step x
        X = result_X[-1]
        G = result_G[-1]
        x_new = np.empty((num,3))
        tic()
        def para(j):
            return (X[j]+np.sum(np.cross(K(X[j]-X),np.squeeze(np.matmul(G,w_start)) ),axis = 0)
            * delta_t+ white_noise[j][p] * np.sqrt(delta_t))
        x_new = np.array(Parallel(n_jobs=200)(delayed(para)(_) for _ in range(num)))

        print(toc())
        #Next step G
        G = np.tile(np.eye(3),(num,1,1))

        for j in result_Gpath[::-1]:
            G = G + np.matmul(G,
                j)*delta_t
        G_new = G.copy()

        #Next step Gpath
        tic()
        def Gpath_calc(k):
            matrix_ = np.sum(K_derivative(x_new[k]-x_new,np.squeeze((np.matmul(G_new,w_start)))),axis = 0)
            return 0.5*matrix_+0.5*matrix_.T###G[i] changed to G[j] here
        c = np.array(Parallel(n_jobs=200)(delayed(Gpath_calc)(_) for _ in range(num)))
        print(toc())
        ###
        result_X.append(x_new)
        result_G.append(G_new)
        result_Gpath.append(c)




#run

evolution()
output = (result_X,result_G,result_Gpath)






class field():
    def __init__(self, x,g,w):
        self.loc_hist = x
        self.vor_mat = g
        self.vor_start = w
        self.num = x[0].shape[1]
    def curve(self, t, n):
        index = np.where(t_interval<=t)[0][-1]
        return self.loc_hist[:index,n], self.vor_hist[:index,n]
    def field(self,t):
        index = np.where(t_interval<=t)[0][-1]
        if t_interval[-1]<t:
            index = -1
        particles = self.loc_hist[index]
        vortices = np.squeeze(np.matmul(self.vor_mat[index],self.vor_start))
        def vorticity_field(x):
            return np.sum(np.apply_along_axis(K,1,x-particles).reshape(-1,1)*vortices,0)
        def velocity_field(x):
            return -np.sum(np.cross(vortices,K(x - particles)),0)
        return vorticity_field,velocity_field


a = field(result_X,result_G,w_start)





###figures of simulation
vel = a.field(0.1)[1]#0.1 represent the time
x, y ,z= np.meshgrid(np.arange(-1,1, 0.1),
                    np.arange(-1,1, 0.1),0)
x = np.expand_dims(x,3)
y = np.expand_dims(y,3)
z = np.expand_dims(z,3)
discrete_points = np.concatenate([x,y,z],3)
discrete_vel = np.array(Parallel(n_jobs=200)(delayed(vel)(_) for _ in discrete_points.reshape(-1,3)))
x, y ,z= np.meshgrid(np.arange(-1,1, 0.1),
                    np.arange(-1,1, 0.1),0)
u,v = discrete_vel[:,0],discrete_vel[:,1]
fig_vel,ax = plt.subplots()
qui = ax.quiver(x.reshape(-1),y.reshape(-1),u.reshape(-1),v.reshape(-1),scale = 5)

plt.show()





#Figures of Ossen-Lamb vortex by explicit expression
def standard_2D(t):
    def f(x):
        x_per = np.array([-x[1],x[0]])
        r_square = np.sum(x[:-1]**2+1e-16)
        return 1/(2*np.pi)*x_per/r_square*(1-np.exp(-r_square/(2*t)))
    return f
x, y ,z= np.meshgrid(np.arange(-1,1, 0.1),
                      np.arange(-1,1,0.1),0)
x = np.expand_dims(x,3)
y = np.expand_dims(y,3)
z = np.expand_dims(z,3)
discrete_points = np.concatenate([x,y,z],3)
discrete_vel = np.apply_along_axis(standard_2D(0.1),-1,discrete_points)
x, y= np.meshgrid(np.arange(-1,1, 0.1),
                      np.arange(-1,1,0.1))
u,v = discrete_vel[:,:,:,0],discrete_vel[:,:,:,1]
fig_vel,ax = plt.subplots()
qui = ax.quiver(x.reshape(-1),y.reshape(-1),u.reshape(-1),v.reshape(-1),scale = 5 )

plt.show()





#To generate the table of ossen-lamb vortex in the paper. We calculate the discrete_vel on the grid
x, y ,z= np.meshgrid(np.arange(-1,1.5, 0.5),
                    np.arange(-1,1.5, 0.5),0)
x = np.expand_dims(x,3)
y = np.expand_dims(y,3)
z = np.expand_dims(z,3)
discrete_points = np.concatenate([x,y,z],3)
discrete_vel = np.array(Parallel(n_jobs=200)(delayed(vel)(_) for _ in discrete_points.reshape(-1,3)))





#L^1 loss
x, y ,z= np.meshgrid(np.arange(-1,1.1, 0.1),
                    np.arange(-1,1.1, 0.1),np.arange(-1,1.1, 0.1))
x = np.expand_dims(x,3)
y = np.expand_dims(y,3)
z = np.expand_dims(z,3)
discrete_points = np.concatenate([x,y,z],3)
discrete_vel = np.array(Parallel(n_jobs=200)(delayed(vel)(_) for _ in discrete_points.reshape(-1,3)))
real_vel = np.apply_along_axis(standard_2D(0.1),-1,discrete_points).reshape(-1,2)
loss = (np.sum(np.abs(discrete_vel[:,[0,1]]-real_vel))+np.sum(np.abs(discrete_vel[:,-1])))*(0.1**3)
