#!/usr/bin/env python
# coding: utf-8



import numpy as np
from scipy import integrate


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
    coef = coef.reshape(-1,1,1)
    temp = np.cross(x,y)
    x = np.expand_dims(x,2)
    temp = np.expand_dims(temp,1)
    result = np.einsum('ijk,ikl->ijl', x, temp)
    result +=np.transpose(result,(0,2,1))
    result=result*coef


    return result


#### initialize the data, data should be
n =1
x = np.arange(-np.pi,3*np.pi+np.pi/16,np.pi/16)
y = np.arange(-np.pi,3*np.pi+np.pi/16,np.pi/16)
z = np.arange(-np.pi,3*np.pi+np.pi/16,np.pi/16)
xx,yy,zz = np.meshgrid(x, y,z)
x_start = np.array((xx.ravel(), yy.ravel(),zz.ravel())).T


def TGV_start(x):
    #c2 = 1,start from t = 1
    a = 1*np.sin(x[0])*np.cos(x[1])*np.cos(x[2])
    b = 1*np.cos(x[0])*np.sin(x[1])*np.cos(x[2])
    c = -2*1*np.cos(x[0])*np.cos(x[1])*np.sin(x[2])
    return (np.pi/16)**3*np.array([[a],[b],[c]])
w_start=np.tile(np.apply_along_axis(TGV_start,1,x_start),(n,1,1))/n
x_start = np.tile(x_start,(n,1))
num = x_start.shape[0]


t_interval= np.linspace(0,1,50,endpoint=False)
delta_t = t_interval[1]-t_interval[0]
white_noise = np.random.randn(num,len(t_interval),3)

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
        def para(j):
            return (X[j]+np.sum(np.cross(K(X[j]-X),np.squeeze(np.matmul(G,w_start)) ),axis = 0)
            * delta_t+ white_noise[j][p] * np.sqrt(2)/40*np.sqrt(delta_t))
        x_new = np.array(Parallel(n_jobs=200)(delayed(para)(_) for _ in range(num)))

        #Next step G
        G = np.tile(np.eye(3),(num,1,1))

        for j in result_Gpath[::-1]:
            G = G + np.matmul(G,
                j)*delta_t
        G_new = G.copy()

        #Next step Gpath
        def Gpath_calc(k):
            matrix_ = np.sum(K_derivative(x_new[k]-x_new,np.squeeze((np.matmul(G_new,w_start)))),axis = 0)
            return 0.5*matrix_+0.5*matrix_.T###G[i] changed to G[j] here
        c = np.array(Parallel(n_jobs=200)(delayed(Gpath_calc)(_) for _ in range(num)))
        ###
        result_X.append(x_new)
        result_G.append(G_new)
        result_Gpath.append(c)




#run

evolution()
output = (result_X,result_G,result_Gpath)





# In[ ]:


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
            #return -np.sum(np.cross(vortices,np.apply_along_axis(K1,1,x - particles)),0)
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
qui = ax.quiver(x.reshape(-1),y.reshape(-1),u.reshape(-1),v.reshape(-1),scale = 1)

plt.show()



#TGV explicit solution
def TGV(t,nu):
    delta1 = 1-3*t*nu+(9*(nu**2)-1/16)*(t**2)/2-(27/2*(nu**2)-23/32)*(t**3)/3*nu
    alpha1 = -1/8*(t-7*t**2*nu+(74/3*nu**2-43/(44*12))*(t**3))
    beta1 = -1/16*(t**2)*(1/2-13/2*t*nu)
    gamma2 = -15/176*(t**2)*(1/2-31/6*t*nu)
    gamma3 = -1/16*(t**2)*(1/2-31/6*t*nu)
    def f(x):
        #a=1,gamma =1
        u = (delta1*np.cos(x[0])*np.sin(x[1])*np.sin(x[2])+alpha1*np.sin(2*x[0])*np.cos(2*x[2]) +
             2/3*gamma2*np.cos(3*x[0])*np.sin(x[1])*np.sin(x[2]) + gamma2*np.cos(x[0])*np.sin(3*x[1])*np.sin(x[2])
            -gamma3*np.cos(x[0])*np.sin(x[1])*np.sin(3*x[2])-beta1*np.cos(3*x[0])*np.sin(x[1])*np.sin(3*x[2]))

        v = (-delta1*np.sin(x[0])*np.cos(x[1])*np.sin(x[2])+alpha1*np.sin(2*x[1])*np.cos(2*x[2]) -
             gamma2*np.sin(3*x[0])*np.cos(x[1])*np.sin(x[2]) -2*gamma2/3*np.sin(x[0])*np.cos(3*x[1])*np.sin(x[2])
             +gamma3*np.sin(x[0])*np.cos(x[1])*np.sin(3*x[2]) + beta1*np.sin(x[0])*np.cos(3*x[1])*np.sin(3*x[2]))
        w = (-alpha1*(np.cos(2*x[0])+np.cos(2*x[1]))*np.sin(2*x[2])-gamma2*np.sin(3*x[0])*np.sin(x[1])*np.cos(x[2])
             +gamma2*np.sin(x[0])*np.sin(3*x[1])*np.cos(x[2]) + beta1*np.sin(3*x[0])*np.sin(x[1])*np.cos(3*x[2])
             -beta1*np.sin(x[0])*np.sin(3*x[1])*np.cos(3*x[2]))


        return          np.array([u,v,w])
    return f

x, y ,z= np.meshgrid(np.arange(np.pi/2,3*np.pi/2, np.pi/16),
                    np.arange(np.pi/2,3*np.pi/2, np.pi/16),0.1)
x = np.expand_dims(x,3)
y = np.expand_dims(y,3)
z = np.expand_dims(z,3)
discrete_points = np.concatenate([x,y,z],3)
discrete_vel = np.apply_along_axis(TGV(1,1/1600),-1,discrete_points)
x, y ,z= np.meshgrid(np.arange(np.pi/2,3*np.pi/2, np.pi/16),
                    np.arange(np.pi/2,3*np.pi/2, np.pi/16),0.1)
u,v = discrete_vel[:,:,:,0],discrete_vel[:,:,:,1]
fig_vel,ax = plt.subplots()
qui = ax.quiver(x.reshape(-1),y.reshape(-1),u.reshape(-1),v.reshape(-1),scale=1)

plt.show()
