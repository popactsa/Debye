# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:15:52 2021

@author: М
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 19:32:09 2021

@author: М
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:46:35 2021

@author: М
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:51:14 2021

@author: М
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:49:32 2020

@author: М
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 4 18:14:16 2017

The code is intended for the simulation of the 1D evaporation problem.

@author: A.A. Stepanenko NRNU MEPhI
@form of the heat equation and fluxes: V.S. Norakidze NRNU MEPhI
"""
from scipy import integrate
from scipy.integrate import ode
from scipy.special import *
import scipy.optimize as opt
from numpy import *
#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import datetime
import time as tm
from scipy.interpolate import UnivariateSpline

##############################################################################
tstart = tm.time()
# Input parameters for the problem
# dc/dt = d/dx(D dc/dx) - dy/dt - diffusion of the solubilized(солюбилизированный) gas in the sample
# dy/dt = f exp(-Q/T) (c/C) (y - Y) - f exp(-(Q+E)/T) y - balance of the trapped particle density
# c(x = 0) = c0 (for now), c(x = L) = cr << 1, c(t = 0) = 0 except for the surface, where c = c0
# y(t = 0) = 0
# The problem is solved by the method of lines.

# the set of physical constants
k = 1.38e-23  # the Boltzmann constant, J/K

# the physical parameters for the problem
L   = 1e-2  # thickness of the sample, m
Tt  = 300.0   # the temperature of the thermostat, K
tau = 1e0     # the characteristic timescale for the problem, s
rho = 19.25e3   # the mass density of the target material, kg/сm^3 
# for tungsten, rho = 19.25 g/cm^3
kappa = 140
Cp = 140
Q0  = 1.0e7   # the maximum heat flux in the impulse, J/m^2/s

C = tau*kappa/(L**2*rho*Cp)
D = -Tt*kappa/(Q0*L)

# the user-defined simulation parameters for the problem
Tstop = 1000    # the end time for the computation, s
Ntime = 100    # the total number of the time points to compute
nnodes = 20    # number of EVOLVING(изменяющихся) nodes for the solution
#T = [0.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7] # all physical time moments to compute, s

# Solver tolerances(допуски решателя)
rtol = 1.e-9
atol = 1.e-9

# define whether to plot concentration of gas in traps or sollubilized state
SaveAnim  = False   # save the animated movie?
SaveData  = True    # evaluate data found within the exact approach?
#SaveAnlt  = False  # evaluate analytic data?

# derived parameters for the problem
# produce the time grid
T = linspace(0, Tstop, Ntime)

# temporal parameters
tmp = copy.deepcopy(T) # create a deep copy of the time array
#tmp.pop(0)             # remove the array element with the initial time moment
tmp = delete(tmp, 0)   # remove the array element with the initial time moment
T  = T/tau             # convert the physical time to the simulation time
t  = tmp/tau           # convert the physical time to the simulation time omitting the initial moment
nT = t.size            # number of timesteps to calculate
dt = zeros(T.size)     # create the array to store timesteps
dt[1:] = t[:] - T[:(T.size-1)] # calculate the timesteps
# spatial(пространственный) variables
dx     = 1./(nnodes + 1.) # the step size along x
                          # the expression is cross-checked with the profile for the temperature
x      = linspace(dx, 1. - dx, nnodes) # the x grid point coordinates, where the profiles evolve
x_ext  = linspace(0., 1., nnodes + 2)  # the extended set of the x grid point coordinates, including the guard cells
dx12   = 12.*dx       # extra metric factor
d2x12  = 12.*dx**2    # extra metric factor
f1d2x12  =  C * 1./d2x12 # extra metric factor
f16d2x12 =  C * 16./d2x12 # extra metric factor
f30d2x12 = C* -30./d2x12 # extra metric factor
# the total amount of grid points
ntotal  = nnodes + 4

# the array to store computed data
u = ndarray((len(T), ntotal))
u_test = u.copy()
# the nodes map is as follows
# 0, 1 - c at the left border
# 2, ..., N+1 - c in the simulation domain
# N+2, N+3 - c at the right border


def ext_heat_flux(t):
    return 25

def rhs(t, u):
    # define the ydot vector
    dudt = ndarray((ntotal))

    #u[-2] = u[nnodes -1] = u[-3] # no heat flux at the thermostat side

    # define udot at the left border and the leftmost node of the simulation domain
    # no temperature variation at the inlet side, u(x=0) is preset manually
    dudt[0] = dudt[1] = 0. 
    
    u[0] = u[1] = u[2] - dx/D * ext_heat_flux(t)
    print(dx / D * ext_heat_flux(t) * Tt * L)
    
    # define udot in the simulation domain
    for i in range(2, ntotal - 2):
        dudx   = ( u[i-2] -  8.*u[i-1]            +  8.*u[i+1] - u[i+2])/dx12
        d2udx2 = (-u[i-2] + 16.*u[i-1] - 30.*u[i] + 16.*u[i+1] - u[i+2])/d2x12
        
        dudt[i] = C*d2udx2

    # define cdot at the right border
    dudt[-2] = dudt[-1] = 0.

    return dudt

def jac(t, u):
    dFdu = zeros((ntotal, ntotal))

    u[0] = u[1] = u[2] - dx/D * ext_heat_flux(t)

    def delta(i, j):
        if i == j:
            return 1
        else: 
            return 0
    for i in range(2, ntotal - 2):
        dudx   = ( u[i-2] -  8.*u[i-1]            +  8.*u[i+1] - u[i+2])
        d2udx2 = (-u[i-2] + 16.*u[i-1] - 30.*u[i] + 16.*u[i+1] - u[i+2])
        
        dFdu[i, i-2] = -f1d2x12
        dFdu[i, i-1] = f16d2x12
        dFdu[i, i]   = f30d2x12
        dFdu[i, i+1] = f16d2x12
        dFdu[i, i+2] = -f1d2x12
        
    return dFdu  

#-----------------------------------------------------------------------------
# SOLUTION FOR THE EXACT DIFFUSION SYSTEM
if SaveData:
    u0 = ones((ntotal))
    u[0,:] = u0[:]
        
    # VODE solver
    r = ode(rhs, jac).set_integrator('vode', method = 'bdf', nsteps = 1e9, atol = atol, rtol = rtol)
    #r = ode(rhs, jac).set_integrator('vode', method='bdf', nsteps = 50000)
    # r = ode(rhs).set_integrator('vode', method='bdf', nsteps = 1e9 , atol = atol, rtol = rtol)
    r.set_initial_value(u0, T[0])
    index = 1
    print('Beginning to solve the thermal conduction system')
    while r.successful() and index <= nT:
        tstart = tm.time()
        r.integrate(r.t + dt[index])
        res = r.y
        u[index,:] = res[:]
        tend = tm.time()
        eta = round( (tend - tstart)*(T[nT] - T[index])/dt[index] )
        print('Completed: ' + str(int(float(index)/nT*100)).rjust(3, ' ') + 
              '%, ETA: ' + str(datetime.timedelta(seconds = eta)).rjust(8, ' '))
        index += 1
    print('Computations complete\n')

# def findu0 (u0):
#        return (spline_kappa(u0)*D*(ubulk - u0)/dx + 
#                E*exp(-uevap/u0) + F*u0**4 - ext_heat_flux())
# h_s = zeros(Ntime)
# for i in range(Ntime): 
#     ubulk = u[i,2]
#     t_loc = T[i]
#     u[i,0] = u[i,1] = opt.newton_krylov(findu0, u[i,2], method='lgmres', verbose=0) 
#     h_s[i] = nu*n**(-1./3.)*exp(-uevap/u[i,0])
# h_s = integrate.cumtrapz(h_s, T*tau, initial = 0)
# Renormalize solutions for the trapped hydrogen to the value Y
if SaveData:
    u *= Tt
#if SaveAnlt:
#    uAnalytic *= Tt
    
    fig, ax = plt.subplots()
    fig.set_facecolor('w')
    plt.title('$Temperature profile in the sample$', fontsize = 14)
    if SaveData:
        line_ex, = ax.plot(x_ext, u[0, 1:(nnodes + 3)], linewidth = 1.5, label = '$u(x, t)$')
    #if SaveAnlt:
    #    ax.plot(x_ext, CprofAnalytic, linewidth = 1.5, label = '$analytic\ solution$')
    ax.legend(loc = 'lower left')
    plot_text = ax.text(0.05, 0.35, '', transform=ax.transAxes, fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize = 12)
    ax.set_xlabel('$x/L$', fontsize = 12)
    ax.set_ylabel(r'$u, \mathrm{K}$', fontsize = 12)
    ax.set_xlim([0., 1.])
    #ax.set_ylim([0., 1.01])
    
    line_ex.set_visible(False)
    plot_text.set_visible(False)

    # Init only required for blitting to give a clean slate.
    def init():
        global line_ex, plot_text
        plot_text.set_text('')
        line_ex.set_ydata(ma.array(x_ext, mask=True))
        
        return line_ex, plot_text,
        
    def animate(i):
        global line_ex, plot_text
        if i==1:
            plot_text.set_visible(True)
            line_ex.set_visible(True)
        plot_text.set_text( '$t = {%3.0f} s$' % (tau*T[i]) )
        if SaveData:
            line_ex.set_ydata(u[i, 1:(nnodes + 3)])  # update the data

        return line_ex, plot_text

    ani = animation.FuncAnimation(fig, animate, arange(1, len(T)), init_func=init,
                                  interval=1, blit=True)
    if SaveAnim:
        ani.save('TempProf.mp4')
        
    plt.show()
print('Total time computed ' + str(T*tau/1.0e6) + ' {10^6}s')
t *= tau # conversion from simulation to real time
if SaveData:
    # save data for the temperature profile
    filename = 'temp_run'
    f = open(filename, mode='w')
    string = 'x'
    for i in range(0, nT):
        string += ' ' + str(t[i])
    string += '\n'
    f.write(string)
    
    for i in range(0, nnodes + 2):
        string = str(x_ext[i])
        for j in range(1, nT + 1):
            string += (' ' + str(u[j, i + 1]))
        string += '\n'
        f.write(string)
    f.close()
tend = tm.time()
print('Done solving the problem in %f\n'%(tend-tstart))
#if SaveAnlt:
#    # save data for interstitial tritium
#    filename = 'zero_conc_anlt_sol_L'+str(L)+'cm_'+'Tin'+str(Tl)+'K' + '_Tout'+str(Tr)+'K' + '_D0_' + str(D0) + 'cm2s-1' + '_tau' + str(tau) + 's' + '_ftr' + str(ftr) + 's-1' + '_Ed'+str(Ed)+'eV' + '_Edtr'+str(Edt)+'eV' + '_Y' + str(Y/C) + '_u0_' + str(c0/C) + '_C' + str(C) + 'cm-3' + '_INTERST'
#    f = open(filename, mode='w')
#    string = 'x u\n'
#    f.write(string)
    
#    for i in range(0, nnodes + 2):
#        string = str(x_ext[i]) + ' ' + str(CprofAnalytic[i]) + '\n'
#        f.write(string)
#    f.close()

#    # save data for trapped tritium
#    filename = 'zero_conc_anlt_sol_L'+str(L)+'cm_'+'Tin'+str(Tl)+'K' + '_Tout'+str(Tr)+'K' + '_D0_' + str(D0) + 'cm2s-1' + '_tau' + str(tau) + 's' + '_ftr' + str(ftr) + 's-1' + '_Ed'+str(Ed)+'eV' + '_Edtr'+str(Edt)+'eV' + '_Y' + str(Y/C) + '_u0_' + str(c0/C) + '_C' + str(C) + 'cm-3' + '_TRAPPED'
#    f = open(filename, mode='w')
#    string = 'x y\n'
#    f.write(string)
    
#    for i in range(0, nnodes):
#        string = str(x[i]) + ' ' + str(YprofAnalytic[i]) + '\n'
#        f.write(string)
#    f.close()
plt.figure(2)
plt.title('Пространственный профиль температуры')
plt.plot(u[-1,2:], label="t1 = 0 нс")
# plt.plot(u[150,:], label="t1 = 10 нс")
# plt.plot(u[300,:], label="t2 = 20 нс")
# plt.plot(u[1499,:], label="t3 = 50 нс")
x_test = linspace(0, 1, ntotal-4)
T_test = Tt + Q0*L/140 * (1 - x_test)
nodes =  linspace(0, len(T_test), len(T_test))
plt.plot(nodes, T_test, label='Test line')
plt.legend()

# plt.figure(3)
# plt.title('Временной профиль температуры')
# plt.plot(u[:,0], label="x = 0 мкм")
# plt.plot(u[:,35], label="x = 0.05 мкм")
# #plt.plot(u[:,70], label="x = 0.1 мкм")
# plt.legend()
# plt.figure(4)
# plt.title('Глубина испарения от времени')
# plt.plot(T*tau, h_s)
# plt.xlabel("$t$",fontsize = 14)
# plt.ylabel("$h_s$", fontsize = 14)
