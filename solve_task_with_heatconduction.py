import numpy as np
from math import fabs
import warnings
import time as tm
import datetime
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.special import erfc, erf
from scipy.integrate import solve_bvp, quad, ode
from scipy.interpolate import make_interp_spline
from math import floor, log10, ceil
from scipy.optimize import fsolve
from scipy.differentiate import derivative

# Ignore some warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')

# Plots settings
plt.rcParams.update({"font.size": 13})
params = {"text.latex.preamble": r"\usepackage{icomma}"}
# locale.setlocale(locale.LC_NUMERIC, "ru_RU")
plt.rcParams["axes.formatter.use_locale"] = True
plt.rcParams.update(params)
plt.rcParams["figure.autolayout"] = True

# Plot hints styles
props = dict(boxstyle="square", facecolor="white")
props_note = dict(boxstyle="ellipse", facecolor="white")

# Physical quantities in cgs

eV_to_J     = 1.6e-19
eV_to_K     = 11604
J_to_eV     = 6.242e18
eV_to_erg   = 1.602e-12
erg_to_eV   = 6.242e11
K_to_erg    = 1.381e-16
erg_to_K    = 7.243e15
Cl_to_cgs   = 3.0e9

me  = 9.11e-28 # electron mass
mi  = 1.67e-24 # proton mass
e   = 1.6e-19 * Cl_to_cgs # electric charge of electron
a   = 120.4 / K_to_erg**2 * Cl_to_cgs # Richardson's constant
sigma   = 4.54 * eV_to_erg # Stefan-Boltzmann constant

# Tungsten properties at T = 1000 K
phiout  = 4.54 * eV_to_erg # Pa6oma
rho = 19.07 # Density
Cp  = 147.7 * 1.0e4 # Isobaric heat capacity
kappa = 116.6 * 1.0e5  # Thermal diffusivity

# Plasma properties
nse = 1.0e13 # plasma density
Te  = 200.0 * eV_to_erg # electron temperature
# deltae  = 0.0 # SEE coefficient

# Calculated constants
cs = np.sqrt(Te / mi) # Speed velocity in plasma
vth = np.sqrt(8 * mi / np.pi / me) # Thermal velocity of plasma electrons(dimensionless)
r_debye = np.sqrt(Te / (4 * np.pi * nse * e**2)) # Debye radius
omega_p = np.sqrt(4 * np.pi * nse * e**2 / me) # Plasma frequency of electrons
tau_p = 2 * np.pi / omega_p # Plasma period

# Conversion functions
TK = lambda T : T * Te * erg_to_K
TD = lambda T : T / Te / erg_to_K

# Problem parameters
L = 0.1 / r_debye # Thickness of a sample, sm
T0 = TD(1000) # Thermostat and initial temperature
C = tau_p * kappa / (r_debye**2 * rho * Cp)
# D = -T0 * kappa / (nse * cs * Te * L)

# Useful calculating functions
def nte_w_func(derw, Tw):
    dA = (
        - np.sign(derw)
        * np.sqrt(abs(derw))
        * np.sqrt(e * Te / r_debye)
        * 300
    )
    return (
        a
        * (Tw * Te)**2.0
        * np.exp(-(phiout / Te + dA) / Tw)
        / (0.25 * e * nse * np.sqrt(Tw * Te * 8.0 / np.pi / me))
    )

upsilon_0_func = lambda phi_se : np.sqrt(-2.0 * phi_se)

def erfcxexp_limit_resolve(x):
    if x > 100:
        return (
            np.power(x, -0.5)
            - np.power(x, -1.5) / 2.0
            + np.power(x, -2.5) * 3.0/4.0
            - np.power(x, -3.5) * 15.0/8.0
            + np.power(x, -4.5) * 105.0/16.0
        ) / np.sqrt(np.pi)
    else:
        return erfc(np.sqrt(x)) * np.exp(x)

# Solving system for transition temperature

def Poisson_integrated_classic_trans(phi, y, args):
    V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return (
        upsilon_0 * np.sqrt(upsilon_0**2 - 2 * (phi - phi_se))
        + ne_se * np.exp(phi - phi_se)
        + nte_w
        * Tw
        * (
            erfcxexp_limit_resolve((phi - (V_f + phi_se)) / Tw)
            + 2 / np.sqrt(np.pi) * np.sqrt((phi - (phi_se + V_f)) / Tw)
        )
    )

def Poisson_classic_trans(y, args):
    V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return -2.0 * (
        Poisson_integrated_classic_trans(phi_se + V_f, y, args)
        - Poisson_integrated_classic_trans(phi_se, y, args)
    )

def quasineutrality_trans(y, args):
    V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return 1 - nte_w * erfcxexp_limit_resolve(-V_f / Tw) - ne_se


def j_wall_trans(y, args):
    V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return V_f - np.log(
        4 * upsilon_0 / (ne_se * np.sqrt(8 * mi / (np.pi * me)))
        + nte_w / ne_se * np.sqrt(Tw)
    )

def sys_trans(y, *args):
    V_f, ne_se = y
    Tw, phi_se, upsilon_0 = args
    args1 = [Tw, nte_w_func(0, Tw), upsilon_0, phi_se]
    return [
        j_wall_trans(y, args1),
        quasineutrality_trans(y, args1)
    ]
 
sol_trans_init_guesses = [
    [-2.7, 0.995],
    [-1.1866, 0.892],
    [-1.1866, 0.892],
    [-1.15, 0.9],
    [-1.25, 0.86]
]

sw_trans = False
phi_se = -0.5
upsilon_0 = upsilon_0_func(phi_se)
args_trans = (T0, phi_se, upsilon_0)
for j in range(len(sol_trans_init_guesses)):
    sol_trans = fsolve(sys_trans, sol_trans_init_guesses[j], args=args_trans)
    if (np.isclose(sys_trans(sol_trans, *args_trans), np.zeros(len(sol_trans))) == np.ones(len(sol_trans), dtype = bool)).all():
        sw_trans = True
        break
    else:
        print(sys_trans(sol_trans, *args_trans))
if sw_trans == False:
    raise NameError('Appropriate sol_trans not found. Add more initial guesses')
else:
    V_f_trans, ne_se_trans = sol_trans
    print(V_f_trans, ne_se_trans)

j_func = lambda V, ne_se : upsilon_0 - vth / 4.0 * ne_se * np.exp(V)

V_func = lambda j : np.log((j_func - upsilon_0) * 4.0 / vth / ne_se)

def q_ion_func(y, phi_se, Tw, Tw_trans, is_MW = False):
    derw, V_f, ne_se, V_vc = y
    if (phi_se > 0) : return 0.0
    vteth = np.sqrt(8 * Tw * mi / (np.pi * me))
    q_net = upsilon_0 * (upsilon_0**2 / 2 - V_vc)
    if (is_MW == True) :
        q_net = q_net * nse * cs * Te * 1.0e6 * 1.0e-2 * 1.0e-7 * 1.0e-6
    return q_net

def q_e_func(y, phi_se, Tw, Tw_trans, is_MW = False):
    derw, V_f, ne_se, V_vc = y
    vteth = np.sqrt(8 * Tw * mi / (np.pi * me))
    if (Tw <= Tw_trans):
        q_net = (
            0.25 * ne_se * vth * 2 * np.exp(V_f)
        )
    else:
        q_net = (
            0.25 * ne_se * vth * 2 * np.exp(V_vc)
        )
    if (is_MW == True) :
        q_net = q_net * nse * cs * Te * 1.0e6 * 1.0e-2 * 1.0e-7 * 1.0e-6
    return q_net

def q_func(y, phi_se, Tw, Tw_trans, is_MW = False): 
    return (
            q_e_func(y, phi_se, Tw, Tw_trans, is_MW)
            + q_ion_func(y, phi_se, Tw, Tw_trans, is_MW)
        )

# System for classic regime
def j_wall_classic(y, args):
    derw, V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return V_f - np.log(
        4 * upsilon_0 / (ne_se * np.sqrt(8 * mi / (np.pi * me)))
        + nte_w / ne_se * np.sqrt(Tw)
    )

def Bohm_criterion_classic(y, args):
    derw, V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return phi_se + 0.5 * Tw / (
        ne_se * Tw
        + nte_w
        * (
            erfcxexp_limit_resolve(-V_f / Tw)
            - 1 / (np.sqrt(np.pi) * np.sqrt(-V_f / Tw))
        )
    )


def quasineutrality_classic(y, args):
    derw, V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return 1 - nte_w * erfcxexp_limit_resolve(-V_f / Tw) - ne_se


def Poisson_integrated_classic(phi, y, args):
    derw, V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return (
        upsilon_0**2 * np.sqrt(1 - 2 * (phi - phi_se) / upsilon_0**2)
        + ne_se * np.exp(phi - phi_se)
        + nte_w
        * Tw
        * (
            erfcxexp_limit_resolve((phi - (V_f + phi_se)) / Tw)
            + 2 / np.sqrt(np.pi) * np.sqrt((phi - (phi_se + V_f)) / Tw)
        )
    )


def Poisson_classic(y, args):
    derw, V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    return derw**2 - 2.0 * (
        Poisson_integrated_classic(phi_se + V_f, y, args)
        - Poisson_integrated_classic(phi_se, y, args)
    )


def sys_classic(y, *args):
    derw, V_f, ne_se = y
    phi_se, Tw = args
    args1 = [Tw, nte_w_func(derw, Tw), upsilon_0_func(phi_se), phi_se]
    return [
        # Bohm_criterion_classic(y, args1),
        quasineutrality_classic(y, args1),
        j_wall_classic(y, args1),
        Poisson_classic(y, args1)
    ]

# System for SCL regime
def jwall_SCL(y, args):
    derw, V_f, ne_se, V_vc = y
    Tw, nte_w, upsilon_0, phi_se = args
    dip = (V_f - V_vc) / Tw
    return (
        upsilon_0
        - 0.25 * ne_se * np.sqrt(8 * mi / (np.pi * me)) * np.exp(V_vc)
        + 0.25 * nte_w * np.sqrt(Tw * 8 * mi / (np.pi * me)) * np.exp(-dip)
    )  #


def quasineutrality_SCL(y, args):
    derw, V_f, ne_se, V_vc = y
    Tw, nte_w, upsilon_0, phi_se = args
    dip = (V_f - V_vc) / Tw
    return (
        1 - ne_se - nte_w * erfcxexp_limit_resolve(-V_vc / Tw) * np.exp(-dip)
    )  #


def Poisson_integrated_SCL_beta(phi, y, args):
    derw, V_f, ne_se, V_vc = y
    Tw, nte_w, upsilon_0, phi_se = args
    dip = (V_f - V_vc) / Tw
    return (
        upsilon_0**2 * np.sqrt(1 - 2 * (phi - phi_se) / upsilon_0**2)
        + ne_se * np.exp(phi - phi_se)
        + nte_w
        * Tw
        * np.exp(-dip)
        * (
            erfcxexp_limit_resolve((phi - (V_vc + phi_se)) / Tw)
            + 2.0
            / np.sqrt(np.pi)
            * np.sqrt((phi - (V_vc + phi_se)) / Tw)
        )
    )  # ~~~


def Poisson_SCL_beta(y, args):
    derw, V_f, ne_se, V_vc = y
    Tw, nte_w, upsilon_0, phi_se = args
    return -2 * (
        Poisson_integrated_SCL_beta(V_vc + phi_se, y, args)
        - Poisson_integrated_SCL_beta(phi_se, y, args)
    )

def Poisson_integrated_SCL_alpha(phi, y, args):
    derw, V_f, ne_se, V_vc = y
    Tw, nte_w, upsilon_0, phi_se = args
    dip = (V_f - V_vc) / Tw
    res = (
        upsilon_0**2 * np.sqrt(1 - 2 * (phi - phi_se) / upsilon_0**2)
        + ne_se
        * (
            # np.exp(V_vc) * erfcxexp_limit_resolve(phi - (V_vc + phi_se))
            np.exp(phi - phi_se) * erfc(np.sqrt(phi - (V_vc + phi_se)))
            + 2.0
            / np.sqrt(np.pi)
            * (np.sqrt(phi - (V_vc + phi_se)))
            * np.exp(V_vc)
        )
        + nte_w
        * Tw
        * (
            np.exp((phi - (V_f + phi_se))/Tw) * (1 + erf(np.sqrt((phi - (V_vc + phi_se)) / Tw)))
            - 2.0
            / np.sqrt(np.pi)
            * (np.sqrt((phi - (V_vc + phi_se))/Tw))
            * np.exp((V_vc - V_f)/Tw)
        )
    )
    return res

def Poisson_SCL_alpha(y, args):
    derw, V_f, ne_se, V_vc = y
    Tw, nte_w, upsilon_0, phi_se = args
    return derw**2 - 2 * (
        Poisson_integrated_SCL_alpha(V_f + phi_se, y, args)
        - Poisson_integrated_SCL_alpha(V_vc + phi_se, y, args)
    )


def sys_SCL(y, *args):
    derw, V_f, ne_se, V_vc = y
    phi_se, Tw = args
    nte_w = nte_w_func(derw, Tw)
    upsilon_0 = upsilon_0_func(phi_se)
    args1 = [Tw, nte_w, upsilon_0, phi_se]
    return [
        quasineutrality_SCL(y, args1),
        jwall_SCL(y, args1),
        Poisson_SCL_alpha(y, args1),
        Poisson_SCL_beta(y, args1),
    ]

# Heat conduction system
def div_dif_1(u, x):
    result = 0.0
    for i in range(len(u)):
        div = 1.
        for j in range(len(u)):
            if (j != i and u[j] != 0.0):
                div *= x[i] - x[j]
        result += u[i] / div
    return result

def u_border(x_point, u, x):
    '''Lagrange polynomial'''
    sh = np.shape(u)
    result = 0.0
    for i in range(sh[0]):
        temp = div_dif_1(u[:i+1], x[:i + 1])
        if i > 0:
            for j in range(i):
                temp *= x_point - x[j]
        result += temp
    return result

def solve_debye(V_f, phi_se, Tw, Tw_trans):
    result = np.ndarray(4, dtype = float)
    args = (phi_se, Tw)
    if Tw < Tw_trans:
        sol = fsolve(sys_classic, [1.0, -3.0, 0.95], args=args)
        result = [sol[0], sol[1], sol[2], sol[1]]
    else:
        j = 0
        sol = np.ones((4, 1))
        while (
            not (
                np.less(sys_SCL(sol, *args), np.full(len(sol), 1.0e-7))
                == np.ones(len(sol), dtype=bool)
            ).all()
        ) and j < 30:
            sol = fsolve(
                sys_SCL,
                [
                    0 - 0.1 * j,  # 0.001
                    V_f_trans + 0.00001 * j,
                    ne_se_trans,
                    V_f_trans,
                ],
                args=args,
            )
            j += 1
        result = sol
    return result

def rhs(t, u):
    dudt = np.ndarray((ntotal))
    # define udot in the simulation domain

    # dudt[1.5] = 0 thermostat
    u[1] = u_border(-0.5, 
                    [T0, u[2], u[3], u[4], u[5]],
                    [0.0, 0.5, 1.5, 2.5, 3.5]
    ) 
    u[0] = u_border(-1.5,
                    [u[1], T0, u[2], u[3], u[4]], 
                    [-0.5, 0.0, 0.5, 1.5, 2.5]
    ) 
    u_right = u_border(0.0, 
                       [u[-7], u[-6], u[-5], u[-4], u[-3]], 
                       [-4.5, -3.5, -2.5, -1.5, -0.5]
    ) 
    debye_solution = solve_debye(V_f, phi_se, Tw, Tw_trans) 
    grad_dx_right = (q_func(debye_solution, phi_se, Tw, Tw_trans)) / (kappa / (nse * cs * r_debye)) * dx / dt
    print(grad_dx_right)
    u[-2] = (
        -24.0 * grad_dx_right
        + 17.0 * u[-3]
        + 9.0 * u[-4]
        - 5.0 * u[-5]
        + 1.0 * u[-6]
            ) / 22.0
    u[-1] = (
        -24.0 * grad_dx_right
        + 27.0 * u[-2]
        - 27.0 * u[-3]
        + 1.0 * u[-4]
            )
    for i in range(2, ntotal - 2):
        d2udx2 = (-u[i-2] + 16.*u[i-1] - 30.*u[i] + 16.*u[i+1] - u[i+2]) / 12
        dudt[i] = C * d2udx2
    u_right = u_border(0.0, [u[-7], u[-6], u[-5], u[-4], u[-3]], [-4.5, -3.5, -2.5, -1.5, -0.5]) 
    debye_solution, Tw_trans = solve_debye(t, u_right) 

    dudt[0] = dudt[1] = dudt[-2] = dudt[-1] = 0.0 # we don't care about evolution of these points
    exit()
    return dudt

periods_to_plot = 10
t_net_max = (
        periods_to_plot # in periods of omega_p
        * 2 * np.pi / omega_p
        )
print('t_net_max : %.2E s' % Decimal(t_net_max * 2 * np.pi / omega_p))
t_net_steps = periods_to_plot * 10
print('t_net_steps : %d' % t_net_steps)
t_net = np.linspace(0, t_net_max, t_net_steps)
dt = t_net[1] - t_net[0]

x_net_main_steps = 21
ntotal = x_net_main_steps + 4
dx = L / x_net_main_steps

x_net = np.zeros(ntotal)
for i in range(ntotal):
    x_net[i] = -2 * dx + dx * i

u0 = np.full((ntotal), T0)
result = np.ndarray((t_net_steps, ntotal))
result[0, :] = u0[:]

r = ode(rhs).set_integrator('vode', method='bdf', nsteps=1e7)
r.set_initial_value(u0, 0)
index = 1
tstep = 3#t_net_steps - 1
while r.successful() and index < t_net_steps and index <= tstep:
    # tstart = tm.time()
    r.integrate(r.t + dt)
    result[index, :] = r.y
    index += 1

print("index : ", index)
fig = plt.figure(figsize=(8, 6), dpi=300)

plt.plot(t_net[:tstep], phi_se_func(base, amp, omega, t_net[:tstep]))
plt.show()

plt.plot(x_net[2:-2] * r_debye, TK(result[tstep, 2:-2]))
plt.grid()
plt.ylim(TK(T0 * 0.9), TK(2 * T0))
plt.xlim(x_net[2] * r_debye, x_net[-2] * r_debye)
plt.show()
