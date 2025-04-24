import numpy as np
from math import fabs
import warnings
import time as tm
import datetime
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.special import erfc, erf
from scipy.integrate import solve_bvp, quad, ode
from scipy.interpolate import make_interp_spline, interp1d
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
Cp  = 147.7 * 1.0e4 / K_to_erg # Isobaric heat capacity
kappa = 116.6 * 1.0e5 / K_to_erg  # Thermal diffusivity
# Cp  = 180 * 1.0e4 # Isobaric heat capacity
# kappa = 100 * 1.0e5  # Thermal diffusivity



# Plasma properties
nse = 1.0e13 # plasma density
Te  = 200.0 * eV_to_erg # electron temperature
# deltae  = 0.0 # SEE coefficient

# Calculated constants
cs = np.sqrt(Te / mi) # Speed velocity in plasma
vth = np.sqrt(8 * mi / np.pi / me) # Thermal velocity of plasma electrons(unitless)
r_debye = np.sqrt(Te / (4 * np.pi * nse * e**2)) # Debye radius
omega_p = np.sqrt(4 * np.pi * nse * e**2 / me) # Plasma frequency of electrons
tau_p = 2 * np.pi / omega_p # Plasma period

# Conversion functions
TK = lambda T : T * Te * erg_to_K
TD = lambda T : T / Te / erg_to_K

spline_T_net = TD(np.array([1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000,
               3200, 3400, 3600, 3695]))
kappa_values = np.array([116.6, 113.5, 111.2, 110.1, 109.0, 108.3, 107.2, 106.8, 107.4,
                108.8, 107.5, 107.2, 103.6, 101.1, 99.0]) * 1.0e5 / K_to_erg
kappa_spline = make_smoothing_spline(spline_T_net, kappa_values, None)
Cp_values = np.array([147.7, 152.2, 157.1, 162.5, 166.0, 174.7, 181.6, 189.1,
                      197.6, 207.0, 217.8, 230.3, 244.8, 261.7, 270.7]) * 1.0e4 / K_to_erg
Cp_spline = make_smoothing_spline(spline_T_net, Cp_values, None)
rho_values = np.array([19.07, 19.01, 18.95, 18.89, 18.82, 18.72, 18.62, 18.52,
                       18.42, 18.32, 18.22, 18.12, 18.0, 17.8, 17.5])
rho_spline = make_smoothing_spline(spline_T_net, rho_values, None)
C_values = np.zeros(np.shape(kappa_values))

for i in range(np.shape(C_values)[0]):
    C_values[i] = kappa_values[i] / (rho_values[i] * Cp_values[i])
C_spline = make_smoothing_spline(spline_T_net, C_values, None)

# Problem parameters
L = 1.2e-1 # Thickness of a sample, sm
T0 = TD(1000) # Thermostat and initial temperature
C = kappa / (rho * Cp)
# D = -T0 * kappa / (nse * cs * Te * L)

t_net_max = 0.35e-2
print('t_net_max : %.2E s' % Decimal(t_net_max))
t_net_steps = 35001
# exit()
print('t_net_steps : %d' % t_net_steps)
t_net = np.linspace(0, t_net_max, t_net_steps)
dt = t_net[1] - t_net[0]

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
    Tw, V_f, ne_se = y
    nte_w, upsilon_0, phi_se = args
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
    Tw, V_f, ne_se = y
    nte_w, upsilon_0, phi_se = args
    return -2.0 * (
        Poisson_integrated_classic_trans(phi_se + V_f, y, args)
        - Poisson_integrated_classic_trans(phi_se, y, args)
    )


def quasineutrality_trans(y, args):
    Tw, V_f, ne_se = y
    nte_w, upsilon_0, phi_se = args
    return 1 - nte_w * erfcxexp_limit_resolve(-V_f / Tw) - ne_se


def j_wall_trans(y, args):
    Tw, V_f, ne_se = y
    nte_w, upsilon_0, phi_se = args
    vte = np.sqrt(8 * Tw * mi / np.pi / me)
    return (
            upsilon_0
            + 0.25 * vte * nte_w
            - 0.25 * vth * ne_se * np.exp(V_f)
            )
    # return V_f - np.log(
    #     4 * upsilon_0 / (ne_se * np.sqrt(8 * mi / (np.pi * me)))
    #     + nte_w / ne_se * np.sqrt(Tw)
    # )



def sys_trans(y, *args):
    Tw, V_f, ne_se = y
    phi_se, = args
    args1 = [nte_w_func(0, Tw), upsilon_0_func(phi_se), phi_se]
    return [
        # Bohm_criterion_trans(y, args1),
        Poisson_classic_trans(y, args1),
        j_wall_trans(y, args1),
        quasineutrality_trans(y, args1),
    ]
 
sol_trans_init_guesses = [
    [TD(3165), -1.1866, 0.892],
    [TD(2900), -1.1866, 0.892],
    [TD(2750), -1.15, 0.9],
    [TD(3200), -1.25, 0.86]
]

phi_se = -0.5
upsilon_0 = upsilon_0_func(phi_se)

properties_trans = np.zeros(5)

sw_trans = False
args_trans = (phi_se, )
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
    properties_trans = 0.0, sol_trans[1], sol_trans[2], sol_trans[1], sol_trans[0]


def j_func(y, Tw, Tw_trans):
    derw, V_f, ne_se, V_vc = y
    vte = np.sqrt(8 * Tw * mi / np.pi / me)
    if (Tw > Tw_trans):
        return (
            1.0
            - 0.25 * vth * ne_se * np.exp(V_vc)
            + 0.25 * vte * nte_w_func(derw, Tw) * np.exp((V_vc - V_f)/Tw)
        )
    else:
        return (
            1.0
            - 0.25 * vth * ne_se * np.exp(V_f)
            + 0.25 * vte * nte_w_func(derw, Tw)
        )

# V_func = lambda j : np.log((j_func - upsilon_0) * 4.0 / vth / ne_se)

# heat fluxes

def q_ion_func(y, is_MW = False):
    derw, V_f, ne_se, V_vc, Tw, Tw_trans = y
    if (phi_se > 0) : return 0.0
    q_net = upsilon_0 * (upsilon_0**2 / 2 - V_vc)
    if (is_MW == True) :
        q_net = q_net * nse * cs * Te * 1.0e6 * 1.0e-2 * 1.0e-7 * 1.0e-6
    return q_net

def q_e_func(y, is_MW = False):
    derw, V_f, ne_se, V_vc, Tw, Tw_trans = y
    if (Tw < Tw_trans):
        q_net = (
            0.25 * ne_se * vth * 2 * np.exp(V_f)
        )
        print("Classic!")
    else:
        q_net = (
            0.25 * ne_se * vth * 2 * np.exp(V_vc)
        )
        print("SCL!")
    if (is_MW == True) :
        q_net = q_net * nse * cs * Te * 1.0e6 * 1.0e-2 * 1.0e-7 * 1.0e-6
    return q_net

def q_func(y, is_MW = False): 
    return (
            q_e_func(y, is_MW)
            + q_ion_func(y, is_MW)
        )

# System for classic regime
def j_wall_classic(y, args):
    derw, V_f, ne_se = y
    Tw, nte_w, upsilon_0, phi_se = args
    vte = np.sqrt(8 * Tw * mi / np.pi / me)
    return (
            upsilon_0
            + 0.25 * vte * nte_w
            - 0.25 * vth * ne_se * np.exp(V_f)
            )
    # return V_f - np.log(
    #     4 * upsilon_0 / (ne_se * np.sqrt(8 * mi / (np.pi * me)))
    #     + nte_w / ne_se * np.sqrt(Tw)
    # )

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


def solve_debye(Tw):
    result = np.ndarray(4, dtype = float)
    derw_trans, V_f_trans, ne_se_trans, V_vc_trans, Tw_trans = properties_trans;
    args = (phi_se, Tw)
    if Tw < Tw_trans:
        line = 2 - Tw / T0 
        # sol = fsolve(sys_classic, [1.0 * line, -2.5 + 1.0  * ((Tw - T0) / (Tw_trans - T0)) , 1.0 - 0.1 * line], args=args)
        sol = fsolve(sys_classic, [1.0, -2.5, 1.0], args=args)
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
                    0.0 - 0.1 * j,
                    V_f_trans + 0.0001 * j,
                    ne_se_trans,
                    V_vc_trans,
                ],
                args=args,
            )
            j += 1
        result = sol
    return result

derw, V_f, ne_se = fsolve(sys_classic, [1.0, -2.7, 0.995], args = (phi_se, T0))
V_vc = V_f
Tw = T0

x_net_main_steps = 101
ntotal = x_net_main_steps + 4
dx = L / x_net_main_steps

dx12   = 12.*dx       # extra metric factor
d2x12  = 12.*dx**2    # extra metric factor

x_net = np.zeros(ntotal)
for i in range(ntotal):
    x_net[i] = -2 * dx + dx * i

def rhs(t, u):
    dudt = np.full_like(u, 0.0)
    # dudt[1.5] = 0 thermostat
    u[1] = u_border(-0.5, 
                    [T0, u[2], u[3], u[4], u[5]],
                    [0.0, 0.5, 1.5, 2.5, 3.5]
    )
    u[0] = u_border(-1.5,
                    [u[1], T0, u[2], u[3], u[4]], 
                    [-0.5, 0.0, 0.5, 1.5, 2.5]
    )
    derw_trans, V_f_trans, ne_se_trans, V_vc_trans, Tw_trans = properties_trans;
    Tw = u_border(0.0, # calculate for the next step
           [u[-7], u[-6], u[-5], u[-4], u[-3]], 
           [-4.5, -3.5, -2.5, -1.5, -0.5]
    ) 
    grad_dx_right = q_func([derw, V_f, ne_se, V_vc, Tw, Tw_trans]) * ((nse * cs)
                                                                      /
                                                                      kappa_spline(Tw)) * dx
    # grad_dx_right = (1.0e12 / (kappa * Te)) * dx
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
    print(TK(u[-3:]))
    for i in range(2, ntotal - 2):
        d2udx2 = (-u[i-2] + 16.*u[i-1] - 30.*u[i] + 16.*u[i+1] - u[i+2]) / d2x12
        dudt[i] = C_spline(u[i]) * d2udx2
    print("dudt * dt : ", dudt[-5:] * dt)
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    return dudt


# f1d2x12  =  C * 1./d2x12 # extra metric factor
# f16d2x12 =  C * 16./d2x12 # extra metric factor
# f30d2x12 = C * -30./d2x12 # extra metric factor
#
# def jac(t, u):
#     dFdu = np.zeros((ntotal, ntotal))
#     for i in range(2, ntotal - 2):
#         dFdu[i, i-2] = -f1d2x12
#         dFdu[i, i-1] = f16d2x12
#         dFdu[i, i]   = f30d2x12
#         dFdu[i, i+1] = f16d2x12
#         dFdu[i, i+2] = -f1d2x12
#     return dFdu  

u0 = np.full((ntotal), T0)
result = np.ndarray((t_net_steps, ntotal))
result[0, :] = u0[:]

r = ode(rhs).set_integrator('vode', method='bdf', nsteps=1e6)
# r = ode(rhs, jac).set_integrator('vode', method='bdf', nsteps=1e6)
r.set_initial_value(u0, 0)
index = 1
tstep = t_net_steps - 1
V_vc_log = np.zeros(tstep)
q_log = np.zeros(tstep)
j_log = np.zeros(tstep)

omega = floor(1.0e5)
phi_func = lambda t, V_f : -0.6 * V_f * np.sin(2*np.pi * t * omega) # approximately 3 periods in elm time
result_w = np.zeros(t_net_steps)
result_w[0] = T0
while r.successful() and index < t_net_steps and index <= tstep:
    r.integrate(r.t + dt)
    result[index, :] = r.y

    Tw = u_border(0.0, # calculate for the next step
           [result[index, -7], result[index, -6], result[index, -5], result[index, -4], result[index, -3]], 
           [-4.5, -3.5, -2.5, -1.5, -0.5]
    ) 
    result_w[index] = Tw

    derw, V_f, ne_se, V_vc = solve_debye(Tw)
    V_vc += phi_func(r.t, V_f)
    V_f += phi_func(r.t, V_f)
    V_vc_log[index - 1] = V_vc

    derw_trans, V_f_trans, ne_se_trans, V_vc_trans, Tw_trans = properties_trans;
    q_log[index - 1] = q_func([derw, V_f, ne_se, V_vc, Tw, Tw_trans])
    j_log[index - 1] = j_func([derw, V_f, ne_se, V_vc], Tw, Tw_trans)
    print(TK(Tw))
    print("derw : ", derw)
    print("V_f : ", V_f)
    print("ne_se : ", ne_se)
    print("V_vc : ", V_vc)
    print("-------------step : ", index, " --------------")
    index += 1
print("index : ", index)
print("Tw : ", TK(result[index - 1, -3]))
fig = plt.figure(figsize=(8, 6), dpi=300)

#### non-oscilating

derw, V_f, ne_se = fsolve(sys_classic, [1.0, -2.7, 0.995], args = (phi_se, T0))
V_vc = V_f
Tw = T0

r = ode(rhs).set_integrator('vode', method='bdf', nsteps=1e6)
# r = ode(rhs, jac).set_integrator('vode', method='bdf', nsteps=1e6)
r.set_initial_value(u0, 0)
index = 1
tstep = t_net_steps - 1
V_vc_0_log = np.zeros(tstep)
q_0_log = np.zeros(tstep)
j_0_log = np.zeros(tstep)

result_0 = np.ndarray((t_net_steps, ntotal))
result_0[0, :] = u0[:]
result_w_0 = np.zeros(t_net_steps)
result_w_0[0] = T0
while r.successful() and index < t_net_steps and index <= tstep:
    r.integrate(r.t + dt)
    result_0[index, :] = r.y

    Tw = u_border(0.0, # calculate for the next step
           [result_0[index, -7], result_0[index, -6], result_0[index, -5], result_0[index, -4], result_0[index, -3]], 
           [-4.5, -3.5, -2.5, -1.5, -0.5]
    ) 
    result_w_0[index] = Tw

    derw, V_f, ne_se, V_vc = solve_debye(Tw)
    V_vc_0_log[index - 1] = V_vc

    derw_trans, V_f_trans, ne_se_trans, V_vc_trans, Tw_trans = properties_trans;
    q_0_log[index - 1] = q_func([derw, V_f, ne_se, V_vc, Tw, Tw_trans])
    j_0_log[index - 1] = j_func([derw, V_f, ne_se, V_vc], Tw, Tw_trans)
    # print(TK(Tw))
    # print("derw : ", derw)
    # print("V_f : ", V_f)
    # print("ne_se : ", ne_se)
    # print("V_vc : ", V_vc)
    # print("-------------step : ", index, " --------------")
    index += 1

msize = 1
mstyle = "x"


plt.plot(x_net[2:-2], TK(result[tstep, 2:-2]))
plt.plot(x_net[2:-2], TK(result_0[tstep, 2:-2]))
#plt.grid()
plt.ylim(TK(T0 * 0.9), TK(max(result[tstep, 2:-2])))
plt.xlim(x_net[2], x_net[-3])
plt.xlabel(r"$x$, cm")
plt.ylabel(r"$T_s$, K")
plt.savefig("temperature_distribution_plot.png")
plt.clf()
# plt.show()
plt.xlim(0, t_net_max)
plt.scatter(t_net[:tstep + 1], TK(result_w), s = msize, marker = mstyle)
plt.plot(t_net[:tstep + 1], TK(result_w))
plt.scatter(t_net[:tstep + 1], TK(result_w_0), s = msize, marker = mstyle)
plt.plot(t_net[:tstep + 1], TK(result_w_0))

plt.xlabel(r"$t$, s")
plt.ylabel(r"$T_s$, K")

modifier = 4
n_periods = floor(t_net_max / (2 * np.pi / omega)) // modifier
interpolation_step = tstep // n_periods

Ts_average = []
for i in range(n_periods):
    Ts_average.append(TK(np.average(result_w[i * interpolation_step: (i + 1) * interpolation_step])))
interp_t_net = dt * interpolation_step * np.arange(0.5, n_periods + 0.5)
# plt.scatter(interp_t_net, Ts_average, s = msize, marker = mstyle, c = 'tab:blue', alpha = 0.3)
plt.plot(interp_t_net, Ts_average, c = 'tab:blue', alpha = 0.5, linestyle = 'dashed')
# #plt.grid()
plt.savefig("temperature_plot.png")
plt.clf()

plt.xlabel(r"$t$, s")
plt.ylabel(r"$T_{osc} / T_{plain}$, K")
plt.xlim(0, t_net_max)
T_rel = []
for i in range(np.shape(result_w)[0]):
    T_rel.append(result_w[i] / result_w_0[i])
plt.scatter(t_net[:tstep + 1], T_rel, s = msize, marker = mstyle)
plt.savefig("temperature_rel_plot.png")
plt.clf()

modifier = 4
n_periods = floor(t_net_max / (2 * np.pi / omega)) // modifier
interpolation_step = tstep // n_periods

# plt.show()
# plt.scatter(t_net[:tstep], V_vc_log, s = msize, marker = mstyle)
# plt.scatter(t_net[:tstep], V_vc_0_log, s = msize, marker = mstyle)
# plt.plot(t_net[:tstep], V_vc_log)
# plt.scatter(TK(result_w[:tstep]), V_vc_log, s = msize, marker = mstyle)
plt.plot(TK(result_w[:tstep:interpolation_step]), V_vc_log[:tstep:interpolation_step], markersize = msize, marker = mstyle)
# plt.scatter(TK(result_w_0[:tstep]), V_vc_0_log, s = msize, marker = mstyle)
plt.plot(TK(result_w_0[:tstep]), V_vc_0_log, markersize = msize, marker = mstyle)
V_vc_average = []
interp_T_net = []
for i in range(n_periods):
    V_vc_average.append(np.average(V_vc_log[i * interpolation_step: (i + 1) * interpolation_step]))
    interp_T_net.append(TK(np.average(result_w[i * interpolation_step: (i + 1) * interpolation_step])))
plt.plot(interp_T_net, V_vc_average, markersize = msize, marker = mstyle)
plt.xlabel(r"$t$, s")
plt.ylabel(r"$V_vc$")
#plt.grid()
plt.savefig("V_vc_plot.png")
plt.clf()
# plt.show()
# plt.scatter(t_net[:tstep], q_log, s = msize, marker = mstyle)
# plt.scatter(t_net[:tstep], q_0_log, s = msize, marker = mstyle)
# plt.plot(t_net[:tstep], q_log, marker = mstyle, markersize = msize)
# plt.plot(t_net[:tstep], q_0_log, marker = mstyle, markersize = msize)
# plt.plot(t_net[:tstep], make_interp_spline(t_net[:tstep:interpolation_step], q_log[::interpolation_step], 3)(t_net[:tstep]))
# plt.xlabel("t, s")
plt.plot(TK(result_w[:tstep]), q_log, marker = mstyle, markersize = msize)
plt.plot(TK(result_w_0[:tstep]), q_0_log, marker = mstyle, markersize = msize)

q_average = []
q_rel = []
interp_T_net = []
print(n_periods * interpolation_step)
for i in range(n_periods):
    q_average.append(np.average(q_log[i * interpolation_step: (i + 1) * interpolation_step]))
    q_rel.append(q_average[-1] / np.average(q_0_log[i * interpolation_step: 
                                                    (i + 1) *
                                                    interpolation_step]))
    interp_T_net.append(TK(np.average(result_w[i * interpolation_step: (i + 1) * interpolation_step])))
# plt.scatter(interp_t_net, Ts_average, s = msize, marker = mstyle, c = 'tab:blue', alpha = 0.3)
plt.plot(interp_T_net, q_average, c = 'tab:blue', alpha = 0.5, linestyle = 'dashed')

plt.xlabel(r"$T_s$")
plt.ylabel(r"$q$")
#plt.grid()
plt.savefig("q_plot.png")
plt.clf()


plt.plot(interp_T_net, q_rel, c = 'tab:blue', alpha = 0.5, linestyle = 'dashed')
plt.xlabel(r"$T_s$")
plt.ylabel(r"$\frac{<q_{osc}>}{q_0}$")
#plt.grid()
plt.savefig("q_rel_plot.png")
plt.clf()

# plt.show()
# plt.scatter(t_net[:tstep], j_log, s = msize, marker = mstyle)
# plt.scatter(t_net[:tstep], j_0_log, s = msize, marker = mstyle)
# plt.plot(t_net[:tstep], j_log, marker = mstyle, markersize = msize)
plt.plot(t_net[:tstep], j_0_log, marker = mstyle, markersize = msize)
j_average = []
for i in range(n_periods):
    j_average.append(np.average(j_log[i * interpolation_step: (i + 1) * interpolation_step]))
plt.plot(dt * interpolation_step * np.arange(0.5, 0.5 + n_periods),j_average)
# plt.plot(t_net[:tstep], make_interp_spline(t_net[:tstep:interpolation_step], j_log[::interpolation_step], 3)(t_net[:tstep]))
plt.xlabel(r"$t$, s")
plt.ylabel(r"$j$")
#plt.grid()
plt.savefig("j_plot.png")
plt.clf()

modifier = 4
n_periods = floor(t_net_max / (2 * np.pi / omega)) // modifier
interpolation_step = tstep // n_periods

jV_average = []
V2_average = []
z = []
for i in range(n_periods):
    jV_average.append(np.average(j_log[i * interpolation_step: (i + 1) * interpolation_step] * V_vc_log[i * interpolation_step: (i + 1) * interpolation_step]))
    V2_average.append(np.average(V_vc_log[i * interpolation_step: (i + 1) * interpolation_step]**2))
    z.append(V2_average[-1]/jV_average[-1])

plt.xlim(0, dt * tstep)
# plt.ylim(-100, 300)
plt.scatter(dt * interpolation_step * np.arange(0.5, 0.5 + n_periods), z, s=10,marker="x")
plt.plot(dt * interpolation_step * np.arange(0.5, 0.5 + n_periods),z)
plt.title(r"$z_{classic}$ = %.1f     $z_{SCL}$ = %.1f" % (z[0], z[-1]), y = -0.2)
#plt.grid()
plt.savefig("z_plot.png")
print(z)
