import numpy as np
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
import matplotlib.pyplot as plt
from scipy.special import erfc, erf
from math import floor, log10, ceil
from scipy.optimize import fsolve

# Plots settings
plt.rcParams.update({"font.size": 13})
params = {"text.latex.preamble": r"\usepackage{icomma}"}
# locale.setlocale(locale.LC_NUMERIC, "ru_RU")
plt.rcParams["axes.formatter.use_locale"] = True
plt.rcParams.update(params)

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
rho = 18.22 # Density
Cp  = 217.8 * 1.0e7 * 1.0e-3 # Isobaric heat capacity
kappa   = 21.2 * 1.0e6 * 1.0e4 # Thermal diffusivity

# Plasma properties
nse = 1.0e13 # plasma density
Te  = 200.0 * eV_to_erg # electron temperature
deltae  = 0.0 # SEE coefficient

# Calculated constants
cs = np.sqrt(Te / mi) # Speed velocity in plasma
r_debye = np.sqrt(Te / (4 * np.pi * nse * e**2)) # Debye radius

# Conversion functions
TK = lambda T : T * Te * erg_to_K
TD = lambda T : T / Te * K_to_erg

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

def Poisson_integrated_classic_trans(phi, y, args):
    Tw, V_f, ne_se, phi_se = y
    nte_w, upsilon_0 = args
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

def Poisson_classic_trans(y, args):
    Tw, V_f, ne_se, phi_se = y
    nte_w, upsilon_0 = args
    return -2.0 * (
        Poisson_integrated_classic_trans(phi_se + V_f, y, args)
        - Poisson_integrated_classic_trans(phi_se, y, args)
    )


def quasineutrality_trans(y, args):
    Tw, V_f, ne_se, phi_se = y
    nte_w, upsilon_0 = args
    return 1 - nte_w * erfcxexp_limit_resolve(-V_f / Tw) - ne_se

def Bohm_criterion_trans(y, args):
    Tw, V_f, ne_se, phi_se = y
    nte_w, upsilon_0 = args
    return phi_se + 0.5 * Tw / (
        ne_se * Tw
        + nte_w
        * (
            erfcxexp_limit_resolve(-V_f / Tw)
            - 1 / (np.sqrt(np.pi) * np.sqrt(-V_f / Tw))
        )
    )

def j_wall_trans(y, args):
    Tw, V_f, ne_se, phi_se = y
    nte_w, upsilon_0 = args
    return V_f - np.log(
        4 * upsilon_0 / (ne_se * np.sqrt(8 * mi / (np.pi * me)))
        + nte_w / ne_se * np.sqrt(Tw)
    )

def sys_trans(y):
    Tw, V_f, ne_se, phi_se = y
    args = [nte_w_func(0, Tw), upsilon_0_func(phi_se)]
    return [
        Poisson_classic_trans(y, args),
        j_wall_trans(y, args),
        Bohm_criterion_trans(y, args),
        quasineutrality_trans(y, args),
    ]
# Solving sys_trans to find transition point properties
sol_trans_init_guesses = [
    [TD(3165), -1.1866, 0.892, -0.591],
    [TD(2900), -1.1866, 0.892, -0.591],
    [TD(2750), -1.15, 0.9, -0.591],
    [TD(3200), -1.25, 0.86, -0.6]
]

is_trans_sol_found = False
for i in range(len(sol_trans_init_guesses)):
    sol_trans = fsolve(sys_trans, sol_trans_init_guesses[i])
    # print(sys_trans(sol_trans))
    if (np.isclose(sys_trans(sol_trans), np.zeros(len(sol_trans))) == np.ones(len(sol_trans), dtype = bool)).all():
        is_trans_sol_found = True
        break
     
if is_trans_sol_found == False:
    raise NameError('Appropriate sol_trans not found. Add more initial guesses')
else:
	Tw_trans, V_f_trans, ne_se_trans, phi_se_trans = sol_trans
	print(
		"Transition at : Tw = ",
		f"{TK(Tw_trans):.0f}\n",
		"\tV_f_trans = ",
		V_f_trans,
		"\n\tphi_se_trans = ",
		phi_se_trans,
		"\n\tne_se_trans = ",
		ne_se_trans,
	)
 # System for classic regime
def j_wall_classic(y, args):
    derw, V_f, ne_se, phi_se = y
    Tw, nte_w, upsilon_0 = args
    return V_f - np.log(
        4 * upsilon_0 / (ne_se * np.sqrt(8 * mi / (np.pi * me)))
        + nte_w / ne_se * np.sqrt(Tw)
    )

def Bohm_criterion_classic(y, args):
    derw, V_f, ne_se, phi_se = y
    Tw, nte_w, upsilon_0 = args
    return phi_se + 0.5 * Tw / (
        ne_se * Tw
        + nte_w
        * (
            erfcxexp_limit_resolve(-V_f / Tw)
            - 1 / (np.sqrt(np.pi) * np.sqrt(-V_f / Tw))
        )
    )


def quasineutrality_classic(y, args):
    derw, V_f, ne_se, phi_se = y
    Tw, nte_w, upsilon_0 = args
    return 1 - nte_w * erfcxexp_limit_resolve(-V_f / Tw) - ne_se


def Poisson_integrated_classic(phi, y, args):
    derw, V_f, ne_se, phi_se = y
    Tw, nte_w, upsilon_0 = args
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
    derw, V_f, ne_se, phi_se = y
    Tw, nte_w, upsilon_0 = args
    return derw**2 - 2.0 * (
        Poisson_integrated_classic(phi_se + V_f, y, args)
        - Poisson_integrated_classic(phi_se, y, args)
    )


def sys_classic(y, *args):
    derw, V_f, ne_se, phi_se = y
    Tw, = args
    args1 = [Tw, nte_w_func(derw, Tw), upsilon_0_func(phi_se)]
    return [
        quasineutrality_classic(y, args1),
        j_wall_classic(y, args1),
        Bohm_criterion_classic(y, args1),
        Poisson_classic(y, args1)
    ]

# Solving classic regime system

Tw_classic_net_min = TD(2000)
Tw_classic_net_max = Tw_trans
Tw_classic_net_steps = 501
Tw_classic_net = np.linspace(
    Tw_classic_net_min, Tw_classic_net_max, Tw_classic_net_steps
)
dT_classic_net = Tw_classic_net[1] - Tw_classic_net[0]

V_f_classic_net = np.zeros(Tw_classic_net_steps)
phi_se_classic_net = np.zeros(Tw_classic_net_steps)
ne_se_classic_net = np.zeros(Tw_classic_net_steps)
derw_classic_net = np.zeros(Tw_classic_net_steps)

for i in range(Tw_classic_net_steps - 1, -1, -1): # can be easily done in a forward iteration, too lazy to rework
    Tw = Tw_classic_net[i]
    args = (Tw,)
    if (i >= Tw_classic_net_steps - 2):
        sol_classic = fsolve(sys_classic, [0.0, V_f_trans, ne_se_trans, phi_se_trans], args=args)
    else:
        j = 0
        while (not (np.isclose(sys_classic(sol_classic, *args), np.zeros(len(sol_classic))) == np.ones(len(sol_classic), dtype = bool)).all()) and j < 10:
            j += 1
            sol_classic = fsolve(sys_classic, 
                [   
                    derw_classic_net[i + 1] + (derw_classic_net[i + 1] - derw_classic_net[i + 2]) / 5 * j,
                    V_f_classic_net[i + 1] + (V_f_classic_net[i + 1] - V_f_classic_net[i + 2]) / 5 * j, 
                    ne_se_classic_net[i + 1] + (ne_se_classic_net[i + 1] - ne_se_classic_net[i + 2]) / 5 * j, 
                    phi_se_classic_net[i + 1] + (phi_se_classic_net[i + 1] - phi_se_classic_net[i + 2]) / 5 * j
                ], 
                args=args
            )
    derw_classic_net[i] = sol_classic[0]
    V_f_classic_net[i] = sol_classic[1]
    ne_se_classic_net[i] = sol_classic[2]
    phi_se_classic_net[i] = sol_classic[3]
    # if not (np.isclose(sys_classic(sol_classic, *args), np.zeros(len(sol_classic))) == np.ones(len(sol_classic), dtype = bool)).all():
    #     print(f"{TK(Tw):.1f}(i = {i})", " : ", sys_classic(sol_classic, *args))

# System for SCL regime

def jwall_SCL(y, args):
    derw, V_f, ne_se, phi_se, V_vc = y
    Tw, nte_w, upsilon_0 = args
    dip = (V_f - V_vc) / Tw
    return (
        upsilon_0
        - 0.25 * ne_se * np.sqrt(8 * mi / (np.pi * me)) * np.exp(V_vc)
        + 0.25 * nte_w * np.sqrt(Tw * 8 * mi / (np.pi * me)) * np.exp(-dip)
    )  #


def quasineutrality_SCL(y, args):
    derw, V_f, ne_se, phi_se, V_vc = y
    Tw, nte_w, upsilon_0 = args
    dip = (V_f - V_vc) / Tw
    return (
        1 - ne_se - nte_w * erfcxexp_limit_resolve(-V_vc / Tw) * np.exp(-dip)
    )  #


def Bohm_SCL(y, args):
    derw, V_f, ne_se, phi_se, V_vc = y
    Tw, nte_w, upsilon_0 = args
    dip = (V_f - V_vc) / Tw
    return (
        -1.0 / upsilon_0**2
        + ne_se
        + nte_w
        / Tw
        * np.exp(-dip)
        * (
            erfcxexp_limit_resolve(-V_vc / Tw)
            - 1 / np.sqrt(np.pi * (-V_vc) / Tw)
        )
    )  # ~~~


def Poisson_integrated_SCL_beta(phi, y, args):
    derw, V_f, ne_se, phi_se, V_vc = y
    Tw, nte_w, upsilon_0 = args
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
    derw, V_f, ne_se, phi_se, V_vc = y
    return -2 * (
        Poisson_integrated_SCL_beta(V_vc + phi_se, y, args)
        - Poisson_integrated_SCL_beta(phi_se, y, args)
    )

def Poisson_integrated_SCL_alpha(phi, y, args):
    derw, V_f, ne_se, phi_se, V_vc = y
    Tw, nte_w, upsilon_0 = args
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
    derw, V_f, ne_se, phi_se, V_vc = y
    return derw**2 - 2 * (
        Poisson_integrated_SCL_alpha(V_f + phi_se, y, args)
        - Poisson_integrated_SCL_alpha(V_vc + phi_se, y, args)
    )

def sys_SCL(y, *args):
    derw, V_f, ne_se, phi_se, V_vc = y
    Tw, = args
    nte_w = nte_w_func(derw, Tw)
    upsilon_0 = upsilon_0_func(phi_se)
    args1 = [Tw, nte_w, upsilon_0]
    return [
        quasineutrality_SCL(y, args1),
        jwall_SCL(y, args1),
        Bohm_SCL(y, args1),
        Poisson_SCL_alpha(y, args1),
        Poisson_SCL_beta(y, args1),
    ]

# Solving systen for SCL regime
Tw_SCL_net_max = TD(3695) # Tungsten melting point
Tw_SCL_net_min = Tw_trans
Tw_SCL_net_steps = 101
Tw_SCL_net = np.linspace(Tw_SCL_net_min, Tw_SCL_net_max, Tw_SCL_net_steps)

V_f_SCL_net = np.zeros(Tw_SCL_net_steps)
phi_se_SCL_net = np.zeros(Tw_SCL_net_steps)
V_vc_SCL_net = np.zeros(Tw_SCL_net_steps)
ne_se_SCL_net = np.zeros(Tw_SCL_net_steps)
derw_SCL_net = np.zeros(Tw_SCL_net_steps)

for i in range(Tw_SCL_net_steps):
    Tw = Tw_SCL_net[i]
    args = (Tw, )
    if i == 0:
        sol_SCL = fsolve(
            sys_SCL,
            [
                derw_classic_net[-1],
                V_f_classic_net[-1],
                ne_se_classic_net[-1],
                phi_se_classic_net[-1],
                V_f_classic_net[-1],
            ],
            args=args,
        )
    else:
        sol_SCL = fsolve(
            sys_SCL,
            [
                derw_SCL_net[i-1] - 0.05,
                V_f_SCL_net[i-1] + 0.00005, # 0.0001 works for Te <= 150 eV, 0.00005 - for higher Te 
                ne_se_SCL_net[i-1],
                phi_se_SCL_net[i-1],
                V_vc_SCL_net[i-1],
            ],
            args=args,
        )
    derw_SCL_net[i] = sol_SCL[0]
    V_f_SCL_net[i] = sol_SCL[1]
    ne_se_SCL_net[i] = sol_SCL[2]
    phi_se_SCL_net[i] = sol_SCL[3]
    V_vc_SCL_net[i] = sol_SCL[4]
    # if not (np.isclose(sys_SCL(sol_SCL, *args), np.zeros(len(sol_SCL))) == np.ones(len(sol_SCL), dtype = bool)).all():
    #     print(f"{TK(Tw):.1f}", " : ", sys_SCL(sol_SCL, *args))
    #     True

# Plotting result

Tw_all_net = np.concatenate((Tw_classic_net, Tw_SCL_net[1:]))
Tw_all_net_plot = np.zeros(np.shape(Tw_all_net)[0])
for i in range(len(Tw_all_net)):
	Tw_all_net_plot[i] = TK(Tw_all_net[i])

V_f_all_net = np.concatenate((V_f_classic_net, V_f_SCL_net[1:]))
V_vc_all_net = np.concatenate((V_f_classic_net, V_vc_SCL_net[1:]))
derw_all_net = np.concatenate((derw_classic_net, derw_SCL_net[1:]))
ne_se_all_net = np.concatenate((ne_se_classic_net, ne_se_SCL_net[1:]))
phi_se_all_net = np.concatenate((phi_se_classic_net, phi_se_SCL_net[1:]))

fig = plt.figure(figsize=(8, 6), dpi=300)

plt.plot(Tw_all_net_plot, V_f_all_net, lw=2, label=r"$V_f$")
plt.plot(Tw_all_net_plot, phi_se_all_net, lw=2, label=r"$\varphi_{se}$")
plt.plot(Tw_all_net_plot, ne_se_all_net, lw=2, label=r"$n_e^{se}$")
plt.plot(Tw_all_net_plot, derw_all_net, lw=2, label=r"$\varphi^'(x = 0)$")
plt.plot(Tw_all_net_plot[Tw_classic_net_steps:], V_vc_SCL_net[1:], lw=2, label=r"$V_{vc}$", color = 'cyan', linestyle = '-.')

plt.text(
    # (min(Tw_all_net_plot) + TK(Tw_trans)) / 2 - 300,
	2200,
    (ne_se_all_net[0] + phi_se_all_net[0])/2,
    r"Классический"
    "\n"
    r"режим",
    fontsize=12,
    bbox=props,
)

plt.text(
    # TK(Tw_trans) + 100,
	3100,
    (ne_se_all_net[0] + phi_se_all_net[0])/2,
    r"Режим экранирования"
    "\n"
    r"объёмным зарядом",
    fontsize=12,
    bbox=props,
)

plt.xlim(min(Tw_all_net_plot), max(Tw_all_net_plot))
plt.ylim(-3.0, 1.5)
plt.plot((TK(Tw_trans), TK(Tw_trans)), (-3.0, 1.5), lw = 2, color = 'k', scaley=False, linestyle = 'dashed')
plt.legend(loc = 'lower right', ncols=2, framealpha = 1)
plt.grid()
plt.title(r"$n_i^{se} = %.1f\cdot10^{%d}$ см$^{-3}$   "
    r"$T_e = %0.f$ эВ   "
    r"$T_{trans} = %0.f$ К"
    % (nse / 10 ** floor(log10(nse)), floor(log10(nse)), Te * erg_to_eV, TK(Tw_trans)),
	y = -0.2)
plt.xlabel(r"$T_w$(K)", fontdict = dict(fontsize = 18))
plt.savefig("data/plot_parameters-Tw/Te=%deV_nse=%0.1fe%d.png" %(ceil(Te * erg_to_eV), nse / 10 ** floor(log10(nse)), floor(log10(nse))))
plt.show()
