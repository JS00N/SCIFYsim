import kernuller
import sympy as sp
import numpy as np
from kernuller import fprint

import pathlib

parent = pathlib.Path(__file__).parent.absolute()
#asym_data_file = parent/"data/asymetric_coupling_rate_gls.csv"
asym_data_file = parent/"data/asymmetric_coupling_rate_gross.csv"
asym_phase_file = parent/"data/asymmetric_diff_phase_gls.csv"
direc_data_file = parent/"data/directional_coupling_bar.csv"


import logging
logit = logging.getLogger(__name__)

def sp_clip_rows(mat, clip):
    """
        A convenience function to clip sympy matrices
    equivalent to `mat[np.logical_not(clip),:]`

    *Arguments:*
    * mat : A sympy matrix
    * clip : A numpy boolean array of the rows to remove

    *Return:* clipped sympy matrix
    """
    a = np.logical_not(clip)
    ass = np.argwhere(a)
    mymat = sp.Matrix([mat[anass[0],:] for anass in ass])
    return mymat

def four2six():
    sigma = sp.symbols("sigma", real=True)
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter_low = psplitter1.subs([(sigma, 1/3)])
    psplitter_high = psplitter1.subs([(sigma, 2/3)])
    A = sp.diag(psplitter_low, psplitter_low, psplitter_high, psplitter_high)
    B = sp.diag(1, kernuller.crossover, kernuller.crossover, kernuller.crossover, 1)
    C = sp.diag(sp.eye(2), kernuller.splitter, sp.eye(5))
    D = sp.diag(sp.eye(3), kernuller.crossover, kernuller.splitter, kernuller.splitter, sp.eye(2))
    E = sp.diag(sp.eye(3), kernuller.splitter, kernuller.crossover, kernuller.crossover, sp.eye(3))
    M = E@D@C@B@A
    return M
    
def four_photometric(tap_ratio):
    sigma = tap_ratio
    #
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1

    A = sp.diag(psplitter1, psplitter1, psplitter2, psplitter2)
    B = sp.diag(1, kernuller.crossover, sp.eye(2), kernuller.crossover, 1)
    M = B@A
    return M
def ph_shifter(phi):
    return sp.exp(sp.I*phi)
def piston2phase(p, lamb):
    phase = (2*sp.pi*p)/lamb
    return phase
def geom_ph_shifter(p, lamb):
    a = sp.exp(sp.I * piston2phase(p, lamb))
    return a

p, lamb = sp.symbols("p, lambda", real=True)
aphi = (2*sp.pi*p)/lamb


#########################################################
# A physically meaningful representation of a coupler
########################################################
sigma, Delta , a = sp.symbols("sigma, Delta, a", real=True)

ccoupler = sp.sqrt(a)*sp.Matrix([[sp.sqrt(sigma), - sp.sqrt(1-sigma)*sp.exp(-sp.I*(-sp.pi/2 + Delta))],
                        [sp.sqrt(1-sigma)*sp.exp(sp.I*(-sp.pi/2 + Delta)), sp.sqrt(sigma)]])


###########################################
# A simple achromatic coupler written in a physical way
# with phasors compatible with the physics
thesubs = [(sigma, 1/2),
           (Delta, 0),
           (a, 1)]

pcoupler = ccoupler.subs(thesubs)

#########################################################
# The implementation from H.-D. Kenchington Goldsmith (11)
#########################################################
##########################################################
# The model provided for 4µm MMI couplers
##########################################################

coupling_KG = 0.1727439475*lamb**6\
            - 3.7494749619*lamb**5\
            + 33.3100408411*lamb**4\
            - 155.0153473295*lamb**3\
            + 398.7785099597*lamb**2\
            - 538.7869436749*lamb\
            + 300.7063779033
coupling_KG = coupling_KG.subs([(lamb, lamb*1e6)])
loss_KG =  -2.056612*lamb**6\
        + 49.537979*lamb**5\
        - 493.329044*lamb**4\
        + 2599.382569*lamb**3\
        - 7642.251261*lamb**2\
        + 11887.167001*lamb\
        - 7642.937666
loss_KG = loss_KG.subs([(lamb, lamb*1e6)])
phase_KG = 1.2036866257*lamb**6\
        - 27.3368521128*lamb**5\
        + 200.3471800676*lamb**4\
        - 377.0537163354*lamb**3\
        - 1747.6266395813*lamb**2\
        + 8385.3602984720*lamb\
        - 9585.2130982070
##########################################
## Convert lambda in µm and phase_KG in deg!!
phase_KG = phase_KG.subs([(lamb, lamb*1e6)])
phase_KG = phase_KG * sp.pi/180
phase_KG = phase_KG - sp.pi/2
##########################################


###########################################
# Implementation of the asymmetric combiner
# from Sharma et al. 2020
# Data taken from Fig. 6 (left)
###########################################
xk = np.genfromtxt(asym_data_file, delimiter=",")
x = xk[1:,0]*1e-6 # will be tune for the correct band #1:20 originally, the 1:23 with points added
k = xk[1:,1]
coefs = np.polyfit(x, k, 6)
#xs = sp.symbols("x", real=True)
coupling_Sharma = sp.Add(*[a*lamb**i for i, a in enumerate(np.flip(coefs))])
# Now for the phase
xk = np.genfromtxt(asym_phase_file, delimiter=",")
x = xk[1:,0]*1e-6# + 0.4e-6 # will be tune for the correct band
k = xk[1:,1] * np.pi/180 + np.pi/5 - np.pi # Converting to degrees
coefs = np.polyfit(x, k, 1)
#xs = sp.symbols("x", real=True)
phase_Gross = sp.Add(*[a*lamb**i for i, a in enumerate(np.flip(coefs))])
#polyn = sp.lambdify((xs), polys, modules="numpy")
#################################################
# Implementation of the directional coupler
# Data from Tepper et al. 2017
xk = np.genfromtxt(direc_data_file, delimiter=",")
# Here, making the asumption that the combiner 
x = xk[1:,0]*1e-6 + 0.4e-6 # will be tune for the correct band
k = xk[1:,1]
coefs = np.polyfit(x, k, 1)
coupling_Tepper = sp.Add(*[a*lamb**i for i, a in enumerate(np.flip(coefs))])
################################################
# The KG case
thesubs = [(sigma, coupling_KG),
           (Delta, phase_KG ),
           (a, loss_KG),
           (lamb, lamb+0.08e-6)]
M_KG = ccoupler.subs(thesubs)
# Needs an adjustment phasor to get the expected matrix
#adjust_phasor = sp.diag(1,sp.exp(-sp.I*sp.pi/2))
#M_KG = M_KG@adjust_phasor
###########################################
# The Sharma 2020 combiner
thesubs = [(sigma, coupling_Sharma),
           (Delta, phase_Gross),
           (a, 1)]
M_Sharma = ccoupler.subs(thesubs)
#M_Sharma = M_Sharma@adjust_phasor
###########################################
# The Tepper 2017 combiner
thesubs = [(sigma, coupling_Tepper),
           (Delta, 0),
           (a, 1)]
M_Tepper = ccoupler.subs(thesubs)
###########################################

def bracewell_ph(include_masks=False, tap_ratio=None):
    """
    in: 2
    out: 4 (ph0, bright, dark, ph1)
    symbols:
        sigma  : the photometric ratio (in intensity)
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner M
    Free symbols can be retrieved in list(M.free_symbols)
    """
    #sigma = sp.symbols("sigma", real=True)
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1
    #fprint(psplitter1, "\mathbf{Y}_1 = ")
    #fprint(psplitter2, "\mathbf{Y}_2 = ")
    #fprint(kernuller.xcoupler, "\mathbf{X} = ")

    A = sp.diag(psplitter1, psplitter2)
    B = sp.diag(1,kernuller.xcoupler, 1)

    combiner = B@A
    #fprint(combiner1, "\mathbf{M}_1 = ")
    #print("Here with a 0.05 splitting ratio")
    #Mn = kernuller.sp2np(combiner1.subs([(sigma, 0.05)])).astype(np.complex128)
    #fig, axs = kernuller.cmp(Mn, nx=1, out_label=np.arange(4), mainlinewidth=0.05)
    if include_masks:
        bright = np.array([False,True,  False, False])
        dark = np.array([False, False, True, False])
        photometric = np.array([True, False, False, True])
    if tap_ratio is not None:
        combiner = combiner.subs([(sigma, tap_ratio)])
    
    if include_masks:
        return combiner, bright, dark, photometric
    else:
        return combiner



def angel_woolf_ph(ph_shifters=None, include_masks=False, tap_ratio=None):
    """
    optional :
    ph_shifters : a list of phase shifters in between the 2 stages
                (eg: for kernel-nuller ph_shifters=[0, sp.pi/2])
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    #sigma = sp.symbols("sigma", real=True)
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1
    #fprint(psplitter1, "\mathbf{Y}_1 = ")
    #fprint(psplitter2, "\mathbf{Y}_2 = ")
    #fprint(kernuller.xcoupler, "\mathbf{X} = ")

    A = sp.diag(psplitter1, psplitter1, psplitter2, psplitter2)
    B = sp.diag(1, kernuller.crossover, sp.eye(2), kernuller.crossover, 1)
    C = sp.diag(sp.eye(2),kernuller.xcoupler, kernuller.xcoupler, sp.eye(2))
    C2 = sp.diag(sp.eye(4), kernuller.crossover, sp.eye(2))
    D = sp.diag(sp.eye(3), kernuller.ph_shifter(phi[0]), kernuller.ph_shifter(phi[1]), sp.eye(3))
    E = sp.diag(sp.eye(3), kernuller.xcoupler, sp.eye(3))
    E1  = sp.diag(sp.eye(3), kernuller.ph_shifter(sp.pi/2), kernuller.ph_shifter(sp.pi), sp.eye(3))


    combiner = E1@E@D@C2@C@B@A
    #fprint(combiner2, "\mathbf{M}_2 = ")
    
    if ph_shifters is not None:
        thesubs = [(phi[0], ph_shifters[0]),
                  (phi[1], ph_shifters[1])]
        combiner = combiner.subs(thesubs)
    
    if tap_ratio is not None:
        combiner = combiner.subs([(sigma, tap_ratio)])
    
    if include_masks:
        bright = np.array([False, False, True, False, False, True, False, False])
        dark = np.array([False, False, False, True, True, False, False, False])
        photometric = np.array([True, True, False, False, False, False, True, True])
        return combiner, bright, dark, photometric
    else:
        return combiner

def angel_woolf_ph_chromatic(ph_shifters=None, include_masks=False,
                             offset=True, tap_ratio=None, 
                             Mc=M_KG, input_ph_shifters=None):
    """
    optional :
    ph_shifters : a list of phase shifters in between the 2 stages
                (eg: for kernel-nuller ph_shifters=[0, sp.pi/2])
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevnant outputs
    Mc          : The chromatic combiner model default: M_KG
                
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    #sigma = sp.symbols("sigma", real=True)
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    psi = sp.Matrix(sp.symbols('psi0:{}'.format(4), real=True))
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1
    #fprint(psplitter1, "\mathbf{Y}_1 = ")
    #fprint(psplitter2, "\mathbf{Y}_2 = ")
    #fprint(kernuller.xcoupler, "\mathbf{X} = ")
    A0 = sp.diag(ph_shifter(psi[0]),ph_shifter(psi[1]), ph_shifter(psi[2]),ph_shifter(psi[3]))
    A = sp.diag(psplitter1, psplitter1, psplitter2, psplitter2)
    B = sp.diag(1, kernuller.crossover, sp.eye(2), kernuller.crossover, 1)
    C = sp.diag(sp.eye(2),Mc, Mc, sp.eye(2))
    C2 = sp.diag(sp.eye(4), kernuller.crossover, sp.eye(2))
    D = sp.diag(sp.eye(3), kernuller.ph_shifter(phi[0]), kernuller.ph_shifter(phi[1]), sp.eye(3))
    E = sp.diag(sp.eye(3), Mc, sp.eye(3))
    E1  = sp.diag(sp.eye(3), kernuller.ph_shifter(sp.pi/2), kernuller.ph_shifter(sp.pi), sp.eye(3))
    
    combiner = E1@E@D@C2@C@B@A@A0
    if offset:
        combiner = combiner# + sp.ones(combiner.shape[0], combiner.shape[1])*1.e-20*lamb
    # fprint(combiner2, "\mathbf{M}_2 = ")
    
    if ph_shifters is not None:
        thesubs = [(phi[0], ph_shifters[0]),
                  (phi[1], ph_shifters[1])]
        combiner = combiner.subs(thesubs)
    if input_ph_shifters is not None:
        thesubs = [(thepsi, thevalue) for thepsi, thevalue in zip(psi, input_ph_shifters)]
        combiner = combiner.subs(thesubs)
    else:
        thesubs = [(thepsi, 0) for thepsi in psi]
        combiner = combiner.subs(thesubs)
    if tap_ratio is not None:
        combiner = combiner.subs([(sigma, tap_ratio)])
    if np.isclose(tap_ratio, 0.):
        clip_photo = True
    else:
        clip_photo = False

    bright = np.array([False, False, True, False, False, True, False, False], dtype=bool)
    dark = np.array([False, False, False, True, True, False, False, False], dtype=bool)
    photometric = np.array([True, True, False, False, False, False, True, True], dtype=bool)

    if clip_photo:
        combiner = sp_clip_rows(combiner, photometric)
        bright = bright[np.logical_not(photometric)]
        dark = dark[np.logical_not(photometric)]
        photometric = photometric[np.logical_not(photometric)]
    
    if include_masks:
        return combiner, bright, dark, photometric
    else:
        return combiner


def kernel_nuller_3T(include_masks=False, tap_ratio=None):
    """
    optional :
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    
    from kernuller.nullers import matrices_3T
    kernel_nuller_3T = matrices_3T[0]
    #sigma = sp.symbols("sigma", real=True)
    
    #photometric_preamble = four_photometric(sigma)
    #C = sp.diag(sp.eye(2), kernel_nuller_4T, sp.eye(2))
    #coupler = C@photometric_preamble
    coupler = kernel_nuller_3T
    
    if tap_ratio is not None:
        coupler = coupler.subs([(sigma, tap_ratio)])
    if include_masks:
        bright = np.array([True, False, False])
        dark = np.array([False, True, True])
        photometric = np.array([False, False, False])
        return coupler, bright, dark, photometric
    else:
        return coupler
def kernel_nuller_4T(include_masks=False, tap_ratio=None):
    """
    optional :
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    
    from kernuller.nullers import matrices_4T
    kernel_nuller_4T = matrices_4T[0]
    #sigma = sp.symbols("sigma", real=True)
    
    #photometric_preamble = four_photometric(sigma)
    #C = sp.diag(sp.eye(2), kernel_nuller_4T, sp.eye(2))
    #coupler = C@photometric_preamble
    coupler = kernel_nuller_4T
    
    if tap_ratio is not None:
        coupler = coupler.subs([(sigma, tap_ratio)])
    if include_masks:
        bright = np.array([True,
                           False, False, False, False, False, False])
        dark = np.array([False,
                         True,  True,  True,  True,  True,  True])
        photometric = np.array([False,
                                False, False, False, False, False, False])
        return coupler, bright, dark, photometric
    else:
        return coupler
    
def kernel_nuller_5T(include_masks=False, tap_ratio=None):
    """
    optional :
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    
    from kernuller.nullers import matrices_5T
    kernel_nuller_5T = matrices_5T[0]
    #sigma = sp.symbols("sigma", real=True)
    
    #photometric_preamble = four_photometric(sigma)
    #C = sp.diag(sp.eye(2), kernel_nuller_4T, sp.eye(2))
    #coupler = C@photometric_preamble
    coupler = kernel_nuller_5T
    
    if tap_ratio is not None:
        coupler = coupler.subs([(sigma, tap_ratio)])
    if include_masks:
        bright = np.array([True,
                           False, False, False, False, False, False,
                           False, False, False, False, False, False])
        dark = np.array([False,
                         True,  True,  True,  True,  True,  True, 
                         True,  True,  True,True,  True,  True])
        photometric = np.array([False,
                                False, False, False, False, False, False,
                           False, False, False, False, False, False])
        return coupler, bright, dark, photometric
    else:
        return coupler
    

def kernel_nuller_6T(include_masks=False, tap_ratio=None):
    """
    optional :
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    
    from kernuller.nullers import matrices_6T
    kernel_nuller_6T = matrices_6T[0]
    #sigma = sp.symbols("sigma", real=True)
    
    #photometric_preamble = four_photometric(sigma)
    #C = sp.diag(sp.eye(2), kernel_nuller_4T, sp.eye(2))
    #coupler = C@photometric_preamble
    coupler = kernel_nuller_6T
    
    if tap_ratio is not None:
        coupler = coupler.subs([(sigma, tap_ratio)])
    if include_masks:
        bright = np.array([True, 
                                False, False, False, False, False,
                                False, False, False, False, False,
                                False, False, False, False, False,
                                False, False, False, False, False])
        dark = np.array([False,
                         True, True, True, True, True, 
                         True, True, True, True, True, 
                         True, True, True, True, True, 
                         True, True, True, True, True])
        photometric = np.array([False,
                                False, False, False, False, False,
                                False, False, False, False, False,
                                False, False, False, False, False,
                                False, False, False, False, False])
        return coupler, bright, dark, photometric
    else:
        return coupler
    

def VIKiNG(ph_shifters=None, include_masks=False, tap_ratio=None):
    """
    optional :
    ph_shifters : a list of phase shifters in between the 2 stages
                (eg: for kernel-nuller ph_shifters=[0, sp.pi/2])
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    
    from kernuller.nullers import matrices_4T
    kernel_nuller_4T = matrices_4T[0]
    #sigma = sp.symbols("sigma", real=True)
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    sigma = sp.symbols("sigma", real=True)
    
    photometric_preamble = four_photometric(sigma)
    C = sp.diag(sp.eye(2), kernel_nuller_4T, sp.eye(2))
    VIKiNG = C@photometric_preamble
    
    if tap_ratio is not None:
        VIKiNG = VIKiNG.subs([(sigma, tap_ratio)])
    if np.isclose(tap_ratio, 0.):
        clip_photo = True
    else:
        clip_photo = False
    bright = np.array([False, False, True, False, False, False, False, False, False, False, False], dtype=bool)
    dark = np.array([False, False, False, True, True, True, True, True, True, False, False], dtype=bool)
    photometric = np.array([True, True, False, False, False, False, False, False, False,  True, True], dtype=bool)
    if clip_photo:
        VIKiNG = sp_clip_rows(VIKiNG, photometric)
        bright = bright[np.logical_not(photometric)]
        dark = dark[np.logical_not(photometric)]
        photometric = photometric[np.logical_not(photometric)]
        
    if include_masks:
        return VIKiNG, bright, dark, photometric
    else:
        return VIKiNG
    

    
def ABCD(Mc=kernuller.xcoupler,
         ph_shifter_type="achromatic",
         wl=None):
    """
    Build an ABCD combiner.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    if ph_shifter_type == "achromatic":
        phasor = sp.exp(sp.I*sp.pi/2)
    elif ph_shifter_type == "geometric":
        piston = wl/4 # This is to get a pi/2 at wl
        phaseterm = aphi.subs([(p,piston)])
        phasor = sp.exp(sp.I*phaseterm)
    A = sp.diag(kernuller.splitter, kernuller.splitter)
    B = sp.diag(1, kernuller.crossover, 1)
    C = sp.diag(phasor, sp.eye(3))
    D = sp.diag(Mc, Mc)
    ABCD = D@C@B@A
    return ABCD

def GRAVITY(Mc=kernuller.xcoupler,
            wl=None,
            ph_shifter_type="achromatic"):
    """
    Build a 4 input baseline-wise ABCD combiner
    similar in principle to the one used in GRAVITY.
    E@D@C@B@A
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    theabcd = ABCD(Mc=Mc,
                   ph_shifter_type=ph_shifter_type,
                   wl=wl)
    F = sp.diag(theabcd, theabcd, theabcd, theabcd, theabcd, theabcd)
    M = four2six()
    GRAVITY = F@M
    return GRAVITY #A, B, C, D, E, F


def GLINT(include_masks=False, tap_ratio=None):
    """
    Build a 4 input baseline-wise Bracewell combiner
    similar in principle to the one used in GLINT.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    b_nuller = kernuller.xcoupler
    sigma = sp.symbols("sigma", real=True)
    
    photometric_preamble = four_photometric(sigma)
    beam2baseline = sp.diag(sp.eye(2), four2six(), sp.eye(2))
    main_stage = sp.diag(sp.eye(2), b_nuller, b_nuller, b_nuller,
                b_nuller,b_nuller,b_nuller, sp.eye(2))
    GLINT = main_stage@beam2baseline@photometric_preamble
    if tap_ratio is not None:
        GLINT = GLINT.subs([(sigma, tap_ratio)])
    if include_masks:
        bright = np.array([False, False,# Photometries 0 and 1
                          True, False,  # Combination 0-1
                          True, False,  # Combination 0-2
                          True, False,  #Combination 1-2
                          True, False,  # Combination 0-3
                          True, False,  # Combination 1-3
                          True, False,  # Combination 2-3
                          False, False  # Photometries 2 and 3
                          ])
        dark = np.array([False, False,  # Photometries 0 and 1
                          False, True,  # Combination 0-1
                          False, True,  # Combination 0-2
                          False, True,   # Combination 1-2
                          False, True,  # Combination 0-3
                          False, True,  # Combination 1-3
                          False, True,  # Combination 2-3
                          False, False  # Photometries 2 and 3
                          ])
        photometric = np.array([True, True, # Photometries 0 and 1
                          False, False,   # Combination 0-1
                          False, False,   # Combination 0-2
                          False, False,   # Combination 1-2
                          False, False,   # Combination 0-3
                          False, False,   # Combination 1-3
                          False, False,   # Combination 2-3
                          True, True      # Photometries 2 and 3
                          ])
        return GLINT, bright, dark, photometric
    else:
        return GLINT

    
def test_combiners():
    b = bracewell_ph()
    a = angel_woolf_ph(ph_shifters=[0, sp.pi/2])
    
    #sigma = list(b.free_symbols)[0]
    thesubs = [(sigma, 0.05)]
    
    fprint(b)
    Mn = kernuller.sp2np(b.subs(thesubs)).astype(np.complex128)
    fig, axs = kernuller.cmp(Mn, nx=1, out_label=np.arange(4), mainlinewidth=0.05)
    
    fprint(a)
    Mn = kernuller.sp2np(a.subs(thesubs)).astype(np.complex128)
    fig, axs = kernuller.cmp(Mn, nx=2, out_label=np.arange(8), mainlinewidth=0.05)
    
