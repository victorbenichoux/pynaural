from pynaural.raytracer.faddeeva import faddeeva
import numpy as np

__all__ = ['c', 'spherical_ref_factor', 'atm_abs']

## Constants
c = 342.  # *meter/second #sound speed
T0 = 293.15  # *kelvin #ref temperature (20C)
hr0 = 0.5  # ref hygrometry
To1 = 273.16  # *kelvin # triple point temperature
Pr = 101.325e3  # *pascal #reference ambient atm pressure


def atm_abs(d, f, T=T0, hr=hr0, p=Pr, ref=1.0, broadbandcorrection=False):
    # Computes the desired amplitude modulation for air absorption
    # returns values in 
    # ISO 9613-1:1993
    # http://www.sengpielaudio.com/AirdampingFormula.htm
    # (...) = range of validity
    # d: distance
    #f: frequency (50HZ ... 10 kHz)
    #T: Temperature (-20C ... 50 C)
    #h: humidity level (.1 ... -1)
    #p: atmospheric pressure

    Psat = Pr * 10 ** (-6.8346 * (To1 / T) ** 1.261 + 4.6151)
    h = hr * (Psat / p)

    frN = (p / Pr) * (T / T0) ** (-0.5) * ( 9 + 280 * h * np.exp(-4.170 * ( (T / T0) ** (-1. / 3) - 1)))
    frO = (p / Pr) * (24 + 4.04 * 10 ** 4 * h * ( (0.02 + h) / (0.391 + h) ) )

    z = 0.1068 * np.exp(-3352 / T) * (frN + f ** 2 / frN) ** -1
    y = (T / T0) ** (-5. / 2) * ( 0.01275 * np.exp(-2239.1 / T) * (frO + f ** 2 / frO) ** -1 + z )
    x = 1 / (10 * np.log10((np.exp(1)) ** 2))

    a = 8.686 * f ** 2 * ( 1.84 * 10 ** (-11) * (p / Pr) ** -1 * (T / T0) ** .5 + y )  #absorption coef

    Aa = -20 * np.log10(np.exp(-x * a * d))

    if broadbandcorrection:
        # correction for broad band sound (crocker chap 28)
        Aa = Aa * (1 + 0.00533 * (1 - 0.2303 * Aa)) ** 1.6

    Aa.shape = (len(Aa), 1)

    return Aa


def spreading_abs(d, condition='spherical', ref=1.0):
    g = {'spherical': 1,
         'plane': 0,
         'cylindrical': 0.5}[condition]
    As = g * 20 * np.log10(d / ref + 1)
    return As


def delay_abs(d, f, ref=1.0):
    delay = float(d / c )
    p = np.exp(- 1j * 2 * np.pi * delay * f)
    p.shape = (len(p), 1)
    return p

def freefield_abs(d, f, ref=1):
    At = np.exp(- ( flipnegfreq(atm_abs(d, fullrangefreq(f), ref=ref)) + spreading_abs(d, ref=ref) ) / 20)
    H = At * delay_abs(d, f, ref=ref)
    return H


################### Ground reflection model ############################

def k21_z21(f, sigma, model=''):
    # computes the ratio of wavenumbers, and ratio of impedances
    # Miki, and then Komatsu
    z21 = (1 + 0.0699 * ((f / sigma) ** (-0.632))) + 1j * 0.1071 * ((f / sigma) ** (-0.632))
    k21 = (1 + 0.1093 * ((f / sigma) ** (-0.618))) + 1j * 0.1597 * ((f / sigma) ** (-0.618))
    # tmp = 2-log(f/(1000*sigma))
    # tmp62 = tmp ** 6.2
    # tmp41 = tmp ** 4.1
    # z21 = (1 + 0.00027 * tmp62) - 0.0047 * tmp41
    # k21 = (0.0069 * tmp41) + 1j * (1 + 0.0004 * tmp62)
    return k21, z21


def spherical_ref_factor(d, phi, f, sigma=20000, allres=False, version='mine'):
    # details of the implementation are in boris' paper
    # arguments:
    # d is the total distance from source to head, with reflection
    # phi is the elevation of the new (virtual) source
    # f is the considered frequency
    # sigma is the resistivity of the material in MKS raylgh
    if version == 'boris':
        c = 342
        k = 2 * np.pi * f / c
        k21, z21 = k21_z21(f, sigma)
        temp = (1 - (k21 ** -2) * (np.cos(phi) ** 2))
        # print Theta,z21*sin((Theta)),(temp**0.5)
        Rp = (z21 * np.abs(np.sin(phi)) - (temp ** 0.5)) / (z21 * np.abs(np.sin(phi)) + (temp ** 0.5))
        w = 1j * 2 * k * d / ((1 - Rp) ** 2) * (z21 ** -2) * temp
        #    numd2=1j*k*d/2*((abs(sin(Theta))+z21**-1)**2)/(1+abs(sin(Theta))/z21)
        Fw = 1 + 1j * np.sqrt(np.pi * w) * faddeeva(np.sqrt(w))  #exp(-numd)*(1-erf(-1j*sqrt(numd)))
        return Rp + (1 - Rp) * Fw

    elif version == 'mine':
        f = f + 0j
        k21, z21 = k21_z21(f, sigma)
        S = (z21 ** -2) * (1 - (k21 ** -2) * (np.cos(phi) ) ** 2)
        sqS = S ** 0.5
        R = (np.sin(phi) - sqS) / (np.sin(phi) + sqS)
        c = 343
        la = 2 * np.pi * f / c
        w = 2j * la * d * S * ((1 - R) ** -2)
        F = lambda x: 1 + 1j * (np.pi * x) ** 0.5 * faddeeva(np.sqrt(x))
        Fw = F(w)
        Q = R + (1 - R) * Fw
    if not allres:
        return np.nan_to_num(Q)  # Q
    else:
        return R, w, Fw
