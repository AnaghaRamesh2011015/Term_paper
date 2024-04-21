import numpy as np
import pandas as pd

import scipy.integrate as spi
import scipy.optimize as spo

from scipy.integrate import quad
from scipy.optimize import curve_fit
import pandas as pd

# Read the CSV file and extract data from the first column, skipping the first row
data = pd.read_csv("D:/ACADEMICS/Coding/Comp Phy/term_paper/pion_data.csv", usecols=[0], skiprows=[0])
# Convert the extracted data to a list
data_list = data.values.tolist()
pT = np.array(data_list).flatten()
#print(pT)
'''
# Define initial parameter bounds
# T_min, T_max = 0, 1
# R_min, R_max = 0, 1
# beta_s_min, beta_s_max = 0, 1
# n_min, n_max = 0, 1
# #mass of pion
# m_0 = 0.13957
# R = R_max
# def my_integral(f, start, end, steps, tol, *extra_args):
#     return 0 #update
    # x = start
    # f(x, *extra_args)

# def I_0_fn_to_integrate(phi, z):
#     return np.exp(z * np.cos(phi))
    
# def K_1_fn_to_integrate(y, z):
#     return np.cosh(y) * np.exp(-z * np.cosh(y))

# # defining modified bessels function of the first kind
# def I_0(z):
#     return my_integral(I_0_fn_to_integrate, 0, 2 * np.pi, 1000, 1e-6, z) / (2 * np.pi)

# # defining modified bessels function of the second kind
# def K_1(z):
#     return my_integral(K_1_fn_to_integrate, 0, R, 1000, 1e-6, z) 

# # defining RHS to integrate
# def RHS_to_integrate(r, R, pT, T, beta_s, n):
    
#     m_T = np.sqrt((m_0 ** 2) + (pT ** 2))
#     rho = beta_s * ((r / R) ** n)
#     val = r * m_T * I_0(pT * (np.sinh(rho)) / T) * K_1(m_T * (np.cosh(rho)) / T)
    

# # Step 3: Define the Boltzmann-Gibbs blast-wave model function
# def blast_wave(pT, T, R, beta_s, n,r):
#     # Define constants
#     m_T = np.sqrt((m_0 ** 2) + (pT ** 2))
#     rho = beta_s * ((r / R) ** n)
#     integral_func = lambda r: r * np.exp((m_T * np.cosh(rho)) / T) * \
#                      np.i0((pT* np.sinh(rho)) / T) * (r / R) ** n
    
#     # Perform integral
#     integral_result, _ = quad(integral_func, 0, R)
    
#     return integral_result

# # dN_dpt = blast_wave(pT, 0.5, 0.5, 0.5, 0.5)

# Step 4: Define the transverse mass function
# def transverse_mass(pT, m_0):
#     return np.sqrt((m_0 ** 2) + (pT ** 2))

# # Step 6: Fit the model to experimental data
# popt, pcov = curve_fit(blast_wave, pT, dN_dpt, bounds=([T_min, R_min, beta_s_min, n_min], [T_max, R_max, beta_s_max, n_max]))

# # Step 7: Visualize the fitted model
# plt.scatter(pT, dN_dpt, label='Experimental Data')
# plt.plot(pT, blast_wave(pT, *popt), label='Fitted Blast-Wave Model')
# plt.xlabel('pT')
# plt.ylabel('dN/dpt')
# plt.legend()
# plt.show()

import numpy as np

def tsallis_blast_wave(p_T, m, T, beta, q, R):
    """
    Calculate the Tsallis Blast Wave Model for transverse momentum distribution.

    Parameters:
    - p_T: Transverse momentum (array-like)
    - m: Particle mass
    - T: Temperature
    - beta: Radial flow velocity
    - q: Non-extensive parameter
    - R: Blast wave radius

    Returns:
    - dN_dpT: Transverse momentum distribution
    """
    
    rho = np.arctanh(beta)
    m_T = np.sqrt(m**2 + p_T**2)

    # Calculate the weight factor
    weight_factor = 1 + ((q - 1) * m_T * np.cosh(rho) - rho * T * np.sinh(rho) * np.cos(phi)) / T
    weight_factor = weight_factor ** (-1 / (q - 1))

    # Calculate the transverse momentum distribution
    dN_dpT = m_T * weight_factor

    return dN_dpT'''

import numpy as np
from scipy.integrate import tplquad

def tsallis_blast_wave(Y, p_T, m, T, q, R, beta_s, n):
    """
    Calculate the Tsallis Blast Wave Model for transverse momentum distribution.

    Parameters:
    - p_T: Transverse momentum (array-like)
    - m: Particle mass
    - T: Temperature
    - q: Non-extensive parameter
    - R: Blast wave radius

    Returns:
    - dN_dpT: Transverse momentum distribution
    """

    def integrand(y, phi, r, p_T, m_T, T, q, beta_s, n):
        beta = beta_s / (1 + 1 / (n + 1))
        rho = np.arctanh(beta)
        weight_factor = 1 + (((m_T * np.cosh(y) * np.cosh(rho)) - p_T * np.sinh(rho) * np.cos(phi)) *( (q - 1) / T))
        weight_factor = weight_factor ** (-1 / (q - 1))
        weight_factor = weight_factor * r * np.cosh(y)
        #print(weight_factor)
        return weight_factor
    
    dN_dpT = np.zeros_like(p_T)
    for i, p in enumerate(p_T):
        m_T = np.sqrt(m**2 + p**2)
        #print(m_T)
        result, _ = tplquad(integrand, -Y, Y, -np.pi / 2, np.pi / 2, 0, R, args=(p, m_T, T, q, beta_s, n))
        #print(result)
        dN_dpT[i] = result * m_T * p

    return dN_dpT

import numpy as np
from scipy.special import kn, iv
from scipy.integrate import quad

def boltzmann_gibbs_blast_wave(p_T, m, T, beta_s, R, n):
    """
    Calculate the Boltzmann Gibbs Blast Wave Model for transverse momentum distribution.

    Parameters:
    - p_T: Transverse momentum (array-like)
    - m: Particle mass
    - T: Temperature
    - beta: Radial flow velocity
    - R: Blast wave radius

    Returns:
    - dN_dpT: Transverse momentum distribution
    """
    def integrand(r, p_T, m_t, T, beta_s, n):
        beta = beta_s * (r / R) ** n
        rho = np.arctanh(beta)
        arg1 = p_T * np.sinh(rho) / T
        arg2 = m_t * np.cosh(rho) / T
        return r * kn(1, arg2) * iv(0, arg1)

    dN_dpT = np.zeros_like(p_T)
    for i, p in enumerate(p_T):
        m_t = np.sqrt(m**2 + p**2)
        result, _ = quad(integrand, 0, R, args=(p, m_t, T, beta_s, n))
        dN_dpT[i] = result * m_t * p
    return dN_dpT


m_0 = 0.13957
T_0 = 1.2676e-1
beta_s_0 = 4.387e-1
n_0 = 0.5
R_0 = 1.0 #update
q_0 = 1.060
Y_0 = 0.5
dN_dpT_bg = boltzmann_gibbs_blast_wave(pT, m_0, T_0, beta_s_0, R_0, n_0)
dN_dpT_ts = tsallis_blast_wave(Y_0, pT, m_0, T_0, q_0, R_0, beta_s_0, n_0)

#print(1e6*dN_dpT_bg)
print(1e6*dN_dpT_ts)
