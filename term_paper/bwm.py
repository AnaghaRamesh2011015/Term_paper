import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
from scipy.special import kn, iv
from scipy.integrate import quad


# Read the CSV file and extract data from the first column, skipping the first row
data = pd.read_csv("D:/ACADEMICS/Coding/Comp Phy/term_paper/pion_data.csv", usecols=[0,3], skiprows=[0])
lim_length = 40
# Convert the extracted data to a list
data_list = data.values.tolist()
data_list = np.array(data_list)
pT = np.array(data_list)[:,0].flatten()
nine = np.array(data_list)[:,1].flatten()

# changing dtype of pT to float
pT = pT[:lim_length].astype(float)
nine = nine[:lim_length].astype(float)

print(pT.shape)
print(nine.shape)
#print(pT)

#defining the bgbw
def boltzmann_gibbs_blast_wave(p_T, m, T, beta_s, R, n, norm):
    #defining the function integration
    def integrand(r, p_T, m_t, T, beta_s, n):
        '''
        input parameters
        - r : radial distance
        - p_T : transverse momentum array
        - m_t : transverse mass
        - T : temperature
        - beta_s : surface flow velocity
        - n : profile shape
        '''
        # transverse radial flow velocity
        beta = beta_s * (r / R) ** n
        #finding flow profile
        rho = np.arctanh(beta)
        #argument for modified bessel's function of 0th order
        arg1 = p_T * np.sinh(rho) / T
        #argument for modified bessel's function of 1st order
        arg2 = m_t * np.cosh(rho) / T
        return r * kn(1, arg2) * iv(0, arg1)
    #integrating to find transverse momenta
    dN_dpT = np.zeros_like(p_T)
    for i,p in enumerate(p_T):
        m_t = np.sqrt(m**2 + p**2)
        result, _ = quad(integrand, 0, R, args=(p, m_t, T, beta_s, n))
        dN_dpT[i] = result * m_t * norm
    return dN_dpT

from scipy.integrate import tplquad

#defining tbw
def tsallis_blast_wave(p_T, Y,  m, T, q, R, beta_s, n, norm):

    def integrand(r, phi, y, p_T, m_T, T, q, beta_s, n):
        # rho = np.arctanh(beta_s * ((r / R) ** n)) # radial beta
        rho = np.arctanh(beta_s * (1 + 1/(n + 1))) # average beta

        weight_factor = 1 + (((m_T * np.cosh(y) * np.cosh(rho)) - (p_T * np.sinh(rho) * np.cos(phi)))*((q - 1) / T))
        weight_factor = weight_factor ** (-1 / (q - 1))
        weight_factor = weight_factor * r * np.cosh(y)
        return weight_factor
    
    dN_dpT = np.zeros_like(p_T)
    for i,p in enumerate(p_T):
        m_T = np.sqrt(m**2 + p**2)
        #print(m_T)
        result, _ = tplquad(integrand, -Y, Y, -np.pi, np.pi, 0, R, args=(p, m_T, T, q, beta_s, n))
        # print(f'{result:.100f}')
        dN_dpT[i] = result * m_T  * norm

    return dN_dpT

m_0_pi = 0.13957
T_0 = 1.2676e-1
beta_s_0 = 4.387e-1
beta = 4.387e-1
n_0_bg = 0.5
R_0 = 0.01 #update
q_0 = 1.060
Y_0 = 0.5
n_0_ts = 1
norm_bg = 1e6
norm_ts = 1e6


#dN_dpT_bg = boltzmann_gibbs_blast_wave(pT, m_0, T_0, beta_s_0, R_0, n_0_bg, norm_bg)
#dN_dpT_ts = tsallis_blast_wave(pT, Y_0,  m_0, T_0, q_0, R_0, beta, n_0_ts, norm_ts)


### ---------- fitting the data to the models ---------- ###

# parameter bounds for boltzmann gibbs blast wave model
mass_bounds_bg = (m_0_pi, m_0_pi+1e-5)
temp_bounds_bg = (0, 1)
beta_bounds_bg = (0, 1)
n_0_bg_bounds_bg = (0, 2)
R_0_bounds_bg = (1e-5, 1)
norm_bg_bounds_bg = (0, 1e10)

all_bounds_bg = ([mass_bounds_bg[0], temp_bounds_bg[0], beta_bounds_bg[0], R_0_bounds_bg[0], n_0_bg_bounds_bg[0], norm_bg_bounds_bg[0]],[mass_bounds_bg[1], temp_bounds_bg[1], beta_bounds_bg[1], R_0_bounds_bg[1], n_0_bg_bounds_bg[1], norm_bg_bounds_bg[1]])

#parameter bounds for tsallis blast wave model
Y_0_bounds_ts = (0.5,0.5+1e-5)
mass_bounds_ts = (m_0_pi,m_0_pi+1e-5)
q_0_bounds_ts = (1.001,2)
temp_bounds_ts = (0.1,1)
R_0_bounds_ts = (1e-5, 1)
beta_bounds_ts = (0.00001, 1)
n_0_ts_bounds_ts = (1, 1+1e-5)
norm_ts_bounds_ts = (0, 1e10)

all_bounds_ts =( [Y_0_bounds_ts[0], mass_bounds_ts[0], temp_bounds_ts[0], q_0_bounds_ts[0], R_0_bounds_ts[0], beta_bounds_ts[0], n_0_ts_bounds_ts[0], norm_ts_bounds_ts[0]], [Y_0_bounds_ts[1], mass_bounds_ts[1], temp_bounds_ts[1], q_0_bounds_ts[1], R_0_bounds_ts[1], beta_bounds_ts[1], n_0_ts_bounds_ts[1], norm_ts_bounds_ts[1]])



print(f"fitting the data to the boltzmann gibbs blast wave model")
#popt, pcov = curve_fit(boltzmann_gibbs_blast_wave, pT, nine, p0=[m_0, T_0, beta_s_0, R_0, n_0_bg, norm_bg], bounds=all_bounds_bg)

# using least squares method
res_lsq = least_squares(lambda x: boltzmann_gibbs_blast_wave(pT, *x) - nine, [m_0_pi, T_0, beta_s_0, R_0, n_0_bg, norm_bg], bounds=all_bounds_bg, verbose=2)
popt = res_lsq.x

#printing the values for which the data is fitted:
print("Fitted Parameter Values for BGBW:")
for i, param_name in enumerate(["m_0", "T", "beta_s", "R", "n", "norm"]):
    print(f"{param_name}: {popt[i]}")

print(f"fitting the data to the tsallis blast wave model")
# popt2, pcov2 = curve_fit(tsallis_blast_wave, pT, nine, p0=[Y_0,  m_0, T_0, q_0, R_0, beta, n_0_ts, norm_ts],bounds=all_bounds_ts)

# using least squares method
iteration_limit = 15
res_lsq = least_squares(lambda x: tsallis_blast_wave(pT, *x) - nine, [Y_0,  m_0_pi, T_0, q_0, R_0, beta, n_0_ts, norm_ts], bounds=all_bounds_ts, verbose=2, max_nfev=iteration_limit)
popt2 = res_lsq.x

#printing the values for which the data is fitted:
print("Fitted Parameter Values for TBW:")
for i, param_name in enumerate(["Y_0", "m_0","T_0","q_0","R", "beta" , "n", "norm"]):
    print(f"{param_name}: {popt2[i]}")
### ---------- plotting the data and the fits ---------- ###

print(f"Getting the datapoints for the fits")
dN_dpT_bg = boltzmann_gibbs_blast_wave(pT, *popt)
dN_dpT_ts = tsallis_blast_wave(pT, *popt2)

# calculate the chi-squared values for the fits
chi2_bg = np.sum(((boltzmann_gibbs_blast_wave(pT, *popt) - nine)**2) / nine)
chi2_ts = np.sum(((tsallis_blast_wave(pT, *popt2) - nine)**2) / nine)


print(f'Chi-squared value for Boltzmann Gibbs Blast Wave fit: {chi2_bg}')
print(f'Chi-squared value for Tsallis Blast Wave fit: {chi2_ts}')

# plotter
print(f"Plotting the data and the fits")
plt.figure(figsize=(10, 6))
plt.plot(pT, nine, 'o', label='original data')
plt.plot(pT, dN_dpT_bg, '--', label=f'BGBW fit, chi2 = {chi2_bg}')
plt.plot(pT, dN_dpT_ts, '--', label=f'TBW fit, chi2 = {chi2_ts}')
plt.yscale('log')
plt.legend()
plt.xlabel('Transverse momentum (GeV)')
plt.ylabel('dN/dpT')
plt.title('Blast Wave Models')
plt.savefig('fits_goodresult2_Rchange.png')
plt.show()



# plotting the data and the fits
# plt.figure(figsize=(10, 6))
# plt.plot(pT, nine, 'o', label='original data')
# plt.plot(pT, dN_dpT_bg, '--', label='Boltzmann Gibbs Blast Wave datapoints')
# plt.plot(pT, dN_dpT_ts, '--', label='Tsallis Blast Wave datapoints')
# plt.plot(pT, np.exp(slope_bg*pT + intercept_bg), label=f'Boltzmann Gibbs Blast Wave fit, chi2 = {chi2_bg}')
# plt.plot(pT, np.exp(slope_ts*pT + intercept_ts), label=f'Tsallis Blast Wave fit, chi2 = {chi2_ts}')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Transverse momentum (GeV)')
# plt.ylabel('dN/dpT')
# plt.title('Blast Wave Models')
# plt.show()