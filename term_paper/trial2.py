import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
from scipy.special import kn, iv
from scipy.integrate import quad


def tsallis_blast_wave(p_T, Y,  m, T, q, R, beta_s, n, norm, num_samples=1000):
    
    def integrand(r, phi, y, p_T, m_T, T, q, beta_s, n):
        rho = np.arctanh(beta_s * (1 + 1/(n + 1))) # average beta

        weight_factor = 1 + (((m_T * np.cosh(y) * np.cosh(rho)) - (p_T * np.sinh(rho) * np.cos(phi)))*((q - 1) / T))
        weight_factor = weight_factor ** (-1 / (q - 1))
        weight_factor = weight_factor * r * np.cosh(y)
        return weight_factor
    
    dN_dpT = np.zeros_like(p_T)
    for i, p in enumerate(p_T):
        m_T = np.sqrt(m**2 + p**2)
        
        samples = np.random.uniform(-Y, Y, size=(num_samples,))  # Sample y
        samples_phi = np.random.uniform(-np.pi, np.pi, size=(num_samples,))  # Sample phi
        samples_r = np.random.uniform(0, R, size=(num_samples,))  # Sample r
        
        integrand_values = integrand(samples_r, samples_phi, samples, p, m_T, T, q, beta_s, n)
        
        integral_approx = np.mean(integrand_values)
        dN_dpT[i] = integral_approx * m_T * norm

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

#Read the CSV file and extract data from the first column, skipping the first row
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

print(f"fitting the data to the tsallis blast wave model")
# popt2, pcov2 = curve_fit(tsallis_blast_wave, pT, nine, p0=[Y_0,  m_0, T_0, q_0, R_0, beta, n_0_ts, norm_ts],bounds=all_bounds_ts)

# using least squares method
iteration_limit = 15
res_lsq = least_squares(lambda x: tsallis_blast_wave(pT, *x) - nine, [Y_0,  m_0_pi, T_0, q_0, R_0, beta, n_0_ts, norm_ts], bounds=all_bounds_ts, verbose=2, max_nfev=iteration_limit)
popt2 = res_lsq.x

print(f"Getting the datapoints for the fits")
#dN_dpT_bg = boltzmann_gibbs_blast_wave(pT, *popt)
dN_dpT_ts = tsallis_blast_wave(pT, *popt2)

# calculate the chi-squared values for the fits
#chi2_bg = np.sum(((boltzmann_gibbs_blast_wave(pT, *popt) - nine)**2) / nine)
chi2_ts = np.sum(((tsallis_blast_wave(pT, *popt2) - nine)**2) / nine)

# plotter
print(f"Plotting the data and the fits")
plt.figure(figsize=(10, 6))
plt.plot(pT, nine, 'o', label='original data')
#plt.plot(pT, dN_dpT_bg, '--', label=f'BGBW fit, chi2 = {chi2_bg}')
plt.plot(pT, dN_dpT_ts, '--', label=f'TBW fit, chi2 = {chi2_ts}')
plt.yscale('log')
plt.legend()
plt.xlabel('Transverse momentum (GeV)')
plt.ylabel('dN/dpT')
plt.title('Blast Wave Models')
#plt.savefig('fits_goodresult2_Rchange.png')
plt.show()

