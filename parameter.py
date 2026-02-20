import numpy as np
from scipy import constants as cte
import sys
from pathlib import Path
import xraydb as xdb
import xraylib as xrl

# --- Grating -----------------------------------------------------------------

px_in_um = 5

# --- Geometry ----------------------------------------------------------------


# Simulated pixel siz in m
sim_pix_size_in_m = 1e-8
# Image size in pix
img_size_in_pix = int(round(300*1e-6/ sim_pix_size_in_m))
#img_size_in_pix = 64 * int(px_in_um * 1e-6 / sim_pix_size_in_m)
grating_periods =  img_size_in_pix/ (px_in_um * 1e-6 / sim_pix_size_in_m)

samp_size_in_m = img_size_in_pix * sim_pix_size_in_m*1e6

# Sample size in pix in X direction
#samp_size_in_pix = int(round(samp_size_in_m/ sim_pix_size_in_m))
samp_size_in_pix = img_size_in_pix
# Propagation distane in m. It is defined as the distance from the middle of

# the sample to the detector position


# --- Source ------------------------------------------------------------------

# Energy in keV
E_in_keV = 60

#E_in_J = E_in_keV * 1e3 * cte.e

#l_in_m = (px_in_um * 1e-6)**2 / (2*grat2det_in_m)
#E_in_keV = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / (l_in_m * 1e3)
# Number of photons per pixel
#num_ph = 1e5
# Wavelength in m 
l_in_m = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_in_keV * 1e3)  
# Wavevector magnitude in 1/m 
k_in_1_m = 2 * np.pi * (E_in_keV * 1e3) / \
           (cte.physical_constants["Planck constant in eV s"][0] * cte.c)

r_e = cte.physical_constants["classical electron radius"][0]  # in m

talbot_in_m = 2 * (px_in_um * 1e-6)**2 / l_in_m 
grat2det_in_m = 3/4 * talbot_in_m  
#print("grating to detector distance",grat2det_in_m)

prop_in_m = grat2det_in_m/2
#print("prop distance",prop_in_m)
# --- Sample ------------------------------------------------------------------

sim_approx = "slice" #"slice"
t_samp_in_mm = 0.5
#d_sph_in_um = 12 * px_in_um
d_sph_in_um = 100
#print("sphere diameter in um",d_sph_in_um)
mat_sph_type = "compound"
mat_bkg_type = "compound"


"""
mat_sph = "SiO2"
name_sph = "glass"
rho_sph_in_g_cm3 = 2.196


mat_bkg = "C2H6O"
name_bkg = "Ethanol"
rho_bkg_in_g_cm3 = 0.78945



mat_bkg = "H2O"
name_bkg = "Water"
rho_bkg_in_g_cm3 = 0.998
"""
name_bkg = "adipose tissue"
mat_bkg="H0.62536C0.27525N0.00276O0.09606Na0.00024S0.00017Cl0.00016"
rho_bkg_in_g_cm3 = 0.95 

name_sph = "Tumor cell"
electron_density = 3.54*1e29 # in electrons/m3
#mat_sph = "N0.78084O0.20946Ar0.00934C0.00036Ne0.000018He0.000005Kr0.000001" 
mat_sph = "cancer"
rho_sph_in_g_cm3 = 0
delta_sph = 2*np.pi * r_e * electron_density / (k_in_1_m**2)
delta_sph = 0

mu_bkg_in_1_m = xrl.CS_Total_CP(mat_bkg, E_in_keV) \
                                 * rho_bkg_in_g_cm3 * 100
intense_tish=np.exp(-mu_bkg_in_1_m*(t_samp_in_mm*1e-3))
intense = np.exp(-mu_bkg_in_1_m*(t_samp_in_mm*1e-3-d_sph_in_um*1e-6)-(0.212 * 100*d_sph_in_um*1e-6))
#print("intensity", intense_tish)
"""
mat_sph = "H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529" 
name_sph = "bone"
rho_sph_in_g_cm3 = 1.92 
"""


# Thickness of a sample slice, in pix. 
# Note: - For the projection approximation, t_slc = t_samp.
#       - For the thin slice approximation, t_slc = 1.4 um (diameter of the 
#         greatest simulated spheres).
if (sim_approx == "proj"):
    t_slc_in_pix = int(t_samp_in_mm * 1e-3 / sim_pix_size_in_m)
elif(sim_approx == "slice"):
    t_slc_in_pix = int(1 * 1e-6 / sim_pix_size_in_m)
else:
    raise ValueError("Please provide a valid simulation approximation")

# --- Detector ----------------------------------------------------------------
detector_pixel_size = 1*1e-6
binning_factor = int(detector_pixel_size/sim_pix_size_in_m)
segment_size_in_um = 5
segment_size_in_pix = int(round(segment_size_in_um * 1e-6 / sim_pix_size_in_m/binning_factor))
