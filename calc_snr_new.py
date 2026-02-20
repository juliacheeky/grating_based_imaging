import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from parameter import *
import xraylib as xrl

hight_of_pixel_in_um = 217
hight_of_pixel_in_pix = int(hight_of_pixel_in_um*1e-6 / detector_pixel_size)

additional_thickness_cm= 11.95
mu_bkg_in_1_cm = xrl.CS_Total_CP(mat_bkg, E_in_keV) \
                                 * rho_bkg_in_g_cm3

data  = "intensity_tumor_background_60keV_all_sizes_talbot3_large_sizes_onlyabsorb.csv"

d_sph = 200
col_name = f"{int(round(d_sph))}um"

'I could read data all at once here'

I_tumor = pd.read_csv(data)[col_name].to_numpy()
I_ref = pd.read_csv(data)["I_ref"].to_numpy()
I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()

def compute_intensity_2D_pixels(I_ref, I_tumor, I_no_tumor, d_sph_um):
    d_sph_in_pix = int(d_sph_um*1e-6 / detector_pixel_size)

    I_ref_2D_pixel = I_ref
    #print("sphere diameter in pix",d_sph_in_pix)

    I_tumor_2D_pixel = (hight_of_pixel_in_pix-d_sph_in_pix)*I_no_tumor + d_sph_in_pix*I_tumor
    I_tumor_2D_pixel = I_tumor_2D_pixel / hight_of_pixel_in_pix

    I_no_tumor_2D_pixel = I_no_tumor
    return I_ref_2D_pixel, I_tumor_2D_pixel, I_no_tumor_2D_pixel

def estimate_phi_mean_fourier_array(Iref, Isamp):
        # Will only work like this if the segment size goes evenly into the total length of the array,
        #  which is the case for our data
        N = Iref.shape[-1]
        Iref_reshape = Iref.reshape(...,int(N/segment_size_in_pix), segment_size_in_pix)
        Isamp_reshape = Isamp.reshape(...,int(N/segment_size_in_pix), segment_size_in_pix)
        
        fourier_Isamp = np.fft.fft(Isamp_reshape, axis=-1)
        fourier_Iref = np.fft.fft(Iref_reshape, axis=-1)
        k = int(N / (px_in_um*1e-6 / detector_pixel_size))
        
        mean_samp = fourier_Isamp[...,0].real / N
        phase_samp = np.angle(fourier_Isamp[...,k])
        phase_ref = np.angle(fourier_Iref[...,k])
        mean_ref = fourier_Iref[...,0].real / N

        means = mean_samp/mean_ref
        phis = phase_samp - phase_ref

        return phis, means


def estimate_phi_mean_fourier(Iref, Isamp):
        x_walk = np.arange(len(Iref))
        
        centers = []
        phi_list = []
        mean_list =[]

        for start in range(0, len(x_walk) - segment_size_in_pix + 1, segment_size_in_pix):
            #print(f"Processing segment starting at pixel {start}")
            x_seg = x_walk[start:start+segment_size_in_pix]
            Iref_seg = Iref[start:start+segment_size_in_pix]
            Isamp_seg = Isamp[start:start+segment_size_in_pix]
            fourier_Isamp = np.fft.fft(Isamp_seg)
            fourier_Iref = np.fft.fft(Iref_seg)
            N = len(Isamp_seg)
            k = int(N / (px_in_um*1e-6 / detector_pixel_size))

            mean_samp = fourier_Isamp[0].real / N
            #amp_samp = 2 * np.abs(fourier_Isamp[k]) / N
            phase_samp = np.angle(fourier_Isamp[k])

            mean_ref = fourier_Iref[0].real / N
            #amp_ref = 2 * np.abs(fourier_Iref[k]) / N
            phase_ref = np.angle(fourier_Iref[k])

            mean_list.append(mean_samp/mean_ref)
            centers.append(np.mean(x_seg))
            phi_list.append(phase_samp - phase_ref)

        return np.array(phi_list), np.array(mean_list)

def compute_total_phase(phi_tumor,phi_no_tumor):
    total_phi_tumor = np.cumsum(phi_tumor, axis=-1)
    total_phi_no_tumor = np.cumsum(phi_no_tumor, axis=-1)
    return total_phi_no_tumor, total_phi_tumor

def for_all_photons_new(I_ref, I_tumor, I_no_tumor,photon_start, photon_end, num_noise_realizations):

    photons = np.logspace(photon_start, photon_end, num=10, base=10.0, dtype=int)[:,np.newaxis]


    I_ref_noisy = np.random.poisson(np.broadcast_to(I_ref*photons, (num_noise_realizations,... )))
    I_tumor_noisy = np.random.poisson(np.broadcast_to(I_tumor*photons, (num_noise_realizations,... )))
    I_no_tumor_noisy = np.random.poisson(np.broadcast_to(I_no_tumor*photons, (num_noise_realizations,... )))
    
    phi_tumor, mean_tumor = estimate_phi_mean_fourier_array(I_ref_noisy, I_tumor_noisy)
    phi_no_tumor, mean_no_tumor = estimate_phi_mean_fourier_array(I_ref_noisy, I_no_tumor_noisy)

    total_phi_tumor = np.cumsum(phi_tumor, axis=-1)
    total_phi_no_tumor = np.cumsum(phi_no_tumor, axis=-1)

    return phi_tumor, phi_no_tumor, mean_tumor, mean_no_tumor, total_phi_tumor, total_phi_no_tumor


#Returns lists of phi, total phi and mean values for both tumor and no tumor cases for a range of photon counts
def for_all_photons(I_ref, I_tumor, I_no_tumor,photon_start, photon_end):

    photons_list = np.logspace(photon_start, photon_end, num=10, base=10.0, dtype=int)

    phi_tumor_results = []
    phi_no_tumor_results = []

    mean_no_tumor_results = []
    mean_tumor_results = []

    total_phi_no_tumor_results = []
    total_phi_tumor_results = []

    for photons in photons_list:
        I_ref_noisy = np.random.poisson(I_ref * photons)
        I_tumor_noisy = np.random.poisson(I_tumor * photons)
        I_no_tumor_noisy = np.random.poisson(I_no_tumor * photons)

        phi_tumor, mean_tumor = estimate_phi_mean_fourier(I_ref_noisy, I_tumor_noisy)
        phi_no_tumor, mean_no_tumor = estimate_phi_mean_fourier(I_ref_noisy, I_no_tumor_noisy)

        total_phi_no_tumor, total_phi_tumor = compute_total_phase(phi_tumor,phi_no_tumor)
        phi_tumor_results.append(phi_tumor)
        phi_no_tumor_results.append(phi_no_tumor)

        mean_no_tumor_results.append(mean_no_tumor)
        mean_tumor_results.append(mean_tumor)

        total_phi_no_tumor_results.append(total_phi_no_tumor)
        total_phi_tumor_results.append(total_phi_tumor)

    return np.array(phi_tumor_results), np.array(phi_no_tumor_results), np.array(mean_tumor_results), np.array(mean_no_tumor_results), np.array(total_phi_tumor_results), np.array(total_phi_no_tumor_results)

def calculate_cnr(tumor, no_tumor, d_sph):
    """
    Calculate CNR for the central region containing the sphere.
    
    Parameters:
    - phi_tumor: phase values with tumor
    - phi_no_tumor: phase values without tumor
    - d_sph: sphere diameter in micrometers
    """
    # Convert sphere diameter from micrometers to pixels
    d_sph_pix = d_sph / px_in_um
    
    # Extract central region
    arr_len = len(no_tumor)
    start_idx = int((arr_len - d_sph_pix) / 2)
    end_idx = int(start_idx + d_sph_pix)
    
    tumor_central = tumor[start_idx:end_idx]
    no_tumor_central = no_tumor[start_idx:end_idx]
    
    signal = np.abs(np.mean(tumor_central) - np.mean(no_tumor_central))
    print(f"Signal: {signal}")
    noise = np.std(no_tumor_central)
    print(f"Noise: {noise}")
    cnr = signal / noise if noise != 0 else 0
    return cnr

def calculate_cnr_whole_array(tumor, no_tumor, d_sph):
    """
    Calculate CNR for the central region containing the sphere.
    
    Parameters:
    - phi_tumor: phase values with tumor
    - phi_no_tumor: phase values without tumor
    - d_sph: sphere diameter in micrometers
    """
    # Convert sphere diameter from micrometers to pixels
    d_sph_pix = d_sph / px_in_um
    
    # Extract central region
    arr_len = no_tumor.shape[1]
    print("array length", arr_len)
    start_idx = int((arr_len - d_sph_pix) / 2)
    end_idx = int(start_idx + d_sph_pix)
    
    tumor_central = tumor[:, start_idx:end_idx]
    print("tumor central", tumor_central.shape)
    no_tumor_central = no_tumor[:, start_idx:end_idx]
    
    signal = np.abs(np.mean(tumor_central, axis=1) - np.mean(no_tumor_central, axis=1))
    print(f"Signal: {signal.shape}")
    noise = np.std(no_tumor_central, axis=1)
    print(f"Noise: {noise}")
    cnr = signal / noise
    return cnr

def calculate_cnr_alter(tumor, no_tumor, d_sph):
    signal = np.abs(np.mean(tumor) - np.mean(no_tumor))
    noise = np.std(no_tumor)
    cnr = signal / noise
    return cnr


all_cnr_results = []

for i in range(5):

    phi_tumor_results, phi_no_tumor_results, mean_tumor_results, mean_no_tumor_results, total_phi_tumor_results, total_phi_no_tumor_results = for_all_photons(I_ref, I_tumor, I_no_tumor, photon_start=3, photon_end=10)
    #print("phi tumor shape", np.array(phi_tumor_results).shape)
    cnr = calculate_cnr_whole_array(total_phi_tumor_results, total_phi_no_tumor_results, d_sph)
    #print(f"CNR for total phase: {cnr}")
    all_cnr_results.append(cnr)
print("all_cnr_results shape:", np.array(all_cnr_results).shape)
mean_cnr_results = np.mean(np.array(all_cnr_results), axis=0)
print(f"Mean CNR across iterations: {mean_cnr_results}")
"""
# Calculate mean CNR across iterations
mean_cnr_results = np.mean(all_cnr_results, axis=0)
std_cnr_results = np.std(all_cnr_results, axis=0)

plt.plot(total_phi_no_tumor_results[0], label='Total Phase No Tumor')
plt.plot(total_phi_tumor_results[0], label='Total Phase With Tumor')
plt.xlabel('Pixel Index')
plt.ylabel('Total Phase Difference')
plt.legend()
plt.show()
"""
"""

d_sphss = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
   
# Photon range to evaluate (same as used above)
photon_start = 3
photon_end = 10
photons_list = np.logspace(photon_start, photon_end, num=40, base=10.0, dtype=int)

# Container to store mean CNR for each sphere size (rows: sizes, cols: photon levels)
results_matrix = []

for d_sph in d_sphss:
    col_name = f"{int(round(d_sph))}um"
    I_tumor = pd.read_csv(data)[col_name].to_numpy()
    I_ref = pd.read_csv(data)["I_ref"].to_numpy()
    I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()
    I_tumor = I_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
    I_no_tumor = I_no_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
    I_ref_2D, I_tumor_2D, I_no_tumor_2D = compute_intensity_2D_pixels(I_ref, I_tumor, I_no_tumor, d_sph)
    all_cnr_results = []
    # Repeat realizations to average out Poisson noise
    for i in range(100):
        cnr_results = []
        phi_tumor_results, phi_no_tumor_results, mean_tumor_results, mean_no_tumor_results, total_phi_tumor_results, total_phi_no_tumor_results = \
            for_all_photons(I_ref_2D, I_tumor_2D, I_no_tumor_2D, photon_start=photon_start, photon_end=photon_end)

        for tot_phi_tumor, tot_phi_no_tumor in zip(total_phi_tumor_results, total_phi_no_tumor_results):
            cnr = calculate_cnr(tot_phi_tumor, tot_phi_no_tumor, d_sph)
            cnr_results.append(cnr)

        for mean_tumor, mean_no_tumor in zip(mean_tumor_results, mean_no_tumor_results):
            cnr = calculate_cnr_alter(mean_tumor, mean_no_tumor, d_sph)
            cnr_results.append(cnr)
            
        all_cnr_results.append(cnr_results)

    mean_cnr = np.mean(all_cnr_results, axis=0)
    std_cnr = np.std(all_cnr_results, axis=0)
    results_matrix.append(mean_cnr)

# Create DataFrame with photon counts as rows and sphere sizes as columns
df = pd.DataFrame(np.array(results_matrix).T, index=photons_list, columns=[f"{int(d)}um" for d in d_sphss])
output_csv = "cnr_vs_photons_mean_centre_section_12cm.csv"
df.index.name = 'photons'
df.to_csv(output_csv)
print(f"Saved CNR results to {output_csv}")
"""