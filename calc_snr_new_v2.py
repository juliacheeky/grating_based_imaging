import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from parameter import *

hight_of_pixel_in_um = 217
hight_of_pixel_in_pix = int(hight_of_pixel_in_um*1e-6 / detector_pixel_size)
file_name = "cnr_results.csv"

additional_thickness_cm = 11.85
mu_bkg_in_1_cm = xrl.CS_Total_CP(mat_bkg, E_in_keV) * rho_bkg_in_g_cm3

#data_phi  = "intensity_tumor_background_60keV_all_sizes_talbot3_large_sizes.csv"
#data_absorb = "intensity_tumor_background_60keV_all_sizes_talbot3_large_sizes_onlyabsorb.csv"

#data_phi  = "intensity_tumor_background_30keV_40-200_talbot3_phi.csv"
#data_absorb = "intensity_tumor_background_30keV_40-200_talbot3_only_absorb.csv"

#data_phi = "intensity_tumor_background_30keV_200-1000_talbot3_phi.csv"
data_phi = "intensity_tumor_background_60keV_40-1000_talbot3_phi.csv"
data_absorb = "intensity_tumor_background_30keV_200-1000_talbot3_only_absorb.csv"

#data_phi = "intensity_tumor_background_60keV_200-1000_talbot3_with_phase.csv"
#data_absorb = "intensity_tumor_background_60keV_200-1000_talbot3_only_absorb.csv"

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
        shape = Iref.shape
        N = shape[-1]

        
        print("N", segment_size_in_pix)
        Iref_reshape = Iref.reshape(shape[0],shape[1],int(N/segment_size_in_pix), segment_size_in_pix)
        Isamp_reshape = Isamp.reshape(shape[0],shape[1],int(N/segment_size_in_pix), segment_size_in_pix)
        
        fourier_Isamp = np.fft.fft(Isamp_reshape, axis=-1)
        fourier_Iref = np.fft.fft(Iref_reshape, axis=-1)
        print("fourier shape", fourier_Isamp.shape)
        k = int(segment_size_in_pix / (px_in_um*1e-6 / detector_pixel_size))
        print("k", k)
        
        mean_samp = fourier_Isamp[...,0].real / segment_size_in_pix
        phase_samp = np.angle(fourier_Isamp[...,k])
        phase_samp = np.where(np.abs(phase_samp)<np.pi, phase_samp, 0) # filter out any artifacts
        phase_ref = np.angle(fourier_Iref[...,k])
        phase_ref = np.where(np.abs(phase_ref)<np.pi, phase_ref, 0) # filter out any artifacts
        mean_ref = fourier_Iref[...,0].real / segment_size_in_pix

        means = mean_samp/mean_ref
        phis = phase_samp - phase_ref

        return phis, means

def for_all_photons_new(I_ref, I_tumor, I_no_tumor,photons, num_noise_realizations):
    photons_stacked = np.broadcast_to(photons[:, np.newaxis], (num_noise_realizations, len(photons), 1))
    I_ref_noisy = np.random.poisson(I_ref*photons_stacked)
    I_tumor_noisy = np.random.poisson(I_tumor*photons_stacked)
    I_no_tumor_noisy = np.random.poisson(I_no_tumor*photons_stacked)
    
    phi_tumor, mean_tumor = estimate_phi_mean_fourier_array(I_ref_noisy, I_tumor_noisy)
    phi_no_tumor, mean_no_tumor = estimate_phi_mean_fourier_array(I_ref_noisy, I_no_tumor_noisy)

    total_phi_tumor = np.cumsum(phi_tumor[:,:,1:-1], axis=-1)
    total_phi_no_tumor = np.cumsum(phi_no_tumor[:,:,1:-1], axis=-1)
    'phi_tumor dimention: (num_noise_realisations, num_photon_levels, num_segments)'
    return phi_tumor, phi_no_tumor, mean_tumor, mean_no_tumor, total_phi_tumor, total_phi_no_tumor

def calculate_cnr_whole_array(tumor, no_tumor, d_sph):

    d_sph_in_segs = 40 #d_sph / segment_size_in_pix
 
    arr_len = no_tumor.shape[-1]
    start_idx = int((arr_len - d_sph_in_segs) / 2)
    end_idx = int(start_idx + d_sph_in_segs)
    
    tumor_central = tumor[..., start_idx:end_idx]
    no_tumor_central = no_tumor[..., start_idx:end_idx]
    
    signal = np.abs(np.mean(tumor_central, axis=-1) - np.mean(no_tumor_central, axis=-1))
    print(f"Signal: {signal.shape}")
    noise = np.std(no_tumor_central, axis=-1, ddof=1)
    print(f"Noise: {noise}")
    cnr = signal / noise
    return cnr

#d_sphss = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
#d_sphss = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
d_sphss = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200, 300, 400, 500, 600, 700, 800, 900, 1000]
photon_start = 2
photon_end = 8
photons = np.logspace(photon_start, photon_end, num=40, base=10.0, dtype=int)

def process(input_file, output_file, phi):
    Intensities = pd.read_csv(input_file)
    I_ref = Intensities["I_ref"].to_numpy()
    I_no_tumor = Intensities["I_no_tumor"].to_numpy()
    I_no_tumor = I_no_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)

    for d_sph in d_sphss:

        col_name = f"{int(round(d_sph))}um"
        I_tumor = Intensities[col_name].to_numpy()
        I_tumor = I_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
        
        #I_ref_2D, I_tumor_2D, I_no_tumor_2D = compute_intensity_2D_pixels(I_ref, I_tumor, I_no_tumor, d_sph)
        phi_tumor_results, phi_no_tumor_results, mean_tumor_results, mean_no_tumor_results, total_phi_tumor_results, total_phi_no_tumor_results = \
            for_all_photons_new(I_ref, I_tumor, I_no_tumor, photons, num_noise_realizations=100)
        cnr = calculate_cnr_whole_array(total_phi_tumor_results, total_phi_no_tumor_results, d_sph) \
                if phi else calculate_cnr_whole_array(mean_tumor_results, mean_no_tumor_results, d_sph)
        mean_cnr = np.mean(cnr, axis=0) # dimensions (photons)

        if not os.path.exists(output_file):
            # File does NOT exist — create it
            results = pd.DataFrame({
            "photons": photons,
            f"SDNR_{int(round(d_sph))}um": mean_cnr
            })
            results.to_csv(output_file, index=False)
        else:
            # File exists — load and add new column
            results = pd.read_csv(output_file)
            
            # Add new column (ensure lengths match)
            results[f"SDNR_{int(round(d_sph))}um"] = mean_cnr
            
            results.to_csv(output_file, index=False)

process(data_phi, "cnr_phi_40-1000_60keV.csv", True)
#process(data_absorb, "cnr_absorb_200-1000_30keV.csv", False)
