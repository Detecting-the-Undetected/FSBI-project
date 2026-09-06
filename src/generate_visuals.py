import torch
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.esbi import ESBI_Dataset

# Fix for potential Matplotlib font issues on some systems
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def generate_visuals_for_report():
    print("Starting generation of professional report visuals...")
    
    # 1. Initialize Dataset
    # We use 'val' phase to get unseen data for the visuals
    dataset = ESBI_Dataset(phase='val', image_size=384, wavelet='sym2')
    
    # Get a sample from the dataset
    # index 10 often provides a clear face in FF++
    img_f_tensor, img_r_tensor, fake_type = dataset[10]
    
    # Convert tensors back to numpy for CV2/Matplotlib (C,H,W) -> (H,W,C)
    img_real = (img_r_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    img_fake = (img_f_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Use standard forensic labeling
    fake_label_text = 'FaceSwap (C1)' if fake_type == 1 else 'Reenactment (C2)'
    print(f"Generating visuals comparing Class 0 (Real) vs. Class {fake_type} ({fake_label_text})...")

    # --- TASK 1: FREQUENCY SPECTROGRAMS ---
    for img, label_id, label_text in [(img_real, '0', 'Real'), (img_fake, str(fake_type), fake_label_text)]:
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Perform 2D Discrete Wavelet Transform
        coeffs = pywt.dwt2(gray, 'sym2')
        LL, (LH, HL, HH) = coeffs

        # NEW: Using constrained_layout to prevent clipping
        fig = plt.figure(figsize=(24, 7), constrained_layout=True)
        gs = fig.add_gridspec(1, 4)
        
        # A. Original Spatial Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title(f'[{label_text}] Spatial Domain', fontweight='bold', pad=15)
        ax1.axis('off')

        freq_titles = ['Horizontal (LH)', 'Vertical (HL)', 'Diagonal (HH)']
        freq_images = [LH, HL, HH]
        
        # B. Frequency Sub-bands (with optimized titles and contrast)
        for i, (f_img, title) in enumerate(zip(freq_images, freq_titles)):
            ax = fig.add_subplot(gs[0, i+1])
            
            # Increase contrast for better report visibility
            f_img = np.abs(f_img)
            f_img = np.power(f_img, 0.75) # Slight gamma boost to see faint details
            f_img = (f_img - np.min(f_img)) / (np.max(f_img) - np.min(f_img))
            
            ax.imshow(f_img, cmap='magma')
            
            # NEW: Standardized, clear title including Class ID
            ax.set_title(f'[{label_text}] DWT {title}', fontweight='bold', pad=15)
            ax.axis('off')

        # NEW: Add a main figure title at the top to anchor everything
        fig.suptitle(f'Forensic Frequency Analysis: Class {label_id} ({label_text})', fontsize=20, fontweight='bold', y=1.02)
        
        # Save high-res PNG for the report
        save_path = f'report_Class{label_id}_DWT_Analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches extra insurance
        print(f" -> Saved frequency analysis to {save_path}")
        plt.close()
        
    

    # --- TASK 2: MASKING COMPARISON ---
    # We regenerate the side-by-side with professional labels
    h, w, _ = img_real.shape
    border = np.zeros((h, 15, 3), dtype=np.uint8) + 255 # White border
    
    comparison = np.hstack((img_real, border, img_fake))
    
    # Since we can't add text easily to an hstack JPG, we make a quick plot for it too
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(comparison)
    ax.axis('off')
    
    # NEW: Formal titles for the methodology section
    ax.set_title(f'Data Augmentation Overview: Real vs. SBI {fake_label_text}', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('report_SBI_Augmentation.jpg', dpi=300, bbox_inches='tight')
    print(" -> Saved SBI comparison to report_SBI_Augmentation.jpg")
    plt.close()
    
    

    print("\nAll report visuals have been saved to your current folder with professional formatting.")

if __name__ == "__main__":
    try:
        generate_visuals_for_report()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("\nFix Recommendation: Double-check that all paths are correct and dependencies are installed.")