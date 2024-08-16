# ISCtoolbox
Python toolbox for performing ISC analysis, based off work done by Gangxin Li: https://github.com/Gangxin-Li/ISCanalysis

# README
How to use the toolbox: 
1. Install bvbabel and nibabel at the following links: 

   https://github.com/ofgulban/bvbabel/tree/main 
   
   https://nipy.org/nibabel/installation.html

2. Convert participant (vtc format) and mask (msk format) files to nifti (nii) format using the conversion scripts
3. Check that the participant and mask files have the same dimensions
4. Use the isc_cli.py file to run the analysis using the following commands:
   python isc_cli.py --input AGA15_RUN5_MNI_3x1.nii.gz AKI10_RUN5_MNI_3x1.nii.gz --output isc_results.nii.gz --mask average_vtc_mask_BBR.nii.gz --zscore --fisherz
   
   python isc_cli.py --input *_RUN5_MNI_3x1.nii.gz --output isc_results.nii.gz --mask average_vtc_mask_BBR.nii.gz --zscore --fisherz
