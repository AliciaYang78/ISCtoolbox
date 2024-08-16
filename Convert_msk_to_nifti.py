#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:49:04 2022

@author: gangxinli
"""
import os
import numpy as np
import bvbabel as bv
import nibabel as nb

# def msk_to_nii(path):
#     print("Transfering msk file to nii file")
#     for root,dirs,files in os.walk(path):
#         for file in files:
#             print(file)
#             if file.split(".")[-1] == 'msk':
#                 FILE = os.path.join(root,file)
         
#                 header, data = bv.msk.read_msk(FILE)
#                 # Save nifti for testing
#                 basename = FILE.split(os.extsep, 1)[0]
#                 outname = "{}.nii".format(basename)
#                 # Export nifti (assign an identity matrix as affine with default header)
#                 n=header.get('Data type (1:short int, 2:float)')
#                 print("resolution:"+str(n))
#                 # img = nb.Nifti1Image(data,affine=np.eye(4)*n)
#                 img = nb.Nifti1Image(data, affine=np.eye(4))
#                 nb.save(img, outname)
#                 print(os.path.join(root,file)+"\nComplete")
#     print("Convert Complete!")
#     return path

# if __name__ =="__main__":
#     file_path = '/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCtoolbox/ISCanalysis/movie-data/mask-files'
#     msk_to_nii(file_path)


def msk_to_nii(filepath):
    # Get information from mask file
    header, data = bv.msk.read_msk(filepath)

    # Export to nifti
    basename = filepath.split(os.extsep, 1)[0]
    outname = "{}.nii.gz".format(basename)
    nii = nb.Nifti1Image(data, affine=np.eye(4))

    # Save
    nb.save(nii, outname)
    print("Conversion complete.")

if __name__ == "__main__":
    # Insert path to mask file you want to convert
    filepath = "/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCtoolbox/ISCanalysis/movie-data/mask-files/average_vtc_mask_BBR.msk"
    msk_to_nii(filepath)

