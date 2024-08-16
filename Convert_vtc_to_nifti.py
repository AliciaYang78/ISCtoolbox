#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 01:52:21 2022

@author: gangxinli
"""
import os
import numpy as np
import nibabel as nb
import bvbabel as bv
# from Hook_header import *

# def vtc_to_nii(path, header_path):
#     print("Transfering vtc file to nii file")
#     for root,dirs,files in os.walk(path):
#         for file in files:
#             if file.split(".")[-1] == 'vtc':
#                 FILE = os.path.join(root,file)
         
#                 header, data = vtc.read_vtc(FILE)
#                 # Save nifti for testing
#                 basename = FILE.split(os.extsep, 1)[0]
#                 outname = "{}.nii".format(basename)
                
#                 # Hook_header test
#                 write_header(header=header, path=header_path)
#                 # 
                
                
#                 # Export nifti (assign an identity matrix as affine with default header)
#                 # np.eye(4)*n, n control the resoultion
#                 n=header.get('Data type (1:short int, 2:float)')
#                 print("resolution:"+str(n))
#                 img = nb.Nifti1Image(data,affine=np.eye(4)*n)
#                 # img = nb.Nifti1Image(data,affine=np.eye(4))
#                 nb.save(img, outname)
#                 print(os.path.join(root,file)+"\nComplete")
#     print("Convert Complete!")
#     return path

# if __name__ =="__main__":
#     file_path = '/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCanalysis/movie-data'
#     header_path = '/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCanalysis/Hook_header.py'
#     vtc_to_nii(file_path, header_path)

def vtc_to_nii(filepath):

    # Load vtc
    header, data = bv.vtc.read_vtc(filepath, rearrange_data_axes=False)

    # Transpose axes
    data = np.transpose(data, [0, 2, 1, 3])
    # Flip axes
    data = data[::-1, ::-1, ::-1, :]

    # Export nifti
    basename = filepath.split(os.extsep, 1)[0]
    outname = "{}.nii.gz".format(basename)
    img = nb.Nifti1Image(data, affine=np.eye(4))
    nb.save(img, outname)

    print("Conversion complete.")

if __name__ == "__main__":
    # Insert path to vtc file you want to convert
    filepath = "/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCanalysis/movie-data/XHN30_RUN5_MNI_3x1.0MM_new.vtc"
    vtc_to_nii(filepath)


