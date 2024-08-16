#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:49:04 2022

@author: aliciayang
"""

import numpy as np
import bvbabel as bv
import nibabel as nb

# Get necessary information from input files
def get_input_details(input):

    header, data = bv.vtc.read_vtc(input)

    return (header['XStart'], header['XEnd'], header['YStart'], header['YEnd'], header['ZStart'], header['ZEnd'], 
            header['VTC resolution relative to VMR (1, 2, or 3)'], header['Nr time points'])

# Create and write vmp (edited from bvbabel)
def create_vmp(input_info, ISC_results_path, vmp_filename):

    XStart, XEnd, YStart, YEnd, ZStart, ZEnd, Resolution, DF = input_info

    header = dict()
    # -------------------------------------------------------------------------
    # NR-VMP Header (Version 6)
    # -------------------------------------------------------------------------

    # Expected binary data: int (4 bytes)
    header["NR-VMP identifier"] = np.int32(-1582119980)

    # Expected binary data: short int (2 bytes)
    header["VersionNumber"] = np.int16(6)
    header["DocumentType"] = np.int16(1)

    # Expected binary data: int (4 bytes)
    header["NrOfSubMaps"] = np.int32(1)
    header["NrOfTimePoints"] = np.int32(0)
    header["NrOfComponentParams"] = np.int32(0)
    header["ShowParamsRangeFrom"] = np.int32(0)
    header["ShowParamsRangeTo"] = np.int32(0)
    header["UseForFingerprintParamsRangeFrom"] = np.int32(0)
    header["UseForFingerprintParamsRangeTo"] = np.int32(0)
    header["XStart"] = np.int32(XStart)
    header["XEnd"] = np.int32(XEnd)
    header["YStart"] = np.int32(YStart)
    header["YEnd"] = np.int32(YEnd)
    header["ZStart"] = np.int32(ZStart)
    header["ZEnd"] = np.int32(ZEnd)
    header["Resolution"] = np.int32(Resolution)
    header["DimX"] = np.int32(256)
    header["DimY"] = np.int32(256)
    header["DimZ"] = np.int32(256)

    # Expected binary data: variable-length string
    header["NameOfVTCFile"] = ""
    header["NameOfProtocolFile"] = ""
    header["NameOfVOIFile"] = ""

    # -------------------------------------------------------------------------
    # Map information
    # -------------------------------------------------------------------------
    header["Map"] = list()
    header["Map"].append(dict())

    # Expected binary data: int (4 bytes)
    header["Map"][0]["TypeOfMap"] = np.int32(2)

    # Expected binary data: float (4 bytes)
    header["Map"][0]["MapThreshold"] = np.float32(0.11000137031078339)
    header["Map"][0]["UpperThreshold"] = np.float32(0.6000000238418579)

    # Expected binary data: variable-length string
    header["Map"][0]["MapName"] = "<averaged>"

    # Expected binary data: char (1 byte) x 3
    header["Map"][0]["RGB positive min"] = np.array([254, 236, 153], dtype=np.uint8)
    header["Map"][0]["RGB positive max"] = np.array([145,   0,  37], dtype=np.uint8)
    header["Map"][0]["RGB negative min"] = np.array([224, 243, 248], dtype=np.uint8)
    header["Map"][0]["RGB negative max"] = np.array([ 40,  51, 144], dtype=np.uint8)

    # Expected binary data: char (1 byte)
    header["Map"][0]["UseVMPColor"] = np.byte(0)

    # Expected binary data: variable-length string
    header["Map"][0]["LUTFileName"] = "<default>"

    # Expected binary data: float (4 bytes)
    header["Map"][0]["TransparentColorFactor"] = np.float32(1.0)

    # Expected binary data: int (4 bytes)
    header["Map"][0]["ClusterSizeThreshold"] = np.int32(4)

    # Expected binary data: char (1 byte)
    header["Map"][0]["EnableClusterSizeThreshold"] = np.byte(0)

    # Expected binary data: int (4 bytes)
    header["Map"][0]["ShowValuesAboveUpperThreshold"] = np.int32(1)
    header["Map"][0]["DF1"] = np.int32(DF-2)
    header["Map"][0]["DF2"] = np.int32(0)

    # Expected binary data: char (1 byte)
    header["Map"][0]["ShowPosNegValues"] = np.byte(3)

    # Expected binary data: int (4 bytes)
    header["Map"][0]["NrOfUsedVoxels"] = np.int32(122712)
    header["Map"][0]["SizeOfFDRTable"] = np.int32(0)

    # Expected binary data: float (4 bytes) x SizeOfFDRTable x 3
    # (q, crit std, crit conservative)
    # TODO: Check FDR Tables
    header["Map"][0]["FDRTableInfo"] = np.empty((0, 3), dtype=np.float64)

    # Expected binary data: int (4 bytes)
    header["Map"][0]["UseFDRTableIndex"] = np.int32(-1)

    # -------------------------------------------------------------------------
    # Write VMP file
    # -------------------------------------------------------------------------
    img = nb.load(ISC_results_path)
    data = img.get_fdata()

    # Code to edit ISC data format (may not always be necessary)
    new_data = np.zeros((data.shape[:3]))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(len(data[0][0])):
                new_data[i][j][k] = data[i][j][k][0]

    bv.vmp.write_vmp(vmp_filename, header, new_data)
    print("Conversion complete.")

if __name__ == "__main__":

    # Insert path to input/participant vtc file being analyzed
    input_filepath = "/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCtoolbox/ISCanalysis/movie-data/hide-input/AGA15_RUN5_MNI_3x1.0MM_new.vtc"
    # Insert path to ISC analysis results file to convert to vmp
    ISC_filepath = "/Users/Alici/Documents/College/MIT/2023-2024/MISTI UK/Project Materials/ISCtoolbox/ISCanalysis/movie-data/isc_20.nii.gz"
    # Type in name of new vmp file
    vmp_filename = "new_isc_vmp_test.vmp"
    
    input_info = get_input_details(input_filepath)
    create_vmp(input_info, ISC_filepath, vmp_filename)