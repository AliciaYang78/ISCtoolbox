#!/usr/bin/env python3

import sys
import os
import argparse
import logging
import textwrap
from glob import glob
import numpy as np
import nibabel as nb
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.stats import norm
from scipy.stats import false_discovery_control
from scipy.interpolate import interp1d
import time


# Set up logger first
logger = logging.getLogger(__name__)


# Set up argument parser
def parse_arguments(args):

    parser = argparse.ArgumentParser(
        description=("Python-based command-line program for computing "
                     "leave-one-out intersubject correlations (ISCs)"),
        epilog=(textwrap.dedent("""
    This program provides a simple Python-based command-line interface (CLI)
    for running intersubject correlation (ISC) analysis. ISCs are computed
    using the leave-one-out approach where each subject's time series (per
    voxel) is correlated with the average of the remaining subjects' time
    series. The --input should be two or more 4-dimensional NIfTI (.nii or
    .nii.gz) files, one for each subject. Alternatively, a wildcard can be used
    to indicate multiple files (e.g., *.nii.gz). The --output path and filename
    will be treated differently depending on if output summarization (the
    --summarize argument) is used. If no --summarize argument is provided, the
    output filename will be treated as a suffix and appended to each input
    filename (with an underscore). If the --summarize argument is provided, the
    output will be single file saved to the filename supplied to --ouotput. In
    either case, a full output path can be provided as well. For example,
    consider the case where no summarization is used and more than two input
    files are supplied; if the output argument is --ouput /out_path/isc, input
    files resembling /in_path/s1.nii.gz, /in_path/s2.nii.gz, /in_path/s3.nii.gz
    will yield corresponding ouput files named /out_path/s1_isc.nii.gz and so
    on. The --summarize argument can be used to compute the mean or median ISC
    value across subjects after completing the ISC analysis (in which case the
    output file will only have one volume), or to stack the output ISC values
    for each subject into a single output file. If mean ISC values are
    requested, ISC values are Fisher z-transformed, the mean is computed, and
    then the mean is inverse Fisher z-transformed. If N subjects are input into
    the ISC analysis and --summarize stack is used, the resulting output file
    will have N volumes (one ISC value for each left-out subject). If only two
    input files a provided, a single output file is returned. Typically a
    3-dimensional NIfTI file should be supplied to the --mask argument so as to
    restrict the analysis to voxels of interest (e.g., the brain, gray matter).
    The mask file will be converted to a Boolean array and should have 1s for
    voxels of interest and 0s elsewhere. All input files (and the mask) must be
    spatially normalized to standard space (e.g., MNI space) prior to ISC
    analysis. The --zscore argument indicates that response time series should
    be z-scored (per voxel) prior to ISC analysis; this may be important when
    computing the average time series for N–1 subjects. The --fisherz argument
    can be used to apply the Fisher z-transformation (arctanh) to the output
    ISC values (e.g., for subsequent statistical tests). This program requires
    an installation of Python 3 with the NumPy/SciPy and NiBabel modules. The
    implementation is based on the BrainIAK (https://brainiak.org)
    implementation, but does not require a BrainIAK installation. Note that
    this software is not written for speed or memory-efficiency, but for
    readability/transparency.

    Example usage:
        python3 isc_cli.py --input s1.nii.gz s2.nii.gz s3.nii.gz \\
        --output isc --mask mask.nii.gz --zscore --fisherz

        python3 isc_cli.py --input s*.nii.gz --output iscs.nii.gz \\
        --mask mask.nii.gz --zscore --summarize stack

        python3 isc_cli.py --input s*.nii.gz --output mean_isc.nii.gz \\
        --mask mask.nii.gz --zscore --summarize mean

    References:
    Nastase, S. A., Gazzola, V., Hasson, U., Keysers, C. (in preparation).
    Measuring shared responses across subjects using intersubject correlation.

    Author: Samuel A. Nastase, 2019
        """)), formatter_class=argparse.RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required_arguments')
    required.add_argument("-i", "--input", nargs='+', required=True,
                          help=("NIfTI input files on which to compute ISCs"))
    required.add_argument("-o", "--output", type=str, required=True,
                          help=("suffix or filename for NIfTI output(s) "
                                "containing ISC values"))
    optional.add_argument("-m", "--mask", type=str,
                          help=("NIfTI mask file for masking input data"))
    optional.add_argument("-s", "--summarize", type=str,
                          choices=['mean', 'median', 'stack'],
                          help=("summarize results across participants "
                                "using either mean, median, or stack"))
    optional.add_argument("-z", "--zscore", action='store_true',
                          dest='apply_zscore',
                          help=("Z-score time series for each voxel in input"))
    optional.add_argument("-f", "--fisherz", action='store_true',
                          dest='apply_fisherz',
                          help=("apply Fisher z-transformation (arctanh) "
                                "to output ISC values"))
    optional.add_argument("-v", "--verbosity", type=int,
                          choices=[1, 2, 3, 4, 5], default=3,
                          help=("increase output verbosity via "
                                "Python's logging module"))
    parser.add_argument('--version', action='version',
                        version='isc_cli.py 1.0.0')
    parser._action_groups.append(optional)
    args = parser.parse_args(args)

    return args


# Function to load NIfTI mask and convert to boolean
def load_mask(mask_arg):

    # Load mask from file
    mask = nb.load(mask_arg).get_fdata().astype(bool)

    # Get indices of nonzero voxels in mask
    mask_indices = np.asarray(mask).nonzero()

    logger.info("finished loading mask from "
                "'{0}'".format(mask_arg))
                
    return mask, mask_indices


# Function for loading (and organizing) data from NIfTI files
def load_data(input_arg, mask=None):

    # Convert input argument string to filenames
    input_fns = [fn for fns in [glob(fn) for fn in input_arg]
                 for fn in fns]

    # Check if not enough input filenames
    if len(input_fns) < 2:
        raise ValueError("--input requires two or more "
                         "input files for ISC computation")

    data = []
    affine, data_shape = None, None
    for input_fn in input_fns:

        # Load in the NIfTI image using NiBabel and check shapes
        input_img = nb.load(input_fn)
        if not data_shape:
            data_shape = input_img.shape
            shape_fn = input_fn
        if len(input_img.shape) != 4:
            raise ValueError("input files should be 4-dimensional "
                             "(three spatial dimensions plus time)")
        if input_img.shape != data_shape:
            raise ValueError("input files have mismatching shape: "
                             "file '{0}' with shape {1} does not "
                             "match file '{2}' with shape "
                             "{3}".format(input_fn,
                                          input_img.shape,
                                          shape_fn,
                                          data_shape))
        logger.debug("input file '{0}' NIfTI image is "
                     "shape {1}".format(input_fn, data_shape))

        # Save the affine and header from the first image
        if affine is None:
            affine, header = input_img.affine, input_img.header
            logger.debug("using affine and header from "
                         "file '{0}'".format(input_fn))

        # Get data from image and apply mask (if provided)
        input_data = input_img.get_fdata()
        if isinstance(mask, np.ndarray):
            input_data = input_data[mask]
        else:
            input_data = input_data.reshape((
                np.product(input_data.shape[:3]),
                input_data.shape[3]))
        data.append(input_data.T)
        logger.info("finished loading data from "
                    "'{0}'".format(input_fn))

    # Stack input data
    data = np.stack(data, axis=2)

    return data, affine, header, input_fns


# # Function to efficiently compute correlations across voxels
# def array_correlation(x, y, axis=0):

#     # Accommodate array-like inputs
#     if not isinstance(x, np.ndarray):
#         x = np.asarray(x)
#     if not isinstance(y, np.ndarray):
#         y = np.asarray(y)

#     # Check that inputs are same shape
#     if x.shape != y.shape:
#         raise ValueError("Input arrays must be the same shape")

#     # Transpose if axis=1 requested (to avoid broadcasting
#     # issues introduced by switching axis in mean and sum)
#     if axis == 1:
#         x, y = x.T, y.T

#     # Center (de-mean) input variables
#     x_demean = x - np.mean(x, axis=0)
#     y_demean = y - np.mean(y, axis=0)

#     # Compute summed product of centered variables
#     numerator = np.sum(x_demean * y_demean, axis=0)

#     # Compute sum squared error
#     denominator = np.sqrt(np.sum(x_demean ** 2, axis=0) *
#                           np.sum(y_demean ** 2, axis=0))

#     return numerator / denominator


# # Function to compute leave-one-out ISCs on input data
# def compute_iscs(data):

#     # Get shape of data
#     n_TRs, n_voxels, n_subjects = data.shape

#     # Check if only two subjects
#     if n_subjects == 2:
#         logger.warning("only two subjects provided! simply "
#                        "computing ISC between them")

#         # Compute correlation for each corresponding voxel
#         iscs = array_correlation(data[..., 0],
#                                  data[..., 1])[np.newaxis, :]

#     # Loop through left-out subjects
#     else:
#         iscs = []
#         for s in np.arange(n_subjects):

#             # Correlation between left-out subject and mean of others
#             iscs.append(array_correlation(
#                 data[..., s],
#                 np.nanmean(np.delete(data, s, axis=2),
#                            axis=2))[np.newaxis, :])

#     logger.info("finished computing ISCs")
#     return iscs

#Function correlation coefficient (r) between 2 data sets
def find_r_value(x,y):
    idxs = []
    for idx in range(len(x)): # find indices with NaN data
        if np.isnan(x[idx]) or np.isnan(y[idx]):
            idxs.append(idx)
    idxs.sort(reverse=True)
    for elt in idxs: #remove NaN data pairs
        x.pop(elt)
        y.pop(elt)
    if len(x) < 2: # pearson correlation can only be done with more than 2 elts
        return 0
    else:
        r_stat = pearsonr(np.array(x),np.array(y)).statistic
        return r_stat

# Function to compute ISCs on input data
def compute_iscs(data):

    start_time_2 = time.time()
    
    # Get shape of data
    n_timepoints, n_voxels, n_subjects = data.shape
    
    iscs = []
    for v in range(n_voxels): #get data for that voxel
        total = 0
        for i in range(0,n_subjects-1):
            for j in range(i+1,n_subjects): #get data for each pair of subjects over time
                x = list(data[:,v,i])
                y = list(data[:,v,j])
                total += find_r_value(x,y)
        corr_coeff = total/(0.5*n_subjects*(n_subjects-1))
        iscs.append(corr_coeff)
    
    elapsed_time_2 = time.time() - start_time_2
    print(f"Time for ISC analysis with {n_subjects} subjects: {elapsed_time_2}")

    logger.info("finished computing ISCs")
    return np.array([iscs])

def apply_FDR_corr(data, iscs_array,total_trials):
    n_timepoints, n_voxels, n_subjects = data.shape
    
    start_time_3 = time.time()

    # bootstrap sampling 
    all_coeffs = []
    for _ in range(total_trials):
        total = 0
        voxel_num = np.random.randint(0, n_voxels)
        all_data = []
        for j in range(n_subjects):
            shift = np.random.randint(0, n_timepoints)
            subj_data = data[:,voxel_num, j]
            shifted_data = np.append(subj_data[shift:], subj_data[:shift])
            all_data.append(shifted_data)
        for k in range(0, n_subjects-1):
            for l in range(k+1,n_subjects):
                x = list(all_data[k])
                y = list(all_data[l])
                total += find_r_value(x,y)
        corr_coeff = total/(0.5*n_subjects*(n_subjects-1))
        all_coeffs.append(corr_coeff)

    elapsed_time_3 = time.time() - start_time_3
    print(f"Time for boostrap sampling with {total_trials} trials: {elapsed_time_3}")

    # running FDR
    start_time_4 = time.time()

    samp_mean = np.mean(all_coeffs)
    samp_sd = np.std(all_coeffs)
    z_scores = (iscs_array[0] - samp_mean)/samp_sd
    p_values = norm.sf(abs(z_scores))*2
    adj_p_values = false_discovery_control(p_values)

    # find corresponding thresholds
    f = interp1d(adj_p_values, iscs_array[0])
    new_th_05 = f(0.05)
    new_th_01 = f(0.01)
    new_th_001 = f(0.001)

    elapsed_time_4 = time.time() - start_time_4
    print(f"Time for FDR: {elapsed_time_4} \n")

    print(f"Threshold for p = 0.05 is r-value of {new_th_05}")
    print(f"Threshold for p = 0.01 is r-value of {new_th_01}")
    print(f"Threshold for p = 0.001 is r-value of {new_th_001}")

    return None

# Function to optionally summarize ISCs
def summarize_iscs(iscs, summary):

    # Compute mean (with Fisher Z transformation)
    if summary == 'mean':
        summarized = np.tanh(np.nanmean(np.arctanh(np.vstack(iscs)),
                                        axis=0))[np.newaxis, :]
        logger.info("computing mean of ISCs (with "
                    "Fisher Z transformation)")

    # Compute median
    elif summary == 'median':
        summarized = np.nanmedian(np.vstack(iscs), axis=0)[np.newaxis, :]
        logger.info("computing median of ISCs")

    # Vertically stack into single array
    elif summary == 'stack':
        summarized = np.vstack(iscs)
        logger.info("stacking output ISCs")

    return summarized


# Function to convert array back to NIfTI image
def to_nifti(iscs, affine, header, mask_indices):

    # Output ISCs image shape
    i, j, k = header.get_data_shape()[:3]
    nifti_shape = (i, j, k, iscs.shape[0])

    # Reshape masked data
    if mask_indices:
        nifti_iscs = np.zeros(nifti_shape)
        nifti_iscs[mask_indices] = iscs.T
    else:
        nifti_iscs = iscs.T.reshape(nifti_shape)

    # Construct output NIfTI image
    nifti_img = nb.Nifti1Image(nifti_iscs, affine)

    return nifti_img


# Function to save NIfTI images
def save_data(iscs, affine, header, input_fns,
              output_fn, mask_indices=None):

    # Save output ISCs for each input subject
    if type(iscs) == list:

        # Loop through input files and output ISCs
        for input_fn, isc in zip(input_fns, iscs):
            output_img = to_nifti(isc, affine, header,
                                  mask_indices)

            # Get input filename from path
            input_fn = os.path.basename(input_fn)

            # Split NIfTI extension off input filename
            input_prefix = input_fn.split('.nii')[0]

            # Split output path into directory and filename
            output_dir, output_suffix = os.path.split(output_fn)

            # Get filename suffix excluding NIfTI extension
            output_suffix = output_suffix.split('.nii')[0]
            output_merged = os.path.join(
                output_dir,
                '{0}_{1}.nii.gz'.format(input_prefix, output_suffix))

            # Save the NIfTI image according to output filename
            nb.save(output_img, output_merged)

            logger.info("saved ISC output to {0}".format(output_merged))

    # Save summarized (or two-subject) ISCs
    else:

        # Construct output NIfTI image
        output_img = to_nifti(iscs, affine, header,
                              mask_indices)

        # Check for NIfTI extension
        if '.nii' not in output_fn:
            output_fn = output_fn + '.nii.gz'

        # Save the NIfTI image according to output filename
        nb.save(output_img, output_fn)

        logger.info("saved ISC output to {0}".format(output_fn))


# Function to execute the above code
def main(args):

    start_time_1 = time.time()
    # Get arguments
    args = parse_arguments(args)

    # Set up logger according to verbosity level
    logging.basicConfig(level=abs(6 - args.verbosity) * 10)
    logger.info("verbosity set to Python logging level '{0}'".format(
        logging.getLevelName(logger.getEffectiveLevel())))

    # Get optional mask
    if args.mask:
        mask, mask_indices = load_mask(args.mask)
    else:
        mask, mask_indices = None, None
        print("Computing ISCs without mask")
       # logger.warning("no mask provided! are you sure you want to compute ISCs for all voxels in image?")
      
    # Load data
    data, affine, header, input_fns = load_data(args.input, mask=mask)

    # Optionally z-score data
    if args.apply_zscore:
        data = zscore(data, axis=0)
        logging.info("z-scored input data prior to computing ISCs")
    
    elapsed_time_1 = time.time() - start_time_1
    print(f"Time for initial setup: {elapsed_time_1}")

    # Compute ISCs and z scores
    iscs = compute_iscs(data)

    # Calculate FDR
    total_trials = 100000
    apply_FDR_corr(data, iscs, total_trials)
    
    # Optionally apply summary statistic
    if args.summarize and len(iscs) > 1:
        iscs = summarize_iscs(iscs,
                              summary=args.summarize)

    # Optinally apply Fisher z-transformation to output ISCs
    if args.apply_fisherz:
        if type(iscs) == list:
            iscs = [np.arctanh(isc) for isc in iscs]

        else:
            iscs = np.arctanh(iscs)

    # Save output ISCs to file
    save_data(iscs, affine, header, input_fns,
              args.output, mask_indices=mask_indices)


# Name guard so we can load these functions elsewhere
# without actually trying to run everything
if __name__ == '__main__':
    main(sys.argv[1:])
