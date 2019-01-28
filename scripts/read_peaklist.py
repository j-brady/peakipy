#!/usr/bin/env python3
""" Read NMRPipe/Analysis peaklist into pandas dataframe

    Usage:
        read_peaklist.py <peaklist> <data> [options]

    Arguments:
        <peaklist>  Analysis2/Sparky/NMRPipe peak list (see below)
        <data>      2D or pseudo3D NMRPipe data

    Options:
        -h --help  Show this screen
        --version  Show version
 
        --noise=<noise>     Noise of spectrum [default: 5e4]
        --pthres=<pthres>   Positive peakpick threshold [default: 10e5]
        --nthres=<nthres>   Negative peakpick threshold [default: None]
        --ndil=<ndil>       Number of iterations for ndimage.binary_dilation [default: 0]
        
        --dims=<planes,F1,F2>    Order of dimensions [default: 0,1,2]

        --a2      Analysis peaklist as input
        --sparky  Sparky peaklist as input
        --pipe    NMRPipe peaklist as input

        --outfmt=<csv/pkl>  Format of output peaklist [default: "pkl"]
 
        --show    Show the clusters on the spectrum color coded

    Examples:
        read_peaklist.py test.tab
        read_peaklist.py test.a2 test.ft2 --a2 --pthres=1e5 --noise=1e4 --dims=0,2,1

    Description:
      
       NMRPipe column headers:

           INDEX X_AXIS Y_AXIS DX DY X_PPM Y_PPM X_HZ Y_HZ XW YW XW_HZ YW_HZ X1 X3 Y1 Y3 HEIGHT DHEIGHT VOL PCHI2 TYPE ASS CLUSTID MEMCNT 
      
       Are mapped onto analysis peak list

           'Number', '#', 'Position F1', 'Position F2', 'Sampled None',
           'Assign F1', 'Assign F2', 'Assign F3', 'Height', 'Volume',
           'Line Width F1 (Hz)', 'Line Width F2 (Hz)', 'Line Width F3 (Hz)',
            'Merit', 'Details', 'Fit Method', 'Vol. Method'

       Or sparky peaklist

             Assignment         w1         w2        Volume   Data Height   lw1 (hz)   lw2 (hz)

       Clusters or peaks are selected

       Standard NMRPipe clustering used or my peak clustering

     ToDo:
         1. allow decimal values for point positions of peaks
         2. figure out whether points start from 0...
 
"""
import os

import yaml
import pandas as pd
import numpy as np
import nmrglue as ng

from scipy import ndimage
from nmrglue.analysis.peakpick import clusters
from docopt import docopt

# Read peak list and output into pandas dataframe for fitting peaks
# make column in dataframe for group identities
#  column for plane of expt
# column for peak masking params
#  column for fit results etc.
# everything done with one df including all peaks.

analysis_to_pipe = {
    "#": "INDEX",
    # "": "X_AXIS",
    # "": "Y_AXIS",
    # "": "DX",
    # "": "DY",
    "Position F1": "X_PPM",
    "Position F2": "Y_PPM",
    # "": "X_HZ",
    # "": "Y_HZ",
    # "": "XW",
    # "": "YW",
    "Line Width F1 (Hz)": "XW_HZ",
    "Line Width F2 (Hz)": "YW_HZ",
    # "": "X1",
    # "": "X3",
    # "": "Y1",
    # "": "Y3",
    "Height": "HEIGHT",
    # "Height": "DHEIGHT",
    "Volume": "VOL",
    # "": "PCHI2",
    # "": "TYPE",
    # "": "ASS",
    # "": "CLUSTID",
    # "": "MEMCNT"
}

sparky_to_pipe = {
    "index": "INDEX",
    # "": "X_AXIS",
    # "": "Y_AXIS",
    # "": "DX",
    # "": "DY",
    "w1": "X_PPM",
    "w2": "Y_PPM",
    # "": "X_HZ",
    # "": "Y_HZ",
    # "": "XW",
    # "": "YW",
    "lw1 (hz)": "XW_HZ",
    "lw2 (hz)": "YW_HZ",
    # "": "X1",
    # "": "X3",
    # "": "Y1",
    # "": "Y3",
    "Height": "HEIGHT",
    # "Height": "DHEIGHT",
    "Volume": "VOL",
    # "": "PCHI2",
    # "": "TYPE",
    "Assignment": "ASS",
    # "": "CLUSTID",
    # "": "MEMCNT"
}


def clusters(data, locations, pthres, nthres, d_struc=None, l_struc=None, ndil=0):
    """
    Perform cluster analysis of peak locations.
    Parameters
    ----------
    data : ndarray
        Array of data which has been peak picked.
    locations : list
        List of peak locations.
    pthres : float
        Postive peak threshold. None for no postive peaks.
    nthres : float
        Negative peak threshold. None for no negative peaks.
    d_struc : ndarray, optional
        Structure of binary dilation to apply on segments before clustering.
        None uses a square structure with connectivity of one.
    l_struc : ndarray, optional
        Structure to use for determining segment connectivity in clustering.
        None uses square structure with connectivity of one.
    dnil : int, optional
        Number of dilation to apply on segments before determining clusters.
    Returns
    -------
    cluster_ids : list
        List of cluster number corresponding to peak locations.
    """
    # make a binary array of regions above/below the noise thresholds
    if pthres == None:  # negative peaks only
        input = data < nthres
    elif nthres == None:  # postive peaks only
        input = data > pthres
    else:  # both positive and negative
        input = np.bitwise_or(data < nthres, data > pthres)

    # apply dialations to these segments
    if ndil != 0:
        input = ndimage.binary_dilation(input, d_struc, iterations=ndil)

    # label this array, these are the clusters.
    labeled_array, num_features = ndimage.label(input, l_struc)

    return [labeled_array[i[0], i[1]] for i in locations]


def read_analysis(path, data, dims, noise=5e4, pthres=20 * 5e4, nthres=None, ndil=0):
    """Read analysis peak list and convert to NMRPipe-ish format also find peak clusters
     
    Parameters
    ----------
    path : path-like or str
        path to peaklist
    data : ndarray
        NMRPipe format data
    dims: [planes,y,x]
        
    noise : float
       estimation of noise of spectrum
    pthres : float
        threshold for positive signals above which clusters are selected
    nthres : float
        negative peak threshold if None then only positive peaks used
    ndil: int
        number of iterations of ndimage.binary_dilation function if set to 0 then function not used
    
    Returns
    -------
    df : pandas DataFrame
        dataframe containing peaklist

    """
    df = pd.read_table(path, delimiter="\t")
    new_columns = [analysis_to_pipe.get(i, i) for i in df.columns]
    # print(df.columns)
    # print(new_columns)
    pipe_columns = dict(zip(df.columns, new_columns))
    df = df.rename(index=str, columns=pipe_columns)
    # df["CLUSTID"] = np.arange(1, len(df) + 1)

    dic, data = ng.pipe.read(data)
    udic = ng.pipe.guess_udic(dic, data)
    ndim = udic["ndim"]
    # need to sort out dimension attribution since a2 is has fucked dims
    # get ready for some gaaaarbage, buddy
    print(" ".join(udic[i]["label"] for i in range(ndim)))
    print(ndim, data.shape)
    planes, f1_dim, f2_dim = dims
    print(dic["FDF%dSW" % 3])
    # calculate points per hertz
    #  number of points / SW
    # pt_per_hz_f2dim = data.shape[f2_dim] / dic["FDF%dSW" % f2_dim]
    # pt_per_hz_f1dim = data.shape[f1_dim] / dic["FDF%dSW" % f1_dim]  #  this needs checking
    pt_per_hz_f2dim = udic[f2_dim]["size"] / udic[f2_dim]["sw"]
    pt_per_hz_f1dim = udic[f1_dim]["size"] / udic[f1_dim]["sw"]
    print("Points per hz f1 = %s, f2 = %s" % (pt_per_hz_f1dim, pt_per_hz_f2dim))
    uc_f2 = ng.pipe.make_uc(dic, data, dim=f2_dim)
    uc_f1 = ng.pipe.make_uc(dic, data, dim=f1_dim)

    # something is wrong with the points
    df["X_AXIS"] = df.X_PPM.apply(lambda x: uc_f2(x, "ppm"))
    df["Y_AXIS"] = df.Y_PPM.apply(lambda x: uc_f1(x, "ppm"))
    # decimal point value
    df["X_AXISf"] = df.X_PPM.apply(lambda x: uc_f2.f(x, "ppm"))
    df["Y_AXISf"] = df.Y_PPM.apply(lambda x: uc_f1.f(x, "ppm"))
    # print(df["XW_HZ"].head())
    # print(df["YW_HZ"].head())
    # print(uc_f2("10 Hz"))

    df.XW_HZ.replace("None", "20.0", inplace=True)
    df.YW_HZ.replace("None", "20.0", inplace=True)
    df.XW_HZ.replace(np.NaN, "20.0", inplace=True)
    df.YW_HZ.replace(np.NaN, "20.0", inplace=True)

    df["XW_HZ"] = df.XW_HZ.apply(lambda x: float(x))
    df["YW_HZ"] = df.YW_HZ.apply(lambda x: float(x))

    #  convert Hz lw to points
    df["XW"] = df.XW_HZ.apply(lambda x: x * pt_per_hz_f2dim)
    df["YW"] = df.YW_HZ.apply(lambda x: x * pt_per_hz_f1dim)
    # makes an assignment column
    df["ASS"] = df.apply(lambda i: "".join([i["Assign F1"], i["Assign F2"]]), axis=1)
    # df["HEIGHT"] = df.apply(lambda i: data[0][i.Y_AXIS,i.X_AXIS],axis=1)
    # cluster peaks using scipy.ndimage.label from nmrglue
    df["CLUSTID"] = clusters(
        data[0],
        [[y, x] for y, x in zip(df.Y_AXIS, df.X_AXIS)],
        pthres=pthres,
        nthres=nthres,
        ndil=ndil,
    )
    #  renumber "0" clusters
    max_clustid = df["CLUSTID"].max()
    n_of_zeros = len(df[df["CLUSTID"] == 0]["CLUSTID"])
    df.loc[df[df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
        max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
    )

    return df


def read_pipe(path):
    to_skip = 0
    with open(path) as f:
        lines = f.readlines()
        columns = lines[0].strip().split()[1:]
        for line in lines:
            if line[:5].strip(" ").isdigit():
                break
            else:
                to_skip += 1
    df = pd.read_table(path, skiprows=to_skip, names=columns, delim_whitespace=True)
    return df


def read_sparky(path, data, dims, noise=5e4, pthres=20 * 5e4, nthres=None, ndil=0):
    df = pd.read_csv(
        path,
        skiprows=2,
        delim_whitespace=True,
        names=["ASS", "Y_PPM", "X_PPM", "VOLUME", "HEIGHT", "YW_HZ", "XW_HZ"],
    )
    print(df.columns)
    print(df.head())
    # new_columns = [analysis_to_pipe.get(i, i) for i in df.columns]
    ## print(new_columns)
    # pipe_columns = dict(zip(df.columns, new_columns))
    # df = df.rename(index=str, columns=pipe_columns)
    ##df["CLUSTID"] = np.arange(1, len(df) + 1)

    dic, data = ng.pipe.read(data)
    udic = ng.pipe.guess_udic(dic, data)
    ndim = udic["ndim"]
    # need to sort out dimension attribution since a2 is has fucked dims
    # get ready for some gaaaarbage, buddy
    print(" ".join(udic[i]["label"] for i in range(ndim)))
    print(ndim, data.shape)
    planes, f1_dim, f2_dim = dims
    # print(dic["FDF%dSW" % 3])

    uc_f2 = ng.pipe.make_uc(dic, data, dim=f2_dim)
    uc_f1 = ng.pipe.make_uc(dic, data, dim=f1_dim)

    df["X_AXIS"] = df.X_PPM.apply(lambda i: uc_f2(i, "ppm"))
    df["Y_AXIS"] = df.Y_PPM.apply(lambda i: uc_f1(i, "ppm"))
    df["X_AXISf"] = df.X_PPM.apply(lambda i: uc_f2.f(i, "ppm"))
    df["Y_AXISf"] = df.Y_PPM.apply(lambda i: uc_f1.f(i, "ppm"))
    # print(df["XW_HZ"].head())
    # print(df["YW_HZ"].head())
    # print(uc_f1("10 Hz"))

    df.XW_HZ.replace("None", "20.0", inplace=True)
    df.YW_HZ.replace("None", "20.0", inplace=True)
    df.XW_HZ.replace(np.NaN, "20.0", inplace=True)
    df.YW_HZ.replace(np.NaN, "20.0", inplace=True)

    df["XW_HZ"] = df.XW_HZ.apply(lambda x: float(x))
    df["YW_HZ"] = df.YW_HZ.apply(lambda x: float(x))

    # calculate points per hertz
    #  number of points / SW
    pt_per_hz_f2dim = udic[f2_dim]["size"] / udic[f2_dim]["sw"]
    pt_per_hz_f1dim = udic[f1_dim]["size"] / udic[f1_dim]["sw"]
    # pt_per_hz_f2dim = data.shape[f2_dim] / dic["FDF%dSW" % f2_dim]
    # pt_per_hz_f1dim = data.shape[f1_dim] / dic["FDF%dSW" % f1_dim]  #  this needs checking
    #  convert Hz lw to points
    df["XW"] = df.XW_HZ.apply(lambda x: x * pt_per_hz_f2dim)
    df["YW"] = df.YW_HZ.apply(lambda x: x * pt_per_hz_f1dim)
    # cluster peaks using scipy.ndimage.label from nmrglue
    df["INDEX"] = np.arange(len(df.X_PPM), dtype=int)
    df["HEIGHT"] = df.apply(lambda i: data[0][i.Y_AXIS, i.X_AXIS], axis=1)

    df["CLUSTID"] = clusters(
        data[0],
        [[y, x] for y, x in zip(df.Y_AXIS, df.X_AXIS)],
        pthres=pthres,
        nthres=nthres,
        ndil=ndil,
    )
    #  renumber "0" clusters
    max_clustid = df["CLUSTID"].max()
    n_of_zeros = len(df[df["CLUSTID"] == 0]["CLUSTID"])
    df.loc[df[df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
        max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
    )

    return df


def to_fuda(df):
    groups = df.groupby("CLUSTID")
    with open("params.fuda", "w") as f:
        pass


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = args["<peaklist>"]
    print(os.path.splitext(filename)[0])
    # print(args)

    if args.get("--nthres") == "None":
        args["--nthres"] = None
    else:
        args["--nthres"] = eval(args["--nthres"])

    if args.get("--pthres") == "None":
        args["--pthres"] = None
    else:
        args["--pthres"] = eval(args["--pthres"])

    noise = eval(args.get("--noise"))
    ndil = int(args.get("--ndil"))
    pthres = args.get("--pthres")
    print(args)

    if args.get("--a2"):
        dims = args.get("--dims")
        dims = [int(i) for i in dims.split(",")]
        pipe_ft_file = args.get("<data>")
        data = read_analysis(
            filename,
            pipe_ft_file,
            dims,
            pthres=args.get("--pthres"),
            nthres=args.get("--nthres"),
            noise=noise,
            ndil=ndil,
        )

    elif args.get("--sparky"):
        dims = args.get("--dims")
        dims = [int(i) for i in dims.split(",")]
        pipe_ft_file = args.get("<data>")
        data = read_sparky(
            filename,
            pipe_ft_file,
            dims,
            pthres=args.get("--pthres"),
            nthres=args.get("--nthres"),
            noise=noise,
            ndil=ndil,
        )
    else:

        data = read_pipe(filename)
    print(data.head())
    outfmt = args.get("--outfmt","pkl")
    outname = os.path.splitext(filename)[0] 
    if outfmt == "csv":
        outname = outname + ".csv"
        data.to_csv(outname)
    else:
        outname = outname + ".pkl"
        data.to_pickle(outname)

    yaml = f"""
    ##########################################################################################################
    #  This first block is global parameters which can be overridden by adding the desired argument          #
    #  to your list of spectra. One exception is "colors" which if set in global params overrides the        #
    #  color option set for individual spectra as the colors will now cycle through the chosen matplotlib    #
    #  colormap                                                                                              #
    ##########################################################################################################

    cs: {pthres}                     # contour start
    contour_num: 10                 # number of contours
    contour_factor: 1.2             # contour factor
    colors: tab20                    # must be matplotlib.cm colormap

    outname: ["clusters.pdf","clusters.png"] # either single value or list of output names
    ncol: 1 #  tells matplotlib how many columns to give the figure legend - if not set defaults to 2
    clusters: {outname}
    dims: {dims}

    # Here is where your list of spectra to plot goes
    spectra:

            - fname: {pipe_ft_file}
              label: ""
              contour_num: 20
              linewidths: 0.1
    """

    if args.get("--show"):
        with open("show_clusters.yml", "w") as out:
            out.write(yaml)
        os.system("spec.py show_clusters.yml")
