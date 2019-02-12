#!/usr/bin/env python3
""" Read NMRPipe/Analysis peaklist into pandas dataframe

    Usage:
        read_peaklist.py <peaklist> <data> (--a2|--sparky|--pipe) [options]

    Arguments:
        <peaklist>  Analysis2/Sparky/NMRPipe peak list (see below)
        <data>      2D or pseudo3D NMRPipe data

        --a2        Analysis peaklist as input
        --sparky    Sparky peaklist as input
        --pipe      NMRPipe peaklist as input

    Options:
        -h --help  Show this screen
        --version  Show version
 
        --noise=<noise>        Noise of spectrum [default: 5e4]
        --pthres=<pthres>      Positive peakpick threshold [default: None]
        --nthres=<nthres>      Negative peakpick threshold [default: None]

        --ndil=<ndil>          Number of iterations for ndimage.binary_dilation [default: 0]
        --clust
        --struc_el=<str>       [default: square]
        --struc_size=<float>   [default: (3,)]

        --dims=<planes,F1,F2>  Order of dimensions [default: 0,1,2]

        --outfmt=<csv/pkl>     Format of output peaklist [default: "pkl"]
 
        --show                 Show the clusters on the spectrum color coded

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
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import nmrglue as ng

from scipy import ndimage
from nmrglue.analysis.peakpick import clusters
from docopt import docopt
from skimage.morphology import square, closing, opening, disk, rectangle
from skimage.filters import threshold_otsu, threshold_adaptive

from peak_deconvolution.core import make_mask 

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


class Peaklist:
    """ Read analysis or sparky peak list and convert to NMRPipe-ish format also find peak clusters
     
    Parameters
    ----------
    path : path-like or str
        path to peaklist
    data : ndarray
        NMRPipe format data
    fmt : a2|sparky|pipe
    dims: [planes,y,x]

    
    Methods
    -------

    clusters : 
    adaptive_clusters : 

    Returns
    -------
    df : pandas DataFrame
        dataframe containing peaklist

    """

    def __init__(self, path, data, fmt="a2", dims=[0, 1, 2]):
        self.fmt = fmt
        self.path = path

        if self.fmt == "a2":
            self.df = self._read_analysis()

        elif self.fmt == "sparky":
            self.df = self._read_sparky()

        elif self.fmt == "pipe":
            self.df = self._read_pipe()

        else:
            raise (TypeError, "I don't know this format")

        #  read pipe data
        dic, self.data = ng.pipe.read(data)
        udic = ng.pipe.guess_udic(dic, self.data)
        ndim = udic["ndim"]
        # need to sort out dimension attribution since a2 is has fucked dims
        # get ready for some gaaaarbage, buddy
        print(" ".join(udic[i]["label"] for i in range(ndim)))
        print(ndim, self.data.shape)
        planes, f1_dim, f2_dim = dims
        # calculate points per hertz
        # number of points / SW
        pt_per_hz_f2dim = udic[f2_dim]["size"] / udic[f2_dim]["sw"]
        pt_per_hz_f1dim = udic[f1_dim]["size"] / udic[f1_dim]["sw"]

        self.pt_per_ppm_f2 = udic[f2_dim]["size"] / (udic[f2_dim]["sw"] / udic[f2_dim]["obs"])
        self.pt_per_ppm_f1 = udic[f1_dim]["size"] / (udic[f1_dim]["sw"] / udic[f1_dim]["obs"])

        
        print("Points per hz f1 = %s, f2 = %s" % (pt_per_hz_f1dim, pt_per_hz_f2dim))
        uc_f2 = ng.pipe.make_uc(dic, self.data, dim=f2_dim)
        uc_f1 = ng.pipe.make_uc(dic, self.data, dim=f1_dim)

        # int point value
        self.df["X_AXIS"] = self.df.X_PPM.apply(lambda x: uc_f2(x, "ppm"))
        self.df["Y_AXIS"] = self.df.Y_PPM.apply(lambda x: uc_f1(x, "ppm"))
        # decimal point value
        self.df["X_AXISf"] = self.df.X_PPM.apply(lambda x: uc_f2.f(x, "ppm"))
        self.df["Y_AXISf"] = self.df.Y_PPM.apply(lambda x: uc_f1.f(x, "ppm"))
        # in case of missing values (should estimate though)
        self.df.XW_HZ.replace("None", "20.0", inplace=True)
        self.df.YW_HZ.replace("None", "20.0", inplace=True)
        self.df.XW_HZ.replace(np.NaN, "20.0", inplace=True)
        self.df.YW_HZ.replace(np.NaN, "20.0", inplace=True)
        # convert linewidths to float
        self.df["XW_HZ"] = self.df.XW_HZ.apply(lambda x: float(x))
        self.df["YW_HZ"] = self.df.YW_HZ.apply(lambda x: float(x))
        #  convert Hz lw to points
        self.df["XW"] = self.df.XW_HZ.apply(lambda x: x * pt_per_hz_f2dim)
        self.df["YW"] = self.df.YW_HZ.apply(lambda x: x * pt_per_hz_f1dim)
        # makes an assignment column
        if self.fmt == "a2":
            self.df["ASS"] = self.df.apply(
                lambda i: "".join([i["Assign F1"], i["Assign F2"]]), axis=1
            )
        
        # make default values for X and Y radii for fit masks
        self.df["X_RADIUS_PPM"] = np.zeros(len(self.df)) + 0.04
        self.df["Y_RADIUS_PPM"] = np.zeros(len(self.df)) + 0.4
        self.df["X_RADIUS"] = self.df.X_RADIUS_PPM.apply(lambda x: x * self.pt_per_ppm_f2) 
        self.df["Y_RADIUS"] = self.df.Y_RADIUS_PPM.apply(lambda x: x * self.pt_per_ppm_f1) 

        # rearrange dims
        if dims != [0, 1, 2]:
            data = np.transpose(data, dims)

    def _read_analysis(self):
        df = pd.read_table(self.path, delimiter="\t")
        new_columns = [analysis_to_pipe.get(i, i) for i in df.columns]
        # print(df.columns)
        # print(new_columns)
        pipe_columns = dict(zip(df.columns, new_columns))
        df = df.rename(index=str, columns=pipe_columns)
        # df["CLUSTID"] = np.arange(1, len(df) + 1)
        return df

    def _read_sparky(self):

        df = pd.read_csv(
            self.path,
            skiprows=2,
            delim_whitespace=True,
            names=["ASS", "Y_PPM", "X_PPM", "VOLUME", "HEIGHT", "YW_HZ", "XW_HZ"],
        )

        return df

    def _read_pipe(self):
        to_skip = 0
        with open(self.path) as f:
            lines = f.readlines()
            columns = lines[0].strip().split()[1:]
            for line in lines:
                if line[:5].strip(" ").isdigit():
                    break
                else:
                    to_skip += 1
        df = pd.read_table(
            self.path, skiprows=to_skip, names=columns, delim_whitespace=True
        )
        return df

    def clusters(self, thres=None, struc_el="square", struc_size=(3,), l_struc=None):
        """ Find clusters of peaks 

        Need to update these docs.

        pthres : float
            threshold for positive signals above which clusters are selected
        nthres : float
            negative peak threshold if None then only positive peaks used
        ndil: int
            number of iterations of ndimage.binary_dilation function if set to 0 then function not used

        """
        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]

        if thres == None:
            self.thresh = threshold_otsu(self.data[0])
        else:
            self.thresh = thres

        thresh_data = np.bitwise_or(
            self.data[0] < (self.thresh * -1.0), self.data[0] > self.thresh
        )

        if struc_el == "disk":
            radius = struc_size[0]
            radius = radius / 2.0
            print(f"using disk with {radius}")
            closed_data = closing(thresh_data, disk(radius))

        elif struc_el == "square":
            width = struc_size[0]
            print(f"using square with {width}")
            closed_data = closing(thresh_data, square(width))

        elif struc_el == "rectangle":
            width, height = struc_size
            print(f"using rectangle with {width} and {height}")
            data = closing(data, rectangle(width, height))

        else:
            print(f"Not using any closing function")

        labeled_array, num_features = ndimage.label(thresh_data, l_struc)
        # print(labeled_array, num_features)

        self.df["CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

    def adaptive_clusters(self, block_size, offset, l_struc=None):

        self.thresh = threshold_otsu(self.data[0])

        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]

        binary_adaptive = threshold_adaptive(
            self.data[0], block_size=block_size, offset=offset
        )

        labeled_array, num_features = ndimage.label(binary_adaptive, l_struc)
        # print(labeled_array, num_features)

        self.df["CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )


    def mask_method(self, x_radius=0.04, y_radius=0.4, l_struc=None):
        

        self.thresh = threshold_otsu(self.data[0])

        x_radius = self.pt_per_ppm_f2 * x_radius
        y_radius = self.pt_per_ppm_f1 * y_radius

        mask = np.zeros(self.data[0].shape, dtype=bool)


        for ind, peak in self.df.iterrows():
            mask += make_mask(self.data[0],peak.X_AXISf,peak.Y_AXISf, x_radius, y_radius)

        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]
        labeled_array, num_features = ndimage.label(mask, l_struc)

        self.df["CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )
        import matplotlib.pyplot as plt
        plt.imshow(mask)
        plt.show()


    def get_df(self):
        return self.df

    def get_pthres(self):
        return self.thresh


def to_fuda(df):
    groups = df.groupby("CLUSTID")
    with open("params.fuda", "w") as f:
        pass


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = Path(args["<peaklist>"])
    # print(filename.stem)

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
    nthres = args.get("--nthres")
    print(args)

    clust_args = {
        "struc_el": args.get("--struc_el"),
        "struc_size": eval(args.get("--struc_size")),
    }

    dims = args.get("--dims")
    dims = [int(i) for i in dims.split(",")]
    pipe_ft_file = args.get("<data>")
    if args.get("--a2"):

        peaks = Peaklist(filename, pipe_ft_file, fmt="a2", dims=dims)
        # peaks.adaptive_clusters(block_size=151,offset=0)
        peaks.clusters(thres=pthres, **clust_args, l_struc=None)
        #peaks.mask_method(x_radius=0.04,y_radius=0.25)
        data = peaks.get_df()
        pthres = peaks.get_pthres()


    elif args.get("--sparky"):

        peaks = Peaklist(filename, pipe_ft_file, fmt="sparky", dims=dims)
        peaks.clusters(thres=pthres, **clust_args, l_struc=None)
        data = peaks.get_df()
        pthres = peaks.get_pthres()

    else:

        data = read_pipe(filename)
    print(data.head())
    outfmt = args.get("--outfmt", "pkl")
    outname = filename.stem
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
    show_cs: True

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
