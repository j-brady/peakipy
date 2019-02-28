#!/usr/bin/env python3
""" Read NMRPipe/Analysis peaklist into pandas dataframe

    Usage:
        read_peaklist.py <peaklist> <data> (--a2|--sparky|--pipe) [options]

    Arguments:
        <peaklist>                Analysis2/Sparky/NMRPipe peak list (see below)
        <data>                    2D or pseudo3D NMRPipe data

        --a2                      Analysis peaklist as input (tab delimited)
        --sparky                  Sparky peaklist as input
        --pipe                    NMRPipe peaklist as input

    Options:
        -h --help                 Show this screen
        --version                 Show version
 
        --thres=<thres>           Threshold for making binary mask that is used for peak clustering [default: None]
                                  If set to None then threshold_otsu from scikit-image is used to determine threshold

        --struc_el=<str>          Structuring element for binary_closing [default: disk]
                                  'square'|'disk'|'rectangle'

        --struc_size=<(float,)>   size/dimensions of structuring element [default: (5,)]
                                  for square and disk first element of tuple is used (for disk value corresponds to diameter)
                                  for rectangle, tuple corresponds to (width,height).

        --f1radius=<float>        F1 radius in ppm for fit mask [default: 0.4]
        --f2radius=<float>        F2 radius in ppm for fit mask [default: 0.04]

        --dims=<planes,F1,F2>     Order of dimensions [default: 0,1,2]

        --outfmt=<csv/pkl>        Format of output peaklist [default: csv]
 
        --show                    Show the clusters on the spectrum color coded using matplotlib

    Examples:
        read_peaklist.py test.tab
        read_peaklist.py test.a2 test.ft2 --a2 --thres=1e5  --dims=0,2,1

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

       Clusters of peaks are selected


"""
import os
from pathlib import Path

import pandas as pd
import numpy as np
import nmrglue as ng

from docopt import docopt
from scipy import ndimage
from skimage.morphology import square, binary_closing, opening, disk, rectangle
from skimage.filters import threshold_otsu, threshold_adaptive

from peakipy.core import make_mask, Pseudo3D 

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
    radii: [x,y]
        Mask radii in ppm

    
    Methods
    -------

    clusters : 
    adaptive_clusters : 

    Returns
    -------
    df : pandas DataFrame
        dataframe containing peaklist

    """

    def __init__(self, path, data, fmt="a2", dims=[0, 1, 2], radii=[0.04, 0.4]):
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
        pseudo3D = Pseudo3D(dic, self.data, dims)
        self.data = pseudo3D.data
        uc_f1 = pseudo3D.uc_f1
        uc_f2 = pseudo3D.uc_f2
        dims = pseudo3D.dims
        self.data = pseudo3D.data
        self.pt_per_ppm_f1 = pseudo3D.pt_per_ppm_f1
        self.pt_per_ppm_f2 = pseudo3D.pt_per_ppm_f2
        pt_per_hz_f2dim = pseudo3D.pt_per_hz_f2
        pt_per_hz_f1dim = pseudo3D.pt_per_hz_f1
        print("Points per hz f1 = %s, f2 = %s" % (pt_per_hz_f1dim, pt_per_hz_f2dim))

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
        f2radius, f1radius = radii
        self.df["X_RADIUS_PPM"] = np.zeros(len(self.df)) + f2radius
        self.df["Y_RADIUS_PPM"] = np.zeros(len(self.df)) + f1radius
        self.df["X_RADIUS"] = self.df.X_RADIUS_PPM.apply(lambda x: x * self.pt_per_ppm_f2) 
        self.df["Y_RADIUS"] = self.df.Y_RADIUS_PPM.apply(lambda x: x * self.pt_per_ppm_f1) 

        ## rearrange dims
        #if dims != [0, 1, 2]:
        #    data = np.transpose(data, dims)

    def _read_analysis(self):

        df = pd.read_table(self.path, delimiter="\t")
        new_columns = [analysis_to_pipe.get(i, i) for i in df.columns]
        pipe_columns = dict(zip(df.columns, new_columns))
        df = df.rename(index=str, columns=pipe_columns)

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

        thres : float
            threshold for positive signals above which clusters are selected. If None then
            threshold_otsu is used

        struc_el: str 
            'square'|'disk'|'rectangle'
            structuring element for binary_closing of thresholded data can be square, disc or rectangle

        struc_size: tuple
            size/dimensions of structuring element
            for square and disk first element of tuple is used (for disk value corresponds to diameter)
            for rectangle, tuple corresponds to (width,height).


        """
        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]

        if thres == None:
            self.thresh = threshold_otsu(self.data[0])
        else:
            self.thresh = thres
        
        # get positive and negative 
        thresh_data = np.bitwise_or(
            self.data[0] < (self.thresh * -1.0), self.data[0] > self.thresh
        )

        if struc_el == "disk":
            radius = struc_size[0]
            radius = radius / 2.0
            print(f"using disk with {radius}")
            closed_data = binary_closing(thresh_data, disk(radius))

        elif struc_el == "square":
            width = struc_size[0]
            print(f"using square with {width}")
            closed_data = binary_closing(thresh_data, square(width))

        elif struc_el == "rectangle":
            width, height = struc_size
            print(f"using rectangle with {width} and {height}")
            closed_data = binary_closing(data, rectangle(width, height))

        else:
            print(f"Not using any closing function")
            closed_data = data

        labeled_array, num_features = ndimage.label(closed_data, l_struc)
        # print(labeled_array, num_features)

        self.df["CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

        # count how many peaks per cluster
        self.df["MEMCNT"] = np.zeros(len(self.df),dtype=int)
        for ind, group in self.df.groupby("CLUSTID"):
            self.df.loc[group.index,"MEMCNT"] = len(group)


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

    def get_thres(self):
        return self.thresh


def to_fuda(df):
    groups = df.groupby("CLUSTID")
    with open("params.fuda", "w") as f:
        pass


if __name__ == "__main__":
    args = docopt(__doc__)
    filename = Path(args["<peaklist>"])
    # print(filename.stem)

    if args.get("--thres") == "None":
        args["--thres"] = None
    else:
        args["--thres"] = eval(args["--thres"])
    
    thres = args.get("--thres")
    print("Using arguments:", args)

    f1radius = float(args.get("--f1radius"))
    f2radius = float(args.get("--f2radius"))

    clust_args = {
        "struc_el": args.get("--struc_el"),
        "struc_size": eval(args.get("--struc_size")),
    }

    dims = args.get("--dims")
    dims = [int(i) for i in dims.split(",")]
    pipe_ft_file = args.get("<data>")
    if args.get("--a2"):

        peaks = Peaklist(filename, pipe_ft_file, fmt="a2", dims=dims, radii=[f2radius,f1radius])
        # peaks.adaptive_clusters(block_size=151,offset=0)
        peaks.clusters(thres=thres, **clust_args, l_struc=None)
        #peaks.mask_method(x_radius=0.04,y_radius=0.25)
        data = peaks.get_df()
        thres = peaks.get_thres()


    elif args.get("--sparky"):

        peaks = Peaklist(filename, pipe_ft_file, fmt="sparky", dims=dims)
        peaks.clusters(thres=thres, **clust_args, l_struc=None)
        data = peaks.get_df()
        thres = peaks.get_thres()

    else:

        data = read_pipe(filename)
    print(data.head())
    outfmt = args.get("--outfmt", "csv")
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

    cs: {thres}                     # contour start
    contour_num: 10                 # number of contours
    contour_factor: 1.2             # contour factor
    colors: tab20                   # must be matplotlib.cm colormap
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
