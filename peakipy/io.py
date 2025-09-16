import sys
from pathlib import Path
from enum import Enum

import numpy as np
import nmrglue as ng
import pandas as pd
import textwrap
from rich import print
from rich.console import Console


from bokeh.palettes import Category20
from scipy import ndimage
from skimage.morphology import binary_closing, disk, footprint_rectangle
from skimage.filters import threshold_otsu
from pydantic import BaseModel

from peakipy.utils import df_to_rich_table
from peakipy.fitting import make_mask

console = Console()


class StrucEl(str, Enum):
    square = "square"
    disk = "disk"
    rectangle = "rectangle"
    mask_method = "mask_method"


class PeaklistFormat(str, Enum):
    a2 = "a2"
    a3 = "a3"
    sparky = "sparky"
    pipe = "pipe"
    peakipy = "peakipy"
    csv = "csv"


class OutFmt(str, Enum):
    csv = "csv"
    pkl = "pkl"


class PeaklistColumns(BaseModel):
    """These are the columns required for performing fits in peakipy"""

    INDEX: int
    X_AXIS: int
    Y_AXIS: int
    X_AXISf: float
    Y_AXISf: float
    X_PPM: float
    Y_PPM: float
    XW: float
    YW: float
    XW_HZ: float
    YW_HZ: float
    HEIGHT: float
    VOL: float
    ASS: str
    X_RADIUS: float
    Y_RADIUS: float
    X_RADIUS_PPM: float
    Y_RADIUS_PPM: float
    include: str


class PeaklistColumnsWithClusters(PeaklistColumns):
    CLUSTID: int
    MEMCNT: int
    color: str


class Pseudo3D:
    """Read dic, data from NMRGlue and dims from input to create a Pseudo3D dataset

    :param dic: from nmrglue.pipe.read
    :type dic: dict

    :param data: data from nmrglue.pipe.read
    :type data: numpy.array

    :param dims: dimension order i.e [0,1,2] where 0 = planes, 1 = f1, 2 = f2
    :type dims: list
    """

    def __init__(self, dic, data, dims):
        # check dimensions
        self._udic = ng.pipe.guess_udic(dic, data)
        self._ndim = self._udic["ndim"]

        if self._ndim == 1:
            err = f"""[red]
            ##########################################
                NMR Data should be either 2D or 3D
            ##########################################
            [/red]"""
            # raise TypeError(err)
            sys.exit(err)

        # check that spectrum has correct number of dims
        elif self._ndim != len(dims):
            err = f"""[red]
            #################################################################
               Your spectrum has {self._ndim} dimensions with shape {data.shape}
               but you have given a dimension order of {dims}...
            #################################################################
            [/red]"""
            # raise ValueError(err)
            sys.exit(err)

        elif (self._ndim == 2) and (len(dims) == 2):
            self._f1_dim, self._f2_dim = dims
            self._planes = 0
            self._uc_f1 = ng.pipe.make_uc(dic, data, dim=self._f1_dim)
            self._uc_f2 = ng.pipe.make_uc(dic, data, dim=self._f2_dim)
            # make data pseudo3d
            self._data = data.reshape((1, data.shape[0], data.shape[1]))
            self._dims = [self._planes, self._f1_dim + 1, self._f2_dim + 1]

        else:
            self._planes, self._f1_dim, self._f2_dim = dims
            self._dims = dims
            self._data = data
            # make unit conversion dicts
            self._uc_f2 = ng.pipe.make_uc(dic, data, dim=self._f2_dim)
            self._uc_f1 = ng.pipe.make_uc(dic, data, dim=self._f1_dim)

        #  rearrange data if dims not in standard order
        if self._dims != [0, 1, 2]:
            # np.argsort returns indices of array for order 0,1,2 to transpose data correctly
            # self._dims = np.argsort(self._dims)
            self._data = np.transpose(data, self._dims)

        self._dic = dic

        self._f1_label = self._udic[self._f1_dim]["label"]
        self._f2_label = self._udic[self._f2_dim]["label"]

    @property
    def uc_f1(self):
        """Return unit conversion dict for F1"""
        return self._uc_f1

    @property
    def uc_f2(self):
        """Return unit conversion dict for F2"""
        return self._uc_f2

    @property
    def dims(self):
        """Return dimension order"""
        return self._dims

    @property
    def data(self):
        """Return array containing data"""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def dic(self):
        return self._dic

    @property
    def udic(self):
        return self._udic

    @property
    def ndim(self):
        return self._ndim

    @property
    def f1_label(self):
        # dim label
        return self._f1_label

    @property
    def f2_label(self):
        # dim label
        return self._f2_label

    @property
    def planes(self):
        return self.dims[0]

    @property
    def n_planes(self):
        return self.data.shape[self.planes]

    @property
    def f1(self):
        return self.dims[1]

    @property
    def f2(self):
        return self.dims[2]

    # size of f1 and f2 in points
    @property
    def f2_size(self):
        """Return size of f2 dimension in points"""
        return self._udic[self._f2_dim]["size"]

    @property
    def f1_size(self):
        """Return size of f1 dimension in points"""
        return self._udic[self._f1_dim]["size"]

    # points per ppm
    @property
    def pt_per_ppm_f1(self):
        return self.f1_size / (
            self._udic[self._f1_dim]["sw"] / self._udic[self._f1_dim]["obs"]
        )

    @property
    def pt_per_ppm_f2(self):
        return self.f2_size / (
            self._udic[self._f2_dim]["sw"] / self._udic[self._f2_dim]["obs"]
        )

    # points per hz
    @property
    def pt_per_hz_f1(self):
        return self.f1_size / self._udic[self._f1_dim]["sw"]

    @property
    def pt_per_hz_f2(self):
        return self.f2_size / self._udic[self._f2_dim]["sw"]

    # hz per point
    @property
    def hz_per_pt_f1(self):
        return 1.0 / self.pt_per_hz_f1

    @property
    def hz_per_pt_f2(self):
        return 1.0 / self.pt_per_hz_f2

    # ppm per point
    @property
    def ppm_per_pt_f1(self):
        return 1.0 / self.pt_per_ppm_f1

    @property
    def ppm_per_pt_f2(self):
        return 1.0 / self.pt_per_ppm_f2

    # get ppm limits for ppm scales
    @property
    def f2_ppm_scale(self):
        return self.uc_f2.ppm_scale()

    @property
    def f1_ppm_scale(self):
        return self.uc_f1.ppm_scale()

    @property
    def f2_ppm_limits(self):
        return self.uc_f2.ppm_limits()

    @property
    def f1_ppm_limits(self):
        return self.uc_f1.ppm_limits()

    @property
    def f1_ppm_max(self):
        return max(self.f1_ppm_limits)

    @property
    def f1_ppm_min(self):
        return min(self.f1_ppm_limits)

    @property
    def f2_ppm_max(self):
        return max(self.f2_ppm_limits)

    @property
    def f2_ppm_min(self):
        return min(self.f2_ppm_limits)

    @property
    def f2_ppm_0(self):
        return self.f2_ppm_limits[0]

    @property
    def f2_ppm_1(self):
        return self.f2_ppm_limits[1]

    @property
    def f1_ppm_0(self):
        return self.f1_ppm_limits[0]

    @property
    def f1_ppm_1(self):
        return self.f1_ppm_limits[1]


class UnknownFormat(Exception):
    pass



class Peaklist(Pseudo3D):
    """Read analysis, sparky or NMRPipe peak list and convert to NMRPipe-ish format also find peak clusters

    Parameters
    ----------
    path : path-like or str
        path to peaklist
    data_path : ndarray
        NMRPipe format data
    fmt : str
        a2|a3|sparky|pipe
    dims: list
        [planes,y,x]
    radii: list
        [x,y] Mask radii in ppm


    Methods
    -------

    clusters :
    mask_method :
    adaptive_clusters :

    Returns
    -------
    df : pandas DataFrame
        dataframe containing peaklist

    """

    def __init__(
        self,
        path,
        data_path,
        fmt: PeaklistFormat = PeaklistFormat.a2,
        dims=[0, 1, 2],
        radii=[0.04, 0.4],
        posF1="Position F2",
        posF2="Position F1",
        verbose=False,
    ):
        dic, data = ng.pipe.read(data_path)
        Pseudo3D.__init__(self, dic, data, dims)
        self.fmt = fmt
        self.peaklist_path = path
        self.data_path = data_path
        self.verbose = verbose
        self._radii = radii
        self._thres = None
        if self.verbose:
            print(
                "Points per hz f1 = %.3f, f2 = %.3f"
                % (self.pt_per_hz_f1, self.pt_per_hz_f2)
            )

        self._analysis_to_pipe_dic = {
            "#": "INDEX",
            "Position F1": "X_PPM",
            "Position F2": "Y_PPM",
            "Line Width F1 (Hz)": "XW_HZ",
            "Line Width F2 (Hz)": "YW_HZ",
            "Height": "HEIGHT",
            "Volume": "VOL",
        }
        self._assign_to_pipe_dic = {
            "#": "INDEX",
            "Pos F1": "X_PPM",
            "Pos F2": "Y_PPM",
            "LW F1 (Hz)": "XW_HZ",
            "LW F2 (Hz)": "YW_HZ",
            "Height": "HEIGHT",
            "Volume": "VOL",
        }

        self._sparky_to_pipe_dic = {
            "index": "INDEX",
            "w1": "X_PPM",
            "w2": "Y_PPM",
            "lw1 (hz)": "XW_HZ",
            "lw2 (hz)": "YW_HZ",
            "Height": "HEIGHT",
            "Volume": "VOL",
            "Assignment": "ASS",
        }

        self._analysis_to_pipe_dic[posF1] = "Y_PPM"
        self._analysis_to_pipe_dic[posF2] = "X_PPM"

        self._df = self.read_peaklist()

    def read_peaklist(self):
        match self.fmt:
            case self.fmt.a2:
                self._df = self._read_analysis()

            case self.fmt.a3:
                self._df = self._read_assign()

            case self.fmt.sparky:
                self._df = self._read_sparky()

            case self.fmt.pipe:
                self._df = self._read_pipe()
            
            case self.fmt.csv:
                self._df = self._read_csv()

            case _:
                raise UnknownFormat("I don't know this format: {self.fmt}")

        return self._df

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df
        return self._df

    @property
    def radii(self):
        return self._radii

    def check_radius_contains_enough_points_for_fitting(self, radius, pt_per_ppm, flag):
        if (radius * pt_per_ppm) < 2.0:
            new_radius = 2.0 * (1./ pt_per_ppm)
            print(
                "\n",
                f"[red]Warning: {flag} is set to {radius:.3f} ppm which is {radius * pt_per_ppm:.3f} points[/red]" + "\n",
                f"[yellow]Setting to 2 points which is {new_radius:.3f} ppm[/yellow]" + "\n",
                f"[yellow]Consider increasing this value to improve robustness of fitting (or increase zero filling)[/yellow]" + "\n",
            )
        else:
            new_radius = radius
        return new_radius

    @property
    def f2_radius(self):
        """radius for fitting mask in f2"""
        _f2_radius = self.check_radius_contains_enough_points_for_fitting(self.radii[0], self.pt_per_ppm_f2, "--x-radius-ppm")
        return _f2_radius

    @property
    def f1_radius(self):
        """radius for fitting mask in f1"""
        _f1_radius = self.check_radius_contains_enough_points_for_fitting(self.radii[1], self.pt_per_ppm_f1, "--y-radius-ppm")
        return _f1_radius

    @property
    def analysis_to_pipe_dic(self):
        return self._analysis_to_pipe_dic

    @property
    def assign_to_pipe_dic(self):
        return self._assign_to_pipe_dic

    @property
    def sparky_to_pipe_dic(self):
        return self._sparky_to_pipe_dic

    @property
    def thres(self):
        if self._thres == None:
            self._thres = abs(threshold_otsu(self.data[0]))
            return self._thres
        else:
            return self._thres

    def validate_peaklist(self):
        self.df = pd.DataFrame(
            [
                PeaklistColumns(**i).model_dump()
                for i in self.df.to_dict(orient="records")
            ]
        )
        return self.df

    def update_df(self):
        # int point value
        self.df["X_AXIS"] = self.df.X_PPM.apply(lambda x: self.uc_f2(x, "ppm"))
        self.df["Y_AXIS"] = self.df.Y_PPM.apply(lambda x: self.uc_f1(x, "ppm"))
        # decimal point value
        self.df["X_AXISf"] = self.df.X_PPM.apply(lambda x: self.uc_f2.f(x, "ppm"))
        self.df["Y_AXISf"] = self.df.Y_PPM.apply(lambda x: self.uc_f1.f(x, "ppm"))
        # in case of missing values (should estimate though)
        self.df["XW_HZ"] = self.df.XW_HZ.replace("None", "20.0")
        self.df["YW_HZ"] = self.df.YW_HZ.replace("None", "20.0")
        self.df["XW_HZ"] = self.df.XW_HZ.replace(np.nan, "20.0")
        self.df["YW_HZ"] = self.df.YW_HZ.replace(np.nan, "20.0")
        # convert linewidths to float
        self.df["XW_HZ"] = self.df.XW_HZ.apply(lambda x: float(x))
        self.df["YW_HZ"] = self.df.YW_HZ.apply(lambda x: float(x))
        # convert Hz lw to points
        self.df["XW"] = self.df.XW_HZ.apply(lambda x: x * self.pt_per_hz_f2)
        self.df["YW"] = self.df.YW_HZ.apply(lambda x: x * self.pt_per_hz_f1)
        # makes an assignment column from Assign F1 and Assign F2 columns
        # in analysis2.x and ccpnmr v3 assign peak lists
        if self.fmt in [PeaklistFormat.a2, PeaklistFormat.a3]:
            self.df["ASS"] = self.df.apply(
                # lambda i: "".join([i["Assign F1"], i["Assign F2"]]), axis=1
                lambda i: f"{i['Assign F1']}_{i['Assign F2']}",
                axis=1,
            )

        # make default values for X and Y radii for fit masks
        self.df["X_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f2_radius
        self.df["Y_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f1_radius
        self.df["X_RADIUS"] = self.df.X_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f2
        )
        self.df["Y_RADIUS"] = self.df.Y_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f1
        )
        # add include column
        if "include" in self.df.columns:
            pass
        else:
            self.df["include"] = self.df.apply(lambda x: "yes", axis=1)

        # check assignments for duplicates
        self.check_assignments()
        # check that peaks are within the bounds of the data
        self.check_peak_bounds()
        self.validate_peaklist()

    def add_fix_bound_columns(self):
        """add columns containing parameter bounds (param_upper/param_lower)
        and whether or not parameter should be fixed (yes/no)

        For parameter bounding:

            Column names are <param_name>_upper and <param_name>_lower for upper and lower bounds respectively.
            Values are given as floating point. Value of 0.0 indicates that parameter is unbounded
            X/Y positions are given in ppm
            Linewidths are given in Hz

        For parameter fixing:

            Column names are <param_name>_fix.
            Values are given as a string 'yes' or 'no'

        """
        pass

    def _read_analysis(self):
        df = pd.read_csv(self.peaklist_path, delimiter="\t")
        new_columns = [self.analysis_to_pipe_dic.get(i, i) for i in df.columns]
        pipe_columns = dict(zip(df.columns, new_columns))
        df = df.rename(index=str, columns=pipe_columns)

        return df

    def _read_assign(self):
        df = pd.read_csv(self.peaklist_path, delimiter="\t")
        new_columns = [self.assign_to_pipe_dic.get(i, i) for i in df.columns]
        pipe_columns = dict(zip(df.columns, new_columns))
        df = df.rename(index=str, columns=pipe_columns)

        return df

    def _read_sparky(self):
        df = pd.read_csv(
            self.peaklist_path,
            skiprows=1,
            sep=r"\s+",
            names=["ASS", "Y_PPM", "X_PPM"],
            # use only first three columns
            usecols=[i for i in range(3)],
        )
        df["INDEX"] = df.index
        # need to add LW estimate
        df["XW_HZ"] = 20.0
        df["YW_HZ"] = 20.0
        # dummy values
        df["HEIGHT"] = 0.0
        df["VOL"] = 0.0
        return df

    def _read_pipe(self):
        to_skip = 0
        with open(self.peaklist_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("VARS"):
                    columns = line.strip().split()[1:]
                elif line[:5].strip(" ").isdigit():
                    break
                else:
                    to_skip += 1
        df = pd.read_csv(
            self.peaklist_path, skiprows=to_skip, names=columns, sep=r"\s+"
        )
        return df

    def _read_csv(self):
        """ Read a csv file containing peaklist data 
        
        Requires the following columns:
            X_PPM: ppm position of peak in X axis
            Y_PPM: ppm position of peak in Y axis
            ASS: assignment of peak
        Optional columns include:
            XW_HZ: estimated X axis linewidth in HZ
            YW_HZ: estimated Y axis linewidth in HZ
            VOL: peak volume
            Height: peak height
        """
        df = pd.read_csv(self.peaklist_path)
        df["INDEX"] = df.index
        # need to add LW estimate
        if not "XW_HZ" in df.columns:
            df["XW_HZ"] = 20.0
        if not "YW_HZ" in df.columns:
            df["YW_HZ"] = 20.0
        # dummy values
        if not "HEIGHT" in df.columns:
            df["HEIGHT"] = 0.0
        if not "VOL" in df.columns:
            df["VOL"] = 0.0
        return df

    def check_assignments(self):
        # self.df["ASS"] = self.df.
        self.df["ASS"] = self.df.ASS.astype(object)
        self.df.loc[self.df["ASS"].isnull(), "ASS"] = "None_dummy_0"
        self.df["ASS"] = self.df.ASS.astype(str)
        duplicates_bool = self.df.ASS.duplicated()
        duplicates = self.df.ASS[duplicates_bool]
        if len(duplicates) > 0:
            console.print(
                textwrap.dedent(
                    """
                #############################################################################
                    You have duplicated assignments in your list...
                    Currently each peak needs a unique assignment. Sorry about that buddy...
                #############################################################################
                """
                ),
                style="yellow",
            )
            self.df.loc[duplicates_bool, "ASS"] = [
                f"{i}_dummy_{num+1}" for num, i in enumerate(duplicates)
            ]
            if self.verbose:
                print("Here are the duplicates")
                print(duplicates)
                print(self.df.ASS)

            print(
                textwrap.dedent(
                    """
                    Creating dummy assignments for duplicates

                """
                )
            )

    def check_peak_bounds(self):
        columns_to_print = ["INDEX", "ASS", "X_AXIS", "Y_AXIS", "X_PPM", "Y_PPM"]
        # check that peaks are within the bounds of spectrum
        within_x = (self.df.X_PPM < self.f2_ppm_max) & (self.df.X_PPM > self.f2_ppm_min)
        within_y = (self.df.Y_PPM < self.f1_ppm_max) & (self.df.Y_PPM > self.f1_ppm_min)
        self.excluded = self.df[~(within_x & within_y)]
        self.df = self.df[within_x & within_y]
        if len(self.excluded) > 0:
            print(
                textwrap.dedent(
                    f"""[red]
                    #################################################################################

                    Excluding the following peaks as they are not within the spectrum which has shape

                    {self.data.shape}
                [/red]"""
                )
            )
            table_to_print = df_to_rich_table(
                self.excluded,
                title="Excluded",
                columns=columns_to_print,
                styles=["red" for i in columns_to_print],
            )
            print(table_to_print)
            print(
                "[red]#################################################################################[/red]"
            )

    def clusters(
        self,
        thres=None,
        struc_el: StrucEl = StrucEl.disk,
        struc_size=(3,),
        l_struc=None,
    ):
        """Find clusters of peaks

        :param thres: threshold for positive signals above which clusters are selected. If None then threshold_otsu is used
        :type thres: float

        :param struc_el: 'square'|'disk'|'rectangle'
            structuring element for binary_closing of thresholded data can be square, disc or rectangle
        :type struc_el: str

        :param struc_size: size/dimensions of structuring element
            for square and disk first element of tuple is used (for disk value corresponds to radius)
            for rectangle, tuple corresponds to (width,height).
        :type struc_size: tuple


        """
        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]

        if thres == None:
            thres = self.thres
            self._thres = abs(threshold_otsu(self.data[0]))
        else:
            self._thres = thres

        # get positive and negative
        thresh_data = np.bitwise_or(
            self.data[0] < (self._thres * -1.0), self.data[0] > self._thres
        )

        match struc_el:
            case struc_el.disk:
                radius = struc_size[0]
                if self.verbose:
                    print(f"using disk with {radius}")
                closed_data = binary_closing(thresh_data, disk(int(radius)))

            case struc_el.square:
                width = struc_size[0]
                if self.verbose:
                    print(f"using square with {width}")
                closed_data = binary_closing(thresh_data, footprint_rectangle((int(width),int(width))))

            case struc_el.rectangle:
                width, height = struc_size
                if self.verbose:
                    print(f"using rectangle with {width} and {height}")
                closed_data = binary_closing(
                    thresh_data, footprint_rectangle((int(width), int(height)))
                )

            case _:
                if self.verbose:
                    print(f"Not using any closing function")
                closed_data = thresh_data

        labeled_array, num_features = ndimage.label(closed_data, l_struc)

        self.df.loc[:, "CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

        # count how many peaks per cluster
        for ind, group in self.df.groupby("CLUSTID"):
            self.df.loc[group.index, "MEMCNT"] = len(group)

        self.df.loc[:, "color"] = self.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )
        return ClustersResult(labeled_array, num_features, closed_data, peaks)

    def mask_method(self, overlap=1.0, l_struc=None):
        """connect clusters based on overlap of fitting masks

        :param overlap: fraction of mask for which overlaps are calculated
        :type overlap: float

        :returns ClusterResult: Instance of ClusterResult
        :rtype: ClustersResult
        """
        # overlap is positive
        overlap = abs(overlap)

        self._thres = threshold_otsu(self.data[0])

        mask = np.zeros(self.data[0].shape, dtype=bool)

        for ind, peak in self.df.iterrows():
            mask += make_mask(
                self.data[0],
                peak.X_AXISf,
                peak.Y_AXISf,
                peak.X_RADIUS * overlap,
                peak.Y_RADIUS * overlap,
            )

        peaks = [[y, x] for y, x in zip(self.df.Y_AXIS, self.df.X_AXIS)]
        labeled_array, num_features = ndimage.label(mask, l_struc)

        self.df.loc[:, "CLUSTID"] = [labeled_array[i[0], i[1]] for i in peaks]

        #  renumber "0" clusters
        max_clustid = self.df["CLUSTID"].max()
        n_of_zeros = len(self.df[self.df["CLUSTID"] == 0]["CLUSTID"])
        self.df.loc[self.df[self.df["CLUSTID"] == 0].index, "CLUSTID"] = np.arange(
            max_clustid + 1, n_of_zeros + max_clustid + 1, dtype=int
        )

        # count how many peaks per cluster
        for ind, group in self.df.groupby("CLUSTID"):
            self.df.loc[group.index, "MEMCNT"] = len(group)

        self.df.loc[:, "color"] = self.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )

        return ClustersResult(labeled_array, num_features, mask, peaks)

    def to_fuda(self):
        fname = self.peaklist_path.parent / "params.fuda"
        with open(self.peaklist_path.parent / "peaks.fuda", "w") as peaks_fuda:
            for ass, f1_ppm, f2_ppm in zip(self.df.ASS, self.df.Y_PPM, self.df.X_PPM):
                peaks_fuda.write(f"{ass}\t{f1_ppm:.3f}\t{f2_ppm:.3f}\n")
        groups = self.df.groupby("CLUSTID")
        fuda_params = Path(fname)
        overlap_peaks = ""

        for ind, group in groups:
            if len(group) > 1:
                overlap_peaks_str = ";".join(group.ASS)
                overlap_peaks += f"OVERLAP_PEAKS=({overlap_peaks_str})\n"

        fuda_file = textwrap.dedent(
            f"""\

# Read peaklist and spectrum info
PEAKLIST=peaks.fuda
SPECFILE={self.data_path}
PARAMETERFILE=(bruker;vclist)
ZCORR=ncyc
NOISE={self.thres} # you'll need to adjust this
BASELINE=N
VERBOSELEVEL=5
PRINTDATA=Y
LM=(MAXFEV=250;TOL=1e-5)
#Specify the default values. All values are in ppm:
DEF_LINEWIDTH_F1={self.f1_radius}
DEF_LINEWIDTH_F2={self.f2_radius}
DEF_RADIUS_F1={self.f1_radius}
DEF_RADIUS_F2={self.f2_radius}
SHAPE=GLORE
# OVERLAP PEAKS
{overlap_peaks}"""
        )
        with open(fuda_params, "w") as f:
            print(f"Writing FuDA file {fuda_file}")
            f.write(fuda_file)
        if self.verbose:
            print(overlap_peaks)


class ClustersResult:
    """Class to store results of clusters function"""

    def __init__(self, labeled_array, num_features, closed_data, peaks):
        self._labeled_array = labeled_array
        self._num_features = num_features
        self._closed_data = closed_data
        self._peaks = peaks

    @property
    def labeled_array(self):
        return self._labeled_array

    @property
    def num_features(self):
        return self._num_features

    @property
    def closed_data(self):
        return self._closed_data

    @property
    def peaks(self):
        return self._peaks


class LoadData(Peaklist):
    """Load peaklist data from peakipy .csv file output from either peakipy read or edit

    read_peaklist is redefined to just read a .csv file

    check_data_frame makes sure data frame is in good shape for setting up fits

    """

    def read_peaklist(self):
        if self.peaklist_path.suffix == ".csv":
            self.df = pd.read_csv(self.peaklist_path)  # , comment="#")

        elif self.peaklist_path.suffix == ".tab":
            self.df = pd.read_csv(self.peaklist_path, sep="\t")  # comment="#")

        else:
            self.df = pd.read_pickle(self.peaklist_path)

        self._thres = threshold_otsu(self.data[0])

        return self.df

    def validate_peaklist(self):
        self.df = pd.DataFrame(
            [
                PeaklistColumnsWithClusters(**i).model_dump()
                for i in self.df.to_dict(orient="records")
            ]
        )
        return self.df

    def check_data_frame(self):
        """
        Ensure the data frame has all required columns and add necessary derived columns for fitting.
        
        Returns
        -------
        pd.DataFrame
            The modified DataFrame after validation.
        """    # make diameter columns
        if "X_DIAMETER_PPM" in self.df.columns:
            pass
        else:
            self.df["X_DIAMETER_PPM"] = self.df["X_RADIUS_PPM"] * 2.0
            self.df["Y_DIAMETER_PPM"] = self.df["Y_RADIUS_PPM"] * 2.0

        #  make a column to track edited peaks
        if "Edited" in self.df.columns:
            pass
        else:
            self.df["Edited"] = np.zeros(len(self.df), dtype=bool)

        # create include column if it doesn't exist
        if "include" in self.df.columns:
            pass
        else:
            self.df["include"] = self.df.apply(lambda _: "yes", axis=1)

        # color clusters
        self.df["color"] = self.df.apply(
            lambda x: Category20[20][int(x.CLUSTID) % 20] if x.MEMCNT > 1 else "black",
            axis=1,
        )

        # get rid of unnamed columns
        unnamed_cols = [i for i in self.df.columns if "Unnamed:" in i]
        self.df = self.df.drop(columns=unnamed_cols)

    def update_df(self):
        """Slightly modified to retain previous configurations"""
        # int point value
        self.df["X_AXIS"] = self.df.X_PPM.apply(lambda x: self.uc_f2(x, "ppm"))
        self.df["Y_AXIS"] = self.df.Y_PPM.apply(lambda x: self.uc_f1(x, "ppm"))
        # decimal point value
        self.df["X_AXISf"] = self.df.X_PPM.apply(lambda x: self.uc_f2.f(x, "ppm"))
        self.df["Y_AXISf"] = self.df.Y_PPM.apply(lambda x: self.uc_f1.f(x, "ppm"))
        # in case of missing values (should estimate though)
        self.df["XW_HZ"] = self.df.XW_HZ.replace(np.nan, "20.0")
        self.df["YW_HZ"] = self.df.YW_HZ.replace(np.nan, "20.0")
        # convert linewidths to float
        self.df["XW_HZ"] = self.df.XW_HZ.apply(lambda x: float(x))
        self.df["YW_HZ"] = self.df.YW_HZ.apply(lambda x: float(x))
        # convert Hz lw to points
        self.df["XW"] = self.df.XW_HZ.apply(lambda x: x * self.pt_per_hz_f2)
        self.df["YW"] = self.df.YW_HZ.apply(lambda x: x * self.pt_per_hz_f1)
        # makes an assignment column
        if self.fmt == "a2":
            self.df["ASS"] = self.df.apply(
                lambda i: "".join([i["Assign F1"], i["Assign F2"]]), axis=1
            )

        # make default values for X and Y radii for fit masks
        # self.df["X_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f2_radius
        # self.df["Y_RADIUS_PPM"] = np.zeros(len(self.df)) + self.f1_radius
        self.df["X_RADIUS"] = self.df.X_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f2
        )
        self.df["Y_RADIUS"] = self.df.Y_RADIUS_PPM.apply(
            lambda x: x * self.pt_per_ppm_f1
        )
        # add include column
        if "include" in self.df.columns:
            pass
        else:
            self.df["include"] = self.df.apply(lambda x: "yes", axis=1)

        # check assignments for duplicates
        self.check_assignments()
        # check that peaks are within the bounds of the data
        self.check_peak_bounds()
        self.validate_peaklist()


def get_vclist(vclist, args):
    # read vclist
    if vclist is None:
        vclist = False
    elif vclist.exists():
        vclist_data = np.genfromtxt(vclist)
        args["vclist_data"] = vclist_data
        vclist = True
    else:
        raise Exception("vclist not found...")

    args["vclist"] = vclist
    return args
