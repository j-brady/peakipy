""" Plot fits
    
    Usage:
        plot_all.py <fits> <nmrdata> [options]

    Options:
        --dims=<id,f1,f2>  Dimension order [default: 0,1,2]
"""
from pathlib import Path

import pandas as pd
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from docopt import docopt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages


from peakipy.core import make_mask, pvoigt2d, make_models, Pseudo3D


if __name__ == "__main__":
    args = docopt(__doc__)
    fits = Path(args.get("<fits>"))
    data_path = args.get("<nmrdata>")
    dims = args.get("--dims")
    dims = [int(i) for i in eval(dims)]

    fits = pd.read_csv(fits)
    groups = fits.groupby("clustid")
    dic, data = ng.pipe.read(data_path)
    pseudo3D = Pseudo3D(dic, data, dims)


    x = np.arange(1, pseudo3D.f2_size + 1) 
    y = np.arange(1, pseudo3D.f1_size + 1)
    XY = np.meshgrid(x,y)
    X,Y = XY


    with PdfPages("plots.pdf") as pdf:
        for ind, group in groups:
            mask = np.zeros((pseudo3D.f1_size,pseudo3D.f2_size),dtype=bool)
            sim_data = np.zeros((pseudo3D.f1_size,pseudo3D.f2_size)) 

            first_plane = group[group.plane==0]

            x_radius = group.x_radius.max()
            y_radius = group.y_radius.max()
            max_x, min_x = (
                int(np.ceil(max(group.center_x) + x_radius + 2)),
                int(np.floor(min(group.center_x) - x_radius - 1)),
            )
            max_y, min_y = (
                int(np.ceil(max(group.center_y) + y_radius + 2)),
                int(np.floor(min(group.center_y) - y_radius - 1)),
            )

            for cx,cy,rx,ry in zip(first_plane.center_x,first_plane.center_y,first_plane.x_radius,first_plane.y_radius):
                #plt.plot(cx,cy,"o")
                #print(cx,cy,rx,ry)
                # have to minus 1!
                mask += make_mask(mask,cx-1,cy-1,rx,ry)

            for plane_id, plane in group.groupby("plane"):

                sim_data = np.zeros((pseudo3D.f1_size,pseudo3D.f2_size)) 
                for amp, c_x, c_y, s_x, s_y, frac in zip(plane.amp,plane.center_x,plane.center_y,plane.sigma_x,plane.sigma_y,plane.fraction):
                    print(amp)
                    shape = sim_data.shape
                    sim_data += pvoigt2d(XY,amp,c_x,c_y,s_x,s_y,frac).reshape(shape)



                masked_data = pseudo3D.data[plane_id].copy()
                masked_data[~mask] = np.nan
                sim_data[~mask] = np.nan
             
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                # slice out plot area
                x_plot = pseudo3D.uc_f2.ppm(X[min_y:max_y, min_x:max_x])
                y_plot = pseudo3D.uc_f1.ppm(Y[min_y:max_y, min_x:max_x])

                masked_data = masked_data[min_y:max_y, min_x:max_x]
                sim_plot = sim_data[min_y:max_y, min_x:max_x]

                ax.plot_wireframe(
                    x_plot, y_plot, sim_plot, colors="r", linestyle="--", label="fit"
                )
                ax.plot_wireframe(
                    x_plot, y_plot, masked_data, colors="k", linestyle="-", label="data"
                )
                plt.legend()
                names = ",".join(plane.assignment)
                plt.title(f"Plane={plane_id},Cluster={plane.clustid.iloc[0]}")
                out_str = "Amplitudes\n----------------\n"
                for amp, name in zip(plane.amp,plane.assignment):
                    out_str += f"{name} = {amp:.3e}\n"

                ax.text2D(-0.15,1.0, out_str, transform= ax.transAxes, fontsize=10,va="top")
                pdf.savefig()
                plt.close()
