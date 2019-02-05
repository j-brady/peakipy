# read peaklist and make clusters
#read_peaklist.py peaks.a2 test.ft2 --a2 --noise=9841. --pthres=1.5e6 --show --ndil=1  
#read_peaklist.py peaks.a2 test.ft2 --a2 --noise=9841. --pthres=1.5e6 --show --clust2 --c2thres=0.075
read_peaklist.py peaks.a2 test.ft2 --a2 --noise=9841. --pthres=1.5e6 --show --clust3 --struc_el=square --struc_size=3,4
# fit peaks
#fit_pipe_peaks.py peaks.pkl test.ft2 fits.csv --x_radius=0.075 --y_radius=0.2 --plot=out --show --max_cluster_size=10
