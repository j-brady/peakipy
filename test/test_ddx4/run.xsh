# read peaklist and make clusters
#read_peaklist.py peaks.a2 test.ft2 --a2 --noise=9841. --pthres=1.5e6 --show --ndil=1  
#read_peaklist.py peaks.a2 test.ft2 --a2 --show --struc_el=disk --outfmt=csv #--pthres=2.3e6
run_check_fits.py peaks.csv test.ft2
#read_peaklist.py peaks.a2 test.ft2 --a2 --noise=9841. --pthres=1.5e6 --show --clust2 --c2thres=0.075
#read_peaklist.py peaks.a2 test.ft2 --a2 --noise=9841. --pthres=1.5e6 --show --clust3 --struc_el=square --struc_size=3,4
# fit peaks
#fit_peaks.py peaks.pkl test.ft2 fits.csv --x_radius=0.075 --y_radius=0.2 --plot=out --show --max_cluster_size=10
#fit_peaks.py peaks.csv test.ft2 fits.csv --x_radius=0.075 --y_radius=0.2 --max_cluster_size=10 --plot=out --show 
