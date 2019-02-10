# read peaklist and make clusters
#read_peaklist.py peaks.a2 test1.ft2 --a2 --noise=3.0365e4 --pthres=3e6 --show   
#read_peaklist.py peaks.a2 test1.ft2 --a2 --show  --pthres=1e6
# fit peaks
fit_peaks_mp.py peaks.pkl test1.ft2 fits.csv --x_radius=0.050 --y_radius=0.5 --vclist=vclist #--plot=out #--show  
