# read peaklist and make clusters
#read_peaklist.py peaks.a2 test1.ft2 --a2 --noise=3.0365e4 --pthres=3e6 --show   
read_peaklist.py peaks.a2 test1.ft2 --a2 --clust3 --show   
# fit peaks
#fit_pipe_peaks.py peaks.pkl test1.ft2 fits.csv --x_radius=0.03 --y_radius=0.3 #--plot=out --show 
