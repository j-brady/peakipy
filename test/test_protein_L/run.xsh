# read peaklist and make clusters
#read_peaklist.py peaks.a2 test1.ft2 --a2 --show  --pthres=1e6
# fit peaks
#fit_peaks_mp.py peaks.pkl test1.ft2 fits.csv --x_radius=0.050 --y_radius=0.5 --vclist=vclist #--plot=out #--show  

# Run test and comparison with NMRPipe
import pandas as pd

# read pipe peaklist
read_peaklist test.tab test1.ft2 --pipe --f1radius=0.35 --f2radius=0.035

# convert Nlin tab to peakipy 
read_peaklist nlin.tab test1.ft2 --pipe

# fit peaks with peakipy
fit_peaks test.csv test1.ft2 fits.csv --lineshape=G

python compare.py
