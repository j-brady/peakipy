# read peaklist and make clusters
#read_peaklist.py peaks.a2 test1.ft2 --a2 --noise=3.0365e4 --pthres=3e6 --show   
#read_peaklist.py peaks.a2 test1.ft2 --a2 --show  --pthres=1e6
# fit peaks
#fit_peaks_mp.py peaks.pkl test1.ft2 fits.csv --x_radius=0.050 --y_radius=0.5 --vclist=vclist #--plot=out #--show  

# Run test and comparison with NMRPipe
import pandas as pd

read_peaklist.py _test.tab test1.ft2 --pipe --f1radius=0.30 --f2radius=0.030

peaks = pd.read_csv("_test.csv")
peaks["ASS"] = ["_%d"%i for i in range(len(peaks))]
peaks.to_csv("_test.csv",sep=',',index=False)

fit_peaks.py _test.csv test1.ft2 _fits.csv --lineshape=PV
python compare.py
