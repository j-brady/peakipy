.PHONY: ccpnmrv3 a2 pipe sparky

a2:
	poetry run peakipy read peaks.a2 test1.ft2 a2
	poetry run peakipy fit peaks.csv test1.ft2 fits_a2.csv

pipe:
	poetry run peakipy read test.tab test1.ft2 --pipe
	poetry run peakipy fit test.csv test1.ft2 fits_pipe.csv

sparky:
	poetry run peakipy read peaks.sparky test1.ft2 --sparky
	poetry run peakipy fit peaks.csv test1.ft2 fits_sparky.csv

ccpnmrv3:
	poetry run peakipy read ccpnTable.tsv test1.ft2 --a3
	#poetry run peakipy edit ccpnTable.csv test1.ft2
	poetry run peakipy fit ccpnTable.csv test1.ft2 fits_a3.csv

all: a2 sparky pipe ccpnmrv3
