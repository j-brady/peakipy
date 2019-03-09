#!/bin/csh
nmrPipe -in test1.ft2\
| nmrPipe -fn ZTP \
-out test_tp.ft2 -verb -ov 

nmrPipe -in test_tp.ft2\
| nmrPipe -fn TP \
-out test_tp2.ft2 -verb -ov
