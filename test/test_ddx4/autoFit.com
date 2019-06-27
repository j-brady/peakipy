#!/bin/csh -f

set specName    = test.ft2
set simName     = sim.ft2
set simDir      = .
set difName     = dif.ft2
set difDir      = .
set inTabName   = cluster.tab
set auxTabName  = axt.tab
set outTabName  = nlin.tab
set errTabName  = err.nlin.tab
set noiseRMS    = 9841.0
set specCount   = 1

set aRegSizeX  = 4
set dRegSizeX  = 4
set maxDX      = 1.0
set minXW      = 0.0
set maxXW      = 5.48
set simSizeX   = 546

set aRegSizeY  = 4
set dRegSizeY  = 4
set maxDY      = 1.0
set minYW      = 0.0
set maxYW      = 5.61
set simSizeY   = 256


cp $inTabName $auxTabName


nlinLS -tol 1.0e-8 -maxf 750 -iter 750 \
 -in $auxTabName -out $outTabName -data $specName \
 -apod  None \
 -noise $noiseRMS \
 -mod    GAUSS1D  GAUSS1D  \
 -delta  X_AXIS $maxDX  Y_AXIS $maxDY  \
 -limit  XW $minXW $maxXW YW $minYW $maxYW \
 -w      $dRegSizeX  $dRegSizeY  \
 -nots -norm -ppm

if (!(-e $simDir)) then
   mkdir $simDir
endif

nmrPipe -in $specName -out $simName -fn SET -r 0.0 -verb -ov

simSpecND -in $outTabName -data $simName \
          -mod   GAUSS1D  GAUSS1D  \
          -w     $simSizeX  $simSizeY  \
          -apod None -nots -verb


if (!(-e $difDir)) then
   mkdir $difDir
endif

addNMR -in1 $specName -in2 $simName -out $difName -sub
