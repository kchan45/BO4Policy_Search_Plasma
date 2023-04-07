import sys
import argparse
from seabreeze.spectrometers import Spectrometer, list_devices
import numpy as np
import matplotlib.pyplot as plt
import time
from KCutils.APPJPythonFunctions import*
print('\n--------------------------------')

# Check that the number of arguments is correct
numArg = 2
if len(sys.argv)!=numArg:
	print("Function expects "+str(numArg-1)+" argument(s). Example: 'spectroscopyLive.py 30' measures spectrometer for 30 seconds")
	exit()

# Parameters
loopTime = int(sys.argv[1])

# Obtain the spectrometer object
devices = list_devices()
print(devices)
spec = Spectrometer(devices[0])
spec.integration_time_micros(12000*6)

# Generate live plot
plt.ion()

# Start counting the time
tStart = time.time()

# Update the live graph
while(time.time()-tStart<=loopTime):
	wavelengthsPlot = spec.wavelengths()[20:]
	intensityPlot = spec.intensities()[20:]
	totalIntensity = sum(intensityPlot)
	print("Total Intensity = ", totalIntensity)
	plt.plot(wavelengthsPlot,intensityPlot)
	plt.draw()
	plt.pause(1)
	plt.clf()
