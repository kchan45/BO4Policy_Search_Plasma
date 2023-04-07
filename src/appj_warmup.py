# this starts up the APPJ testbed (for warm-up) can also check connection to
# measurement devices
#
# Requirements:
# * Python 3
# * several 3rd party packages including CasADi, NumPy, Scikit-Optimize for
# the implemented algorithms and Seabreeze, os, serial, etc. for connection to
# the experimental setup.
#
# Copyright (c) 2021 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
# Affiliation: University of California, Berkeley
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

## import 3rd party packages
import sys
sys.dont_write_bytecode = True
import numpy as np
from seabreeze.spectrometers import Spectrometer, list_devices
import time
import os
import serial
import cv2
import asyncio

## import user functions
import KCutils.APPJPythonFunctions as appj
from KCutils.APPJPythonFunctions import sendInputsArduino

def close_instr(instr, dev, ctx, spec):
    # print("terminated prematurely... closing instruments")
    instr.close()
    appj.closeThermalCamera(dev, ctx)
    spec.close()
    print("closed devices")


################################################################################
## Startup/prepare APPJ
################################################################################
if __name__=="__main__":
    # configure run options
    runOpts = appj.RunOpts()
    runOpts.collectData = False
    runOpts.collectEntireSpectra = False
    runOpts.collectOscMeas = False
    runOpts.collectSpatialTemp = False
    runOpts.saveSpectra = False
    runOpts.saveOscMeas = False
    runOpts.saveSpatialTemp = False
    runOpts.tSampling = 0.5

    ## Set startup values
    dutyCycleIn = 100
    powerIn = 2.0
    flowIn = 3.0

    ## connect to/open connection to devices in setup
    # Arduino
    arduinoAddress = appj.getArduinoAddress(os="ubuntu")
    print("Arduino Address:", arduinoAddress)
    arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
    s = time.time()
    # Oscilloscope
    oscilloscope = appj.Oscilloscope()       # Instantiate object from class
    instr = oscilloscope.initialize()	# Initialize oscilloscope
    # Spectrometer
    devices = list_devices()
    print(devices)
    spec = Spectrometer(devices[0])
    spec.integration_time_micros(12000*6)
    # Thermal Camera
    dev, ctx = appj.openThermalCamera()
    print("Devices opened/connected to sucessfully!")

    devices = {}
    devices['arduinoPI'] = arduinoPI
    devices['arduinoAddress'] = arduinoAddress
    devices['instr'] = instr
    devices['spec'] = spec

    # send startup inputs
    time.sleep(2)
    sendInputsArduino(arduinoPI, powerIn, flowIn, dutyCycleIn, arduinoAddress)
    time.sleep(2)
    input("Ensure plasma has ignited and press Return to begin.\n")

    ## Startup asynchronous measurement
    if os.name == 'nt':
        ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
        asyncio.set_event_loop(ioloop)
    else:
        ioloop = asyncio.get_event_loop()
    # run once to initialize measurements
    prevTime = (time.time()-s)*1e3
    tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
    print('measurement devices ready!')
    s = time.time()

    # let APPJ run for a bit
    time.sleep(2)
    sendInputsArduino(arduinoPI, 2.0, 1.5, dutyCycleIn, arduinoAddress)
    sendInputsArduino(arduinoPI, 2.0, 1.5, dutyCycleIn, arduinoAddress)
    time.sleep(2)

    print("Waiting 15 minutes to warm up the plasma jet...\n")
    time.sleep(60*5)
    print("10 minutes left...")
    time.sleep(60*5)
    print("5 minutes left...")
    time.sleep(60*5)
    print("15 minutes have passed!")

    sendInputsArduino(arduinoPI, 0.0, 0.0, dutyCycleIn, arduinoAddress)
    close_instr(instr, dev, ctx, spec)
