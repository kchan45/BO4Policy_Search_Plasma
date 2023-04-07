import subprocess
import time
import os
import cv2
import numpy as np
# from applescript import tell
import pyvisa as visa
import usbtmc
from KCutils.uvcRadiometry import*
# new imports since 2021/03/17:
import asyncio
import crcmod
crc8 = crcmod.predefined.mkCrcFun('crc-8-maxim')
# new imports since 2021/04/16:
import serial
from seabreeze.spectrometers import Spectrometer, list_devices

# Define constants
NORMALIZATION = 25000

##################################################################################################################
# PARAMETERS
##################################################################################################################
class Parameters():
	def __init__(self):
		self.INORM = 25000
		#[...]

class RunOpts():
	'''
	Class for run options of an experiment. Users can specify data collection
	options and save options for data (by default, all values are True)
	'''
	def __init__(self):
		# collect Data options
		self.collectData = True # collects Ts and total Intensity
		self.collectSpatialTemp = True # collects average spatial temps and entire image capture
		self.collectEntireSpectra = True # collects entire intensity spectrum
		self.collectOscMeas = True # collects oscilloscope measurements
		self.collectEmbedded = True # collects embedded measurements

		# save data options
		self.saveData = True # saves inputs and outputs to a file timeStamp_dataCollectionOL.csv
		self.saveSpatialTemp = True # saves spatial temperature values to a file timeStamp_dataCollectionSpatialTemps.csv
		self.saveSpectra = True # saves entire spectrum at each sampling time to a file timeStamp_dataCollectionSpectra.csv
		self.saveOscMeas = True # saves oscilloscope measurements to a file timeStamp_dataCollectionOscilloscope.csv
		self.saveEmbMeas = True # saves embedded measurements to a file timeStamp_dataCollectionEmbedded.csv

		self.tSampling = 1.0

	def setSamplingTime(tSampling):
		if self.collectOscMeas == True:
			if tSampling > 0.8:
				print('WARNING: sampling time may be greater than measurement collection + input actuation time!!')
		self.tSampling = tSampling
		return

##################################################################################################################
# ARDUINO
##################################################################################################################

def sendInputsArduino(arduino, appliedPower, flow, dutyCycle, arduinoAddress):
	arduino.reset_input_buffer()
	# Send input values to the microcontroller to actuate them
	subprocess.run('echo "p,{:.2f}" > '.format(dutyCycle) + arduinoAddress, shell=True) #firmware v14
	time.sleep(0.5)
	subprocess.run('echo "w,{:.2f}" > '.format(appliedPower) + arduinoAddress, shell=True) #firmware v14
	time.sleep(0.5)
	subprocess.run('echo "q,{:.2f}" > '.format(flow) + arduinoAddress, shell=True)
	time.sleep(0.5)
	outString = "Input values: Power: %.2f, Flow: %.2f, Duty Cycle: %.2f" %(appliedPower,flow,dutyCycle)
	print(outString)

def sendControlledInputsArduino(arduino, appliedPower, flow, arduinoAddress):
	arduino.reset_input_buffer()
	# Send input values to the microcontroller to actuate them
	subprocess.run('echo "w,{:.2f}" > '.format(appliedPower) + arduinoAddress, shell=True) #firmware v14
	time.sleep(0.05)
	subprocess.run('echo "q,{:.2f}" > '.format(flow) + arduinoAddress, shell=True)
	time.sleep(0.05)
	outString = "Input value(s): Power: %.2f, Flow: %.2f" %(appliedPower,flow)
	print(outString)

def getMeasArduino(dev):
	'''
	function to get embedded measurements from the Arduino (microcontroller)

	Inputs:
	dev 	device object for Arduino

	Outputs:
	Is			embedded surface intensity measurement
	U			inputs (applied peak to peak Voltage, frequency, flow rate)
	x_pos		X position
	y_pos		Y position
	dsep		separation distance from jet tip to substrate (Z position)
	T_emb		embedded temperature measurement
	P_emb		embedded power measurement
	Pset		power setpoint
	Dc			duty cycle
	elec		electrical measurements (embedded voltage and current)
	'''
	# set default values for data/initialize data values
	Is = 0
	U = [0,0,0,0]	# inputs (applied Voltage, frequency, flow rate)
	x_pos = 0
	y_pos = 0
	dsep = 0
	T_emb = 0
	elec = [0,0]	# electrical measurements (embedded voltage and current)

	# run the data capture
	run = True
	while run:
		try:
			# dev.reset_input_buffer()
			dev.readline()
			line = dev.readline().decode('ascii')
			if is_line_valid(line):
				# print(line)
				run = False
				# data read from line indexed as programmed on the Arduino
				V = float(line.split(',')[1])	# p2p Voltage
				f = float(line.split(',')[2])	# frequency
				q = float(line.split(',')[3])	# Helium flow rate
				dsep = float(line.split(',')[4])	# Z position
				Dc = float(line.split(',')[5])	# duty cycle
				Is = float(line.split(',')[6])	# embedded intensity
				V_emb = float(line.split(',')[7])	# embedded voltage
				T_emb = float(line.split(',')[8])	# embedded temperature
				I_emb = float(line.split(',')[9])	# embedded current
				x_pos = float(line.split(',')[10])	# X position
				y_pos = float(line.split(',')[11])	# Y position
				# q2 = float(line.split(',')[12])		# Oxygen flow rate
				Pset = float(line.split(',')[13])	# power setpoint
				P_emb = float(line.split(',')[14])	# embedded power
			else:
				print("CRC8 failed. Invalid line!")
			U = [V,f,q]
			elec = [V_emb, I_emb]
		except Exception as e:
			print(e)
			pass
	print(line)
	return np.array([Is,*U,x_pos,y_pos,dsep,T_emb,P_emb,Pset,Dc,*elec])

def getArduinoAddress(os="macos"):
	'''
	function to get Arduino address. The Arduino address changes each time a new
	connection is made using a either a different computer or USB hub. This
	function works for Unix systems, where devices connected to the computer
	have the path footprint: /dev/...

	UPDATED: 2021/03/18, automatically gets device path (no need for user input)

	Inputs:
	None

	Outputs:
	path of the connected device (Arduino)
	'''
	if os == "macos":
		# command to list devices connected to the computer that can be used as call-out devices
		listDevicesCommand = 'ls /dev/cu.usbmodem*'
		print('Getting devices that under the path: /dev/cu.usbmodem* ...')

	elif os == "ubuntu":
		# command to list devices connected to the computer that can be used as call-out devices
		listDevicesCommand = 'ls /dev/ttyACM*'
		print('Getting devices that under the path: /dev/ttyACM* ...')

	else:
		print('OS not currently supported! Manually input the device path.')

	df = subprocess.check_output(listDevicesCommand, shell=True, text=True)
	devices = []
	for i in df.split('\n'):
	    if i:
	        devices.append(i)

	if len(devices)>1:
		print('There are multiple devices with this path format.')
		devIdx = int(input('Please input the index of the device that corresponds to the master Arduino (first index = 0):\n'))
	else:
		print('Only one device found! This will be considered the Arduino device.')
		devIdx = 0

	return devices[devIdx]

def is_line_valid(line):
	'''
	Copied from Dogan's code: Verify that the line read from Arduino is complete
	and correct

	Inputs:
	line 	line read from Arduino

	Outputs:
	boolean value representing the verification of the line
	'''
	l = line.split(',')
	crc = int(l[-1])
	data = ','.join(l[:-1])
	return crc_check(data,crc)

def crc_check(data,crc):
	'''
	Copied from Dogan's code: Check the CRC value to make sure it's consistent
	with data collected

	Inputs:
	data 		line of data collected
	crc 		CRC value

	Outputs:
	boolean value representing the verification of the CRC
	'''
	crc_from_data = crc8("{}\x00".format(data).encode('ascii'))
	# print("crc:{} calculated: {} data: {}".format(crc,crc_from_data,data))
	return crc == crc_from_data


##################################################################################################################
# THERMAL CAMERA
##################################################################################################################
def openThermalCamera():
	ctx = POINTER(uvc_context)()
	dev = POINTER(uvc_device)()
	devh = POINTER(uvc_device_handle)()
	ctrl = uvc_stream_ctrl()

	res = libuvc.uvc_init(byref(ctx), 0)
	if res < 0:
		print("uvc_init error")
		exit(1)

	try:
		res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
		if res < 0:
			print("uvc_find_device error")
			exit(1)

		try:
			res = libuvc.uvc_open(dev, byref(devh))
			if res < 0:
				print("uvc_open error")
				exit(1)

			print("device opened!")

      # print_device_info(devh)
      # print_device_formats(devh)

			frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
			if len(frame_formats) == 0:
				print("device does not support Y16")
				exit(1)

			libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
			frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
			)

			res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
			if res < 0:
				print("uvc_start_streaming failed: {0}".format(res))
				exit(1)

      # try:
      #   data = q.get(True, 500)
      #   data = cv2.resize(data[:,:], (640, 480))
      #   minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
      #   img = raw_to_8bit(data)
      #   Ts_max = display_temperature(img, maxVal, maxLoc, (0, 0, 255))
      #   Ts_min = display_temperature(img, minVal, minLoc, (255, 0, 0))
			# finally:
			# 	pass
      #   libuvc.uvc_stop_streaming(devh)
		finally:
			pass
			# libuvc.uvc_unref_device(dev)
	finally:
		pass
		# libuvc.uvc_exit(ctx)

	return dev, ctx


def getSurfaceTemperature():
	data = q.get(True, 500)
	data = cv2.resize(data[:,:], (640, 480))
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
	img = raw_to_8bit(data)
	Ts_max = display_temperature(img, maxVal, maxLoc, (0, 0, 255))
	Ts_min = display_temperature(img, minVal, minLoc, (255, 0, 0))

	# get offset values of surface temperature (added 2021/03/18)
	# 2 pixels away
	n_offset1 = 2
	Ts2 = get_avg_spatial_temp(n_offset1, data, maxLoc)

	# 12 pixels away
	n_offset2 = 12
	Ts3 = get_avg_spatial_temp(n_offset2, data, maxLoc)

	# TODO: add spatial measurements to return values as desired
	return Ts_max

def get_avg_spatial_temp(n_pix, data, loc):
	'''
	function to get the average temperature about a certain radius of the
	surface temperature. function gets the values from the four cardinal
	directions and returns the average value of those four values. while the
	function accounts for data out of the image bounds, the best practice is to
	make sure the measured area is near the center of the captured image

	Inputs:
	n_pix		number of pixels offset from the surface temperature measurement
	data		raw image data
	loc			location of the surface temperature measurement

	Outputs:
	avg_temp	average value of the temperature from measurements in the four
				cardinal directions
	'''
	# extract the x and y values from the location
	maxX, maxY = loc
	# east
	if maxX+n_pix >= 640:
		maxValE = ktoc(data[maxY, maxX])
	else:
		maxValE = ktoc(data[maxY, maxX+n_pix])
	# west
	if maxX-n_pix < 0:
		maxValW = ktoc(data[maxY, maxX])
	else:
		maxValW = ktoc(data[maxY, maxX-n_pix])
	# south
	if maxY+n_pix >= 480:
		maxValS = ktoc(data[maxY, maxX])
	else:
		maxValS = ktoc(data[maxY+n_pix, maxX])
	# north
	if maxY-n_pix < 0:
		maxValN = ktoc(data[maxY, maxY])
	else:
		maxValN = ktoc(data[maxY-n_pix, maxX])

	avg_temp = (maxValE+maxValW+maxValN+maxValS)/4
	return avg_temp


def closeThermalCamera(dev, ctx):
	libuvc.uvc_unref_device(dev)
	libuvc.uvc_exit(ctx)


##################################################################################################################
# OSCILLOSCOPE
##################################################################################################################
class Oscilloscope():
	def __init__(self):
		pass

	# Method that initializes the oscilloscope
	def initialize(self, retry = 10):
		oscilloscopeStr = ''
		nAttempts = 0
		while (oscilloscopeStr == '') & (nAttempts < retry):
			try:
				instr = usbtmc.Instrument(0x1ab1, 0x04ce)
				instr.open()
				oscilloscopeStr = instr.ask("*IDN?\n")
			except Exception as e:
				nAttempts += 1
				print("{} in oscilloscope check loop".format(e))
				# If initialization fails, close and restart the oscilloscope connection
				instr.close()

		print("Oscilloscope info: {}".format(oscilloscopeStr))
		print("Oscilloscope timeout: {}".format(instr.timeout))

		return instr

	# Method that records the measurements
	def measurement(self, instr):
		# Measurement from channel 1 (voltage)
		instr.write(":MEAS:SOUR CHAN1")
		Vrms = float(instr.ask("MEAS:ITEM? PVRMS"))
		# Vmax=float(instr.ask("MEAS:VMAX?"))
		# Vp2p = float(instr.ask("MEAS:VPP?"))
		# Freq=float(instr.ask("MEAS:FREQ?"))
		# o.Vwave=oscilloscope.ask(':WAV:DATA?')

		# Measurement from channel 2 (current)
		instr.write(":MEAS:SOUR CHAN2")
		Irms = float(instr.ask("MEAS:ITEM? PVRMS"))
		# Imax = float(instr.ask("MEAS:VMAX?"))*1000
		# Ip2p=float(instr.ask("MEAS:VPP?"))*1000
		# o.Iwave=oscilloscope.ask(':WAV:DATA?')

		# Measurement from math channel (V*I)
		instr.write(":MEAS:SOUR MATH")
		Pavg = float(instr.ask("MEAS:VAVG?"))
		# Prms=float(instr.ask("MEAS:ITEM? PVRMS"))
		Prms = Vrms*Irms

		# out = np.array([Vrms, Vmax, Vp2p, Freq, Irms, Imax, Ip2p, Pavg, Prms])
		out = np.array([Vrms, Irms, Pavg, Prms])

		return out

##################################################################################################################
# ASYNCHRONOUS MEASUREMENT
##################################################################################################################

async def async_measure(ard, prevTime, osc_instr, spec, runOpts):
	'''
	function to get measurements from all devices asynchronously to optimize
	time to get measurements

	Inputs:
	ard 		Arduino device reference
	osc_instr	initialized Oscilloscope instance
	spec		Spectrometer device reference
	runOpts 	run options; if data should be saved, then measurements will be
				taken, otherwise the task will return None

	Outputs:
	tasks		completed list of tasks containing data measurements; the first
				task obtains temperature measurements, second task obtains
				spectrometer measurements, third task gets oscilloscope
				measurements, and the fourth (final) task gets embedded
				measurements from the Arduino output
	runTime 	run time to complete all tasks
	'''
	# create list of tasks to complete asynchronously
	tasks = [asyncio.create_task(async_get_temp(runOpts)),
			asyncio.create_task(async_get_spectra(spec, runOpts)),
			asyncio.create_task(async_get_osc(osc_instr, runOpts)),
			asyncio.create_task(async_get_emb(ard, prevTime, runOpts))]

	startTime = time.time()
	await asyncio.wait(tasks)
	# await asyncio.gather(*tasks)
	endTime = time.time()
	runTime = endTime-startTime
	# print time to complete measurements
	print('...completed data collection tasks after {} seconds'.format(runTime))
	return tasks, runTime

async def async_get_temp(runOpts):
	'''
	asynchronous definition of surface temperature measurement. Assumes the
	camera device has already been initialized. Also can include spatial
	temperature measurements. If spatial temperatures are not desired, then the
	spatial measurements output by this function are -300.

	Inputs:
	runOpts 	run options
	**assumes thermal camera device has been successfully opened prior

	Outputs:
	Ts		surface temperature (max temperature from thermal camera) in Celsius
	Ts2		average spatial temperature from 2 pixels away from Ts in Celsius
	Ts3 	average spatial temperature from 12 pixels away from Ts in Celsius
	data	raw data matrix of the image captured
	if data collection is specified otherwise, outputs None
	'''
	if runOpts.collectData:
		# run the data capture
		run = True
		while run:
			# image data is processed in a Queue
			data = q.get(True, 500)
			if data is None:
				print("No data read from thermal camera. Check connection.")
				exit(1)
			# data is resized to the appropriate array size
			data = cv2.resize(data[:,:], (640, 480))
			# get min and max values as well as their respective locations
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
			# convert to Celsius (from Kelvin) with appropriate scaling
			Ts = ktoc(maxVal)

			if runOpts.collectSpatialTemp:
				# get offset values of surface temperature
				# 2 pixels away
				n_offset1 = 2
				Ts2 = get_avg_spatial_temp(n_offset1, data, maxLoc)

				# 12 pixels away
				n_offset2 = 12
				Ts3 = get_avg_spatial_temp(n_offset2, data, maxLoc)
			else:
				Ts2 = -300
				Ts3 = -300

			run = False
		# print('temperature measurement done!')
		return [Ts, Ts2, Ts3, data]
	else:
		return None

async def async_get_spectra(spec, runOpts):
	'''
	asynchronous definition of optical emission spectra data

	Inputs:
	spec 		Spectrometer device
	runOpts 	run options

	Outputs:
	totalIntensity		total intensity measurement
	intensitySpectrum	intensity spectrum
	wavelengths			wavelengths that correspond to the intensity spectrum
	if data collection is specified otherwise, outputs None
	'''
	if runOpts.collectData:
		intensitySpectrum = spec.intensities()
		meanShift = np.mean(intensitySpectrum[-20:-1])
		intensitySpectrum = intensitySpectrum - meanShift
		totalIntensity = sum(intensitySpectrum[20:])

		if runOpts.collectEntireSpectra:
			wavelengths = spec.wavelengths()
		else:
			wavelengths = None
		# print('spectra recorded!')
		return [totalIntensity, intensitySpectrum, wavelengths, meanShift]
	else:
		return None

async def async_get_osc(instr, runOpts):
	'''
	asynchronous definition of oscilloscope measurements

	Inputs:
	obj 		the oscilloscope object as defined in its Class definition
	instr 		initialized oscilloscope object

	Outputs:
	Vrms 		root mean square (RMS) Voltage measurement
	Vp2p		peak to peak voltage measurement
	Irms 		RMS Current measurement
	Imax 		maximum current measurement
	Pavg 		average Power measurement
	Prms 		RMS Power
	if data collection is specified otherwise, outputs None
	'''
	if runOpts.collectOscMeas:
		# Measurement from channel 1 (voltage)
		Vrms = float(instr.ask("MEAS:VRMS? CHAN1"))
		# Vmax=float(instr.ask("MEAS:VMAX?"))
		# Vp2p = float(instr.ask("MEAS:VPP?"))
		# Freq=float(instr.ask("MEAS:FREQ?"))

		# Measurement from channel 2 (current)
		Irms = float(instr.ask("MEAS:VRMS? CHAN2"))
		# Imax = float(instr.ask("MEAS:VMAX?"))*1000
		# Ip2p=float(instr.ask("MEAS:VPP?"))*1000

		# Measurement from math channel (V*I)
		# Pavg = float(instr.ask("MEAS:VAVG? MATH"))
		# Prms=float(instr.ask("MEAS:ITEM? PVRMS"))

		Prms = Vrms*Irms
		# print('oscilloscope measurement done!')
		return np.array([Vrms, Irms, Prms])
	else:
		return None

async def async_get_emb(dev, prevTime, runOpts):
	'''
	asynchronous definition to get embedded measurements from the Arduino
	(microcontroller)

	Inputs:
	dev 		device object for Arduino
	runOpts 	run options

	Outputs:
	Outputs:
	Is			embedded surface intensity measurement
	U			inputs (applied peak to peak Voltage, frequency, flow rate)
	x_pos		X position
	y_pos		Y position
	dsep		separation distance from jet tip to substrate (Z position)
	T_emb		embedded temperature measurement
	P_emb		embedded power measurement
	Pset		power setpoint
	Dc			duty cycle
	elec		electrical measurements (embedded voltage and current)
	if data collection is specified otherwise, outputs None
	'''
	if runOpts.collectEmbedded:
		# set default values for data/initialize data values
		Is = 0
		U = [0,0,0]	# inputs (applied Voltage, frequency, flow rate)
		x_pos = 0
		y_pos = 0
		dsep = 0
		T_emb = 0
		elec = [0,0]	# electrical measurements (embedded voltage and current)
		P_emb = 0
		Pset = 0
		Dc = 0

		# run the data capture
		run = True
		while run:
			try:
				# dev.reset_input_buffer()
				# dev.readline()
				line = dev.readline().decode('ascii')
				if is_line_valid(line):
					# print(line)
					data = line.split(',')
					timeStamp = float(data[0])
					if True:
					# if (timeStamp-prevTime)/1e3 >= runOpts.tSampling-0.025:
						run = False
						# data read from line indexed as programmed on the Arduino
						V = float(data[1])	# p2p Voltage
						f = float(data[2])	# frequency
						q = float(data[3])	# Helium flow rate
						dsep = float(data[4])	# Z position
						Dc = float(data[5])	# duty cycle
						Is = float(data[6])	# embedded intensity
						V_emb = float(data[7])	# embedded voltage
						T_emb = float(data[8])	# embedded temperature
						I_emb = float(data[9])	# embedded current
						x_pos = float(data[10])	# X position
						y_pos = float(data[11])	# Y position
						# q2 = float(data[12])		# Oxygen flow rate
						Pset = float(data[13])	# power setpoint
						P_emb = float(data[14])	# embedded power

						U = [V,f,q]
						elec = [V_emb, I_emb]
				else:
					print("CRC8 failed. Invalid line!")
			except Exception as e:
				print(e)
				pass
		print(line)
		# print('embedded measurement done!')
		return np.array([timeStamp, Is,*U,x_pos,y_pos,dsep,T_emb,P_emb,Pset,Dc,*elec])
	else:
		return None
