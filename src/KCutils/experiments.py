# experiment functions
#
# This file defines an Experiment class to be used for real time experiments on
# the atmospheric pressure plasma jet (APPJ) of model predictive controllers
# (MPCs) generated via the controller subclasses defined in controllers.py
#
# Requirements:
# * Python 3
#
# Copyright (c) 2022 Mesbah Lab. All Rights Reserved.
# Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

## import 3rd party packages
import sys
sys.dont_write_bytecode = True
import numpy as np
import time
from datetime import datetime
import os

## import user functions
import KCutils.APPJPythonFunctions as appj

def ctok(T):
    """
    Function to convert from Celsius to Kelvin.
    """
    return T+273.15

class Experiment():
    """
    The Experiment class is used to create a wrapper for real-time experiments
    using the APPJ.
    """

    def __init__(self, Nsim, saveDir=os.getcwd(), name=None):
        """
        Instantiation of the Experiment class requires the input arguments
        Nsim, which denotes the length of the experimental run; saveDir
        (optional), which is a path to a particular save location; and name
        (optional) which is an additional identifier of the data from this
        class.
        """
        super(Experiment, self).__init__()
        self.Nsim = Nsim
        self.prob_info = None
        self.rand_seed = None

        self.saveDir = saveDir
        if not os.path.exists(saveDir+"Backup/"):
            self.backupSaveDir = saveDir+"Backup/"
            os.makedirs(saveDir+"Backup", exist_ok=True)
        print('\n\nBackup data will be saved in the following directory:')
        print(self.backupSaveDir)
        self.count = 0
        self.name = name
        if self.name is None:
            self.exp_name = 'Experiment_'+str(self.count)
        else:
            self.exp_name = self.name+'_Experiment_'+str(self.count)

        self.ol_count = 0

    def load_prob_info(self, prob_info):
        """
        This method loads the relevant problem information for experiment and
        assigns them as attributes of the class from the prob_info dict used by
        other classes included in this package.
        """
        # extract and save relevant problem information
        self.prob_info = prob_info

        # system sizes
        self.nu = prob_info['nu']
        self.nx = prob_info['nx']
        self.ny = prob_info['ny']
        self.nyc = prob_info['nyc']

        self.xss = prob_info['xss']
        self.uss = prob_info['uss']
        self.u_min = prob_info['u_min']
        self.u_max = prob_info['u_max']

        self.Np = prob_info['Np'] # prediction horizon
        self.x0 = prob_info['x0'] # initial state/point
        # self.y0 = prob_info['y0'] # initial outputs/measurements
        # self.u0 = prob_info['u0'] # startup inputs
        self.myref = prob_info['myref'] # reference function for the controlled output
        self.ts = prob_info['ts'] # simulation sampling time

    def run_closed_loop_mpc(self, c, ioloop, model=None, observer=None,
                            runOpts=appj.RunOpts(), devices=None, prevTime=0.0,
                            CEM=True, hw_flag=True, dnn_flag=False,
                            I_NORM=1e-4):
        """
        This method runs a closed-loop experiment of the APPJ using information
        derived from loading problem information and using an explicit MPC
        (defined by solving an optimal control problem (OCP) at each step). The
        problem information must be loaded before a closed-loop simulation can
        occur. The argument c is a MPC controller object created using one of
        the classes defined in the controller.py.
        """
        # check to ensure problem data has been loaded
        if self.prob_info is None:
            print('Problem data not loaded. Please load the appropriate problem data by running the load_prob_info method.')
            raise

        # get devices
        if devices is None:
            print('Device information not given! Please provide device info.')
            raise
        else:
            # serial device representation of Arduino
            key = 'arduinoPI'
            if key in devices:
                arduinoPI = devices[key]
            else:
                arduinoPI = None
                print(f'WARNING: {key} not in devices dict! Code will error...')
            # Arduino address
            key = 'arduinoAddress'
            if key in devices:
                arduinoAddress = devices[key]
            else:
                arduinoAddress = None
                print(f'WARNING: {key} not in devices dict! Code will error...')
            # Spectrometer
            key = 'spec'
            if key in devices:
                spec = devices[key]
            else:
                spec = None
                print(f'WARNING: {key} not in devices dict! Code will error...')
            # Oscilloscope
            key = 'instr'
            if key in devices:
                instr = devices[key]
            else:
                instr = None
                print(f'WARNING: {key} not in devices dict! Code will error...')

        ## get data sizes
        tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
        if runOpts.collectData:
            thermalCamOut = tasks[0].result()
            Ts0 = thermalCamOut[0]
            specOut = tasks[1].result()
            I0 = specOut[0]*I_NORM
            oscOut = tasks[2].result()
            arduinoOut = tasks[3].result()
            outString = "Measured Outputs: Temperature: %.2f, Intensity: %.2f" % (Ts0, I0)
            print(outString)
        else:
            Ts0 = 37.0
            I0 = 100.0

        ## get controller type:
        mpc = False
        if hw_flag or dnn_flag:
            pass
        else:
            mpc = True
            Wset = self.prob_info['Wset']

        ## Instantiate container variables for storing experimental data
        if runOpts.collectData:
            if runOpts.saveData:
                Tsave = np.empty((self.Nsim,))
                Isave = np.empty((self.Nsim,))
                Psave = np.empty((self.Nsim,))
                qSave = np.empty((self.Nsim,))
                CEMsave = np.empty((self.Nsim))
                badTimes = []
            if runOpts.saveSpatialTemp:
                Ts2save = np.empty((self.Nsim,))
                Ts3save = np.empty((self.Nsim,))
            if runOpts.saveSpectra:
                if specOut is not None:
                    waveSave = np.empty((self.Nsim,len(specOut[2])))
                    specSave = np.empty_like(waveSave)
                    meanShiftSave = np.empty((self.Nsim,))
                else:
                    print('Intensity Data not collected! Entire spectrum will not be saved.')
                    runOpts.saveSpectra = False
            if runOpts.saveOscMeas:
                if oscOut is not None:
                    oscSave = np.empty((self.Nsim,len(oscOut)))
                else:
                    print('Oscilloscope data not collected! Nothing to save.')
                    runOpts.saveOscMeas = False
            if runOpts.saveEmbMeas:
                if arduinoOut is not None:
                    ArdSave = np.empty((self.Nsim,len(arduinoOut)))
                else:
                    print('Arduino Data not collected! Nothing to save.')
                    runOpts.saveEmbMeas = False
        # additional containers for system operation (controller, observer)
        Xhat = np.zeros((self.nx,self.Nsim+1))    # state estimation
        ctime = np.zeros(self.Nsim)   # computation time
        Yrefsim = np.zeros((self.nyc,self.Nsim))  # output reference/target (as sent to controller)
        CEMsim = np.zeros((self.nyc,self.Nsim+1)) # CEM accumulation
        if mpc:
            Jsim = np.zeros(self.Nsim)    # cost/optimal objective value (controller)
            Ypred = np.zeros((self.ny,self.Nsim,self.Np))
            Feasibility = np.zeros(self.Nsim) # feasibility of OCP

        # set initial states and reset controller for consistency
        Xhat[:,0] = np.ravel([Ts0,I0]-self.xss)
        CEMsim[:,0] = np.zeros((self.nyc,))
        count = 0

        # loop over simulation time
        if mpc:
            c.reset_initial_guesses()
        if CEM:
            CEM_stop_time = self.Nsim

        for k in range(self.Nsim):
            startTime = time.time()
            iterString = f'\nIteration {k} out of {self.Nsim}'
            print(iterString)

            ## Get measurements
            tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
            # Temperature
            if runOpts.collectData:
                thermalCamMeasure = tasks[0].result()
                if thermalCamMeasure is not None:
                    Ts = thermalCamMeasure[0]
                    Ts2 = thermalCamMeasure[1]
                    Ts3 = thermalCamMeasure[2]
                else:
                    print('Temperature data not collected! Thermal Camera measurements will be set to -300.')
                    Ts = -300
                    Ts2 = -300
                    Ts3 = -300
                # Intensity
                specOut = tasks[1].result()
                if specOut is not None:
                    totalIntensity = specOut[0]*I_NORM
                    intensitySpectrum = specOut[1]
                    wavelengths = specOut[2]
                    meanShift = specOut[3]
                else:
                    print('Intensity data not collected! Spectrometer outputs will be set to -1.')
                    totalIntensity = -1
                    intensitySpectrum = -1
                    wavelengths = -1
                    meanShift = -1

                outString = "Measured Outputs: Temperature: %.2f, Intensity: %.2f" % (Ts, totalIntensity)
                print(outString)

                ## Save measurements to containers
                if runOpts.saveData:
                    Tsave[k] = Ts
                    Isave[k] = totalIntensity
                if runOpts.saveSpatialTemp:
                    Ts2save[k] = Ts2
                    Ts3save[k] = Ts3
                # Intensity spectra
                if runOpts.saveSpectra:
                    waveSave[k,:] = np.ravel(wavelengths)
                    specSave[k,:] = np.ravel(intensitySpectrum)
                    meanShiftSave[k] = meanShift
                # Oscilloscope
                if runOpts.saveOscMeas:
                    oscOut = tasks[2].result()
                    oscSave[k,:] = np.ravel(oscOut)
                # Embedded Measurements from the Arduino (note: several embedded measurements are not fully functional as of 2020/12)
                arduinoOut = tasks[3].result()
                prevTime = arduinoOut[0]
                if runOpts.saveEmbMeas:
                    ArdSave[k,:] = np.ravel(arduinoOut)
            else:
                Ts = 37.0
                totalIntensity = 100.0
            # measurements are collected after a control input has been applied
            Yrefsim[:,k] = self.myref(k)
            if k>0:
                Xhat[:,k] = np.ravel(np.asarray([[Ts,totalIntensity]])-self.xss)
                if CEM:
                    CEMsim[:,k] = CEMsim[:,k-1] + np.ravel(self.prob_info['CEMadd'](Ts).full())
                    if CEMsim[:,k]>Yrefsim[:,k]:
                        if CEMsim[:,k-1]<Yrefsim[:,k-1]:
                            CEM_stop_time = k
                        count+=1
                        if count > 3:
                            break

            ## Compute control input
            ctrl_stime = time.time()
            if mpc:
                print(Xhat[:,k])
                print(CEMsim[:,k])
                c.set_parameters([Xhat[:,k], Yrefsim[:,k], CEMsim[:,k], Wset])
                res, feas = c.solve_mpc(warm_start=self.prob_info['warm_start'])
                print(res['U'])
                print(feas)
                Uopt = np.asarray(res['U'])
                Jopt = res['J']
            else:
                if CEM:
                    x_in = np.concatenate((Xhat[:,k], CEMsim[:,k]))
                else:
                    x_in = np.concatenate((Xhat[:,k], Yrefsim[:,k]))
                Uopt = np.ravel((c.netca(x_in)).full())
                print(Uopt)

            Uopt[0] = np.clip(Uopt[0], self.u_min[0], self.u_max[0])
            Uopt[1] = np.clip(Uopt[1], self.u_min[1], self.u_max[1])
            Uopt = np.ravel(Uopt.T+self.uss)
            print(Uopt)
            ctrl_etime = time.time()
            ctime[k] = ctrl_etime - ctrl_stime

            powerIn = float(Uopt[0])
            flowIn = float(Uopt[1])
            ## Send optimal inputs to APPJ
            appj.sendControlledInputsArduino(arduinoPI, powerIn, flowIn, arduinoAddress)

            # save inputs
            if runOpts.saveData:
                Psave[k] = np.ravel(Uopt[0])
                qSave[k] = np.ravel(Uopt[1])
            if mpc:
                Jsim[k] = Jopt
                # Ypred[:,:,k] = res['Y']
                # Wpred[k] = res['wPred']

            # Pause for the duration of the sampling time to allow the system to evolve
            endTime = time.time()
            runTime = endTime-startTime
            print('Total Runtime was:', runTime)
            pauseTime = self.ts - runTime
            if pauseTime>0:
                print("Pausing for {} seconds...".format(pauseTime))
                time.sleep(pauseTime)
            else:
                print('WARNING: Measurement Time was greater than Sampling Time! Data may be inaccurate.')
                if runOpts.saveData:
                    badTimes += [k]

        # shut off plasma
        appj.sendInputsArduino(arduinoPI, 0.0, 0.0, 100.0, arduinoAddress)

        # create dictionary of experimental data
        exp_data = {}
        exp_data['Tsave'] = Tsave
        exp_data['Isave'] = Isave
        exp_data['Psave'] = Psave
        exp_data['qSave'] = qSave
        exp_data['Yrefsim'] = Yrefsim
        exp_data['ctime'] = ctime
        exp_data['Xhat'] = Xhat
        exp_data['badTimes'] = badTimes
        if mpc:
            exp_data['Jsim'] = Jsim
            exp_data['Feasibility'] = Feasibility
            exp_data['Ypred'] = Ypred
        if CEM:
            exp_data['CEMsim'] = CEMsim
            exp_data['CEM_stop_time'] = CEM_stop_time
        if runOpts.collectSpatialTemp:
            exp_data['Ts2save'] = Ts2save
            exp_data['Ts3save'] = Ts3save
        if runOpts.collectEntireSpectra:
            exp_data['waveSave'] = waveSave
            exp_data['specSave'] = specSave
            exp_data['meanShiftSave'] = meanShiftSave
        if runOpts.collectOscMeas:
            exp_data['oscSave'] = oscSave
        if runOpts.collectEmbedded:
            exp_data['ArdSave'] = ArdSave

        # save experimental data to have a backup copy
        self.exp_data = exp_data
        np.save(self.backupSaveDir+self.exp_name+'.npy', exp_data)

        # save csv version of experimental data
        exp_saveDir = self.saveDir+self.exp_name+'/'
        if not os.path.exists(exp_saveDir):
            os.makedirs(exp_saveDir, exist_ok=True)
        exp_data_saver(exp_data, exp_saveDir, self.exp_name, runOpts)

        # increment and update class attributes to prepare for additional experiments
        self.count += 1
        if self.name is None:
            self.exp_name = 'Experiment_'+str(self.count)
        else:
            self.exp_name = self.name+'_Experiment_'+str(self.count)

        return exp_data

    def run_open_loop(self, ioloop, power_seq=None, flow_seq=None, runOpts=appj.RunOpts(), devices=None, prevTime=0.0):
        """
        This method runs a open-loop experiment of the APPJ using provided
        sequences of inputs.
        """
        # check for provided sequence of inputs
        if power_seq is None and flow_seq is None:
            print('Sequence of inputs not given! Please provide inputs.')
            quit()
        elif power_seq is None:
            P0 = float(input('Please enter a value for the power.\n'))
            flow_seq = np.asarray(flow_seq)
            power_seq = P0*np.ones_like(flow_seq)

        elif flow_seq is None:
            q0 = float(input('Please enter a value for the flow rate.\n'))
            power_seq = np.asarray(power_seq)
            flow_seq = q0*np.ones_like(power_seq)

        nP = len(power_seq)
        nq = len(flow_seq)

        if nP > nq:
            print('Sequence of POWER inputs longer than sequence of FLOW inputs. Using the shorter sequence...')
            Niter = nq
        elif nq > nP:
            print('Sequence of FLOW inputs longer than sequence of POWER inputs. Using the shorter sequence...')
            Niter = nP
        else:
            Niter = nP

        # unpack devices
        if devices is None:
            print('Device information not given! Please provide device info.')
            raise
        else:
            # serial device representation of Arduino
            key = 'arduinoPI'
            if key in devices:
                arduinoPI = devices[key]
            else:
                arduinoPI = None
                print(f'WARNING: {key} not in devices dict! Code will error...')
            # Arduino address
            key = 'arduinoAddress'
            if key in devices:
                arduinoAddress = devices[key]
            else:
                arduinoAddress = None
                print(f'WARNING: {key} not in devices dict! Code will error...')
            # Spectrometer
            key = 'spec'
            if key in devices:
                spec = devices[key]
            else:
                spec = None
                print(f'WARNING: {key} not in devices dict! Code will error...')
            # Oscilloscope
            key = 'instr'
            if key in devices:
                instr = devices[key]
            else:
                instr = None
                print(f'WARNING: {key} not in devices dict! Code will error...')

        # initial measurement to get data sizes
        tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
        thermalCamOut = tasks[0].result()
        Ts0 = thermalCamOut[0]
        specOut = tasks[1].result()
        I0 = specOut[0]
        oscOut = tasks[2].result()
        arduinoOut = tasks[3].result()

        ## Instantiate container variables for storing experimental data
        if runOpts.saveData:
            Tsave = np.empty((Niter,))
            Isave = np.empty((Niter,))
            badTimes = []
        if runOpts.saveSpatialTemp:
            Ts2save = np.empty((Niter,))
            Ts3save = np.empty((Niter,))
        if runOpts.saveSpectra:
            if specOut is not None:
                waveSave = np.empty((Niter,len(specOut[2])))
                specSave = np.empty_like(waveSave)
                meanShiftSave = np.empty((Niter,))
            else:
                print('Intensity Data not collected! Entire spectrum will not be saved.')
                runOpts.saveSpectra = False
        if runOpts.saveOscMeas:
            if oscOut is not None:
                oscSave = np.empty((Niter,len(oscOut)))
            else:
                print('Oscilloscope data not collected! Nothing to save.')
                runOpts.saveOscMeas = False
        if runOpts.saveEmbMeas:
            if arduinoOut is not None:
                ArdSave = np.empty((Niter,len(arduinoOut)))
            else:
                print('Arduino Data not collected! Nothing to save.')
                runOpts.saveEmbMeas = False


        for i in range(Niter):
            startTime = time.time()
            print(f'\nIteration {i} out of {Niter}')

            # asynchronous measurement
            tasks, _ = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))

            # Temperature
            thermalCamMeasure = tasks[0].result()
            if thermalCamMeasure is not None:
                Ts = thermalCamMeasure[0]
                Ts2 = thermalCamMeasure[1]
                Ts3 = thermalCamMeasure[2]
            else:
                print('Temperature data not collected! Thermal Camera measurements will be set to -300.')
                Ts = -300
                Ts2 = -300
                Ts3 = -300

            # Total intensity
            specOut = tasks[1].result()
            if specOut is not None:
                totalIntensity = specOut[0]
                intensitySpectrum = specOut[1]
                wavelengths = specOut[2]
                meanShift = specOut[3]
            else:
                print('Intensity data not collected! Spectrometer outputs will be set to -1.')
                totalIntensity = -1
                intensitySpectrum = -1
                wavelengths = -1
                meanShift = -1

            # Save measurements <--- takes on the order of 1-2e-5 seconds
            if runOpts.saveData:
                Tsave[i] = Ts
                Isave[i] = totalIntensity
            if runOpts.saveSpatialTemp:
                Ts2save[i] = Ts2
                Ts3save[i] = Ts3
            # Intensity spectra (row 1: wavelengths; row 2: intensities; row 3: mean value used to shift spectra)
            if runOpts.saveSpectra:
                waveSave[i,:] = np.ravel(wavelengths)
                specSave[i,:] = np.ravel(intensitySpectrum)
                meanShiftSave[i] = meanShift
            # Oscilloscope
            if runOpts.saveOscMeas:
                oscOut = tasks[2].result()
                oscSave[i,:] = np.ravel(oscOut)
            # Embedded Measurements from the Arduino
            arduinoOut = tasks[3].result()
            prevTime = arduinoOut[0]
            if runOpts.saveEmbMeas:
                ArdSave[i,:] = np.ravel(arduinoOut)

            print(f'Measured Outputs: Temperature: {Ts:.2f}, Intensity: {totalIntensity:.2f}\n')

            # Send inputs <--- takes at least 0.15 seconds (due to programmed pauses)
            # appj.sendInputsArduino(arduinoPI, power_seq[i], flow_seq[i], dutyCycle, arduinoAddress)
            appj.sendControlledInputsArduino(arduinoPI, float(power_seq[i]), float(flow_seq[i]), arduinoAddress)

            # Pause for the duration of the sampling time to allow the system to evolve
            endTime = time.time()
            runTime = endTime-startTime
            print('Total Runtime was:', runTime)
            pauseTime = runOpts.tSampling - runTime
            if pauseTime>0:
                print(f'Pausing for {pauseTime} seconds...')
                time.sleep(pauseTime)
            else:
                print('WARNING: Measurement Time was greater than Sampling Time! Data may be inaccurate.')
                if runOpts.saveData:
                    badTimes += [i]

        # shut off APPJ
        appj.sendInputsArduino(arduinoPI, 0.0, 0.0, 100.0, arduinoAddress)

        # create dictionary of experimental data
        exp_data = {}
        exp_data['Tsave'] = Tsave
        exp_data['Isave'] = Isave
        exp_data['Psave'] = power_seq
        exp_data['qSave'] = flow_seq
        exp_data['badTimes'] = badTimes
        if runOpts.collectSpatialTemp:
            exp_data['Ts2save'] = Ts2save
            exp_data['Ts3save'] = Ts3save
        if runOpts.collectEntireSpectra:
            exp_data['waveSave'] = waveSave
            exp_data['specSave'] = specSave
            exp_data['meanShiftSave'] = meanShiftSave
        if runOpts.collectOscMeas:
            exp_data['oscSave'] = oscSave
        if runOpts.collectEmbedded:
            exp_data['ArdSave'] = ArdSave

        # save experimental data to have a backup copy
        self.exp_data = exp_data
        np.save(self.backupSaveDir+'OL_data_'+str(self.ol_count)+'.npy', exp_data)

        # save csv version of experimental data
        exp_saveDir = self.saveDir
        if not os.path.exists(exp_saveDir):
            os.makedirs(exp_saveDir, exist_ok=True)
        exp_data_saver(exp_data, exp_saveDir, 'OL_data_'+str(self.ol_count), runOpts)

        self.ol_count += 1
        return exp_data


def exp_data_saver(exp_data, saveDir, exp_name, runOpts):
    """
    This function saves experimental data generated using the Experiment class.
    This function is different from the automatic saving performed by the
    Experiment class when running an individual experiment. This function will
    save most data to csv files to make data easily interpretable without
    having to write a Python script to read the data.

    exp_data is the dictionary of experimental data obtained by running an
            experiment via the the Experiments class
    saveDir is the path to the save location
    timeStamp is the time stamp identifier of the series of experiments
    runOpts is a class that defines the run options used during the experiment
    """
    if runOpts.saveData:
        # extract data
        Tsave = exp_data['Tsave']
        Isave = exp_data['Isave']
        Psave = exp_data['Psave']
        qSave = exp_data['qSave']
        badTimes = exp_data['badTimes']

        dataHeader = "Ts (degC),I (a.u.),P (W),q (slm)"
        # Concetenate inputs and outputs into one numpy array to save it as a csv
        saveArray = np.hstack((Tsave.reshape(-1,1), Isave.reshape(-1,1), Psave.reshape(-1,1), qSave.reshape(-1,1)))
        np.savetxt( saveDir+exp_name+"_inputOutputData.csv", saveArray, delimiter=",", header=dataHeader, comments='')
        if badTimes:
            np.savetxt( saveDir+exp_name+"_badMeasurementTimes.csv", badTimes, delimiter=',')

    if runOpts.saveSpatialTemp:
        # extract data
        Tsave = exp_data['Tsave']
        Ts2save = exp_data['Ts2save']
        Ts3save = exp_data['Ts3save']

        dataHeader = "Ts (degC),Ts2 (degC),Ts3 (degC)"
        saveArray = np.hstack((Tsave.reshape(-1,1), Ts2save.reshape(-1,1), Ts3save.reshape(-1,1)))
        np.savetxt( saveDir+exp_name+"_dataCollectionSpatialTemps.csv", saveArray, delimiter=",", header=dataHeader, comments='')

    if runOpts.saveSpectra:
        # extract data
        waveSave = exp_data['waveSave']
        specSave = exp_data['specSave']
        meanShiftSave = exp_data['meanShiftSave']

        print("Entire spectra will be saved in a compressed .npz file with the following array variable names:\n"
              +"'wavelengths' for the range of wavelength values\n"
              +"'intensities' for the full intensity spectra corresponding to the wavelength range\n"
              +"'meanShifts' for the mean value used to shift the spectra.\n"
              +"Please use a Python script and numpy.load(file_name) to load this data.")
        np.savez_compressed( saveDir+exp_name+"_dataCollectionSpectra", wavelengths=waveSave, intensities=specSave, meanShifts=meanShiftSave)

    if runOpts.saveOscMeas:
        # extract data
        oscSave = exp_data['oscSave']

        dataHeader = "Vrms (V),Irms (A),Prms (W)"
        np.savetxt( saveDir+exp_name+"_dataCollectionOscilloscope.csv", oscSave, delimiter=",", header=dataHeader, comments='')

    if runOpts.saveEmbMeas:
        # extract data
        ArdSave = exp_data['ArdSave']

        dataHeader = "t_emb (ms),Isemb (a.u.),Vp2p (V),f (kHz),q (slm),x_pos (mm),y_pos (mm),dsep (mm),T_emb (K),P_emb (W),Pset (W),duty (%),V_emb (kV),I_emb (mA)"
        np.savetxt( saveDir+exp_name+"_dataCollectionEmbedded.csv", ArdSave, delimiter=",", header=dataHeader, comments='')

    print('\n\nData saved in the following directory:')
    print(saveDir)
    return

def process_experimental_data(exp_data, prob_info):
    pass
