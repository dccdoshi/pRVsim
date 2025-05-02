import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pylab as plt
from stellar import StellarSpectrum
from observations import Observations
from process import Process
from template import Template
from convolution import *
from interpolater import *
from vfinder import Velocity_Finder
from telluric_model import *
import json, argparse, sys
import itertools
from tqdm import tqdm
from multiprocessing import Pool


def load_param(config):
    # Generate combinations of hyperparameters
    hyperparam_combinations = []
    for section, params in config.items():
        param_values = []
        # if section!="likelihood":
        for param, value in params.items():
            if isinstance(value, list):
                param_values.append(value)
            else:
                param_values.append([value])  # Convert single value to list
        hyperparam_combinations.append(list(itertools.product(*param_values)))

            
    # Hold the hyperparams for each individual experiment
    params_experiments = []

    # Iterate through combinations
    for combination in itertools.product(*hyperparam_combinations):
        experiment_params = {}
        for i, section in enumerate(config.keys()):
            for j, param in enumerate(config[section].keys()):
                if param=='obs':
                    experiment_params[param] = combination[i][0]
                else:
                    experiment_params[param] = combination[i][j]
        params_experiments.append(experiment_params)

    return params_experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Template')

    parser.add_argument('--param', type=str, help="file name for json attributes")
    parser.add_argument('--output', type=str, help="file name for df output")

    args = parser.parse_args()

    param_file = args.param
    output_file = args.output

    # Load hyperparameters from JSON file
    with open(param_file) as f:
        config = json.load(f)
    params = load_param(config)

    def loop_function(exp_param):
        dfs = []
        evaluation_points = np.arange(0,200,1)[100:]

        # exp_param = params[i]

        SNR = exp_param['SNR'] # SNR of individual observations
        K_amp = exp_param['Kamp'] # K amplitude of single planetary signal

        ''' Parameters to determine Native Resolution of PHOENIX spectra '''
        R_start = exp_param['R_start'] # Original "broadening" factor of PHOENIX spectra
        oversample = exp_param['oversample'] # Oversampling factor of PHOENIX spectra
        # The native resolution is then R_start*oversample

        ''' Parameters to determine Instrument Resolution '''
        inst_res = exp_param['broadening'] # Broadening factor added to observations by instrument
        use_inst_wgrid = exp_param['use_inst'] # Are you going to downgrade spectra to an instrument resolution
        new_res = exp_param['inst_resolution']

        ##########
        ##########
        ### Stellar Spectrum ##########
        ##########
        ##########

        # Define the spectrum
        stellar = StellarSpectrum(starting_resolution=R_start,oversample=oversample,order=exp_param['order'])#,order=20,T=3000,logg=5,metal=0)

        # We are going to immediately change this flux into a specific SNR flux
        # This is valid because we do not bin into instrument wavelegth grid, we will interpolate to the instrument wavelength grid
        stellar.incorporate_SNR(SNR,use_inst_wgrid,new_res)


        ##########
        ##########
        ### RV Curve ##########
        ##########
        ##########

        # Define the observations object
        observations = Observations(native_wavegrid=stellar.phoenix_wgrid,instrument_wavegrid=stellar.instrument_wgrid,num=200)

        # Define the dates
        start_date = '2020-01-01'
        end_date = '2021-01-03'
        dates = observations.dates(start_date,end_date,seed=exp_param['date_seed'])

        # Define the curve
        RV, planet = observations.RV_signal(planet_amp=K_amp)

        #########
        #########
        ## Create Observations ##########
        #########
        #########

        # Doppler shift them
        shifted = observations.doppler_shift(stellar.stellar)

        # Contaminate with tellurics (turned off for now)
        # contaminated = observations.telluric_contaminate(real_tellurics)

        # Put into Instrument Resolution 
        inst_wgrid, inst_spec, convolved = observations.instrument_captures(inst_res,use_inst_wgrid,new_res)

        # Create noisy observations
        noisy, sig = observations.poisson(seed=exp_param['seeds'])
        

        ##########
        ##########
        ### Process Data ##########
        ##########
        ##########

        processor = Process(observations.instrument_wgrid,observations.instrument_spectra)


        ##########
        ##########
        ### Create Template ##########
        ##########
        ##########

        # We will only create the template out of T observations 
        T = exp_param['Ntemp']
        if T==0:
            template = stellar.stellar.copy()
        else:
            # The template will be placed in an upsampled space, this will just be kept as the native (original)
            # grid of the observations to make it easy for comparison of the interpolation issue
            template_maker = Template(observations.berv[:100][::int(100/T)], upsampled_wgrid = observations.native_wgrid, inst_wgrid = observations.instrument_wgrid)

            # Shift the observations back by their berv
            shiftedflux, upsampledflux = template_maker.shift_back_by_bervs(processor.spectra[:100][::int(100/T)])

            template = template_maker.create_template()
    



        ##########
        ##########
        ### RV_Finder ##########
        ##########
        ##########
        dvs=1
        
        if T==0:
            is_base = True
        else:
            is_base = False



        for o in evaluation_points:

            # First we will just shift the template by berv
            total_shift = observations.berv[o].copy()
            iterated_RVs = np.zeros(5)

            # Find the planet RV iteratively
            for j in range(5):
                # Define the finder object
                vf = Velocity_Finder(template,observations.native_wgrid,observations.instrument_wgrid,total_shift,is_base,inst_res=inst_res,T_epochs=T)

                # Determine the RV
                RV, uncRV = vf.find_dv(processor.spectra[o],dvs,sig[o])

                # Next time we will shift the template by the berv plus this additional RV
                total_shift+=RV
                iterated_RVs[j] = RV.item()
                if RV<=0.001:
                    break

            vf = Velocity_Finder(template,observations.native_wgrid,observations.instrument_wgrid,observations.berv[o],is_base,inst_res=inst_res,T_epochs=T)  
                    
            # Determine the uncertainty after you found the RV
            uncRV = vf.find_unc(processor.spectra[o],sig[o],np.sum(iterated_RVs),dvs)
            final_RV = np.sum(iterated_RVs)
            true_planet = observations.planet[o]
            
            std_zscore = vf.get_zscore(processor.spectra[o],final_RV,sig[o])


            # ### Save in Dataframe #### 
            # # Create DataFrame from dictionary
            experiment = pd.DataFrame([exp_param])
            experiment['True_Planet_RV'] = true_planet

            # Add additional column from the array
            experiment['Retrieved'] = final_RV
            experiment['UncRV'] = uncRV
            experiment['End_Zscore'] = std_zscore
            # experiment['BouchyRV'] = dvc


            dfs.append(experiment)
            print("Experiment Stats", true_planet,SNR,new_res,exp_param['seeds'],flush=True)

        return dfs


    with Pool(16) as p:
        ### Run Through Each Parameter Configuration ###
        # pbar = tqdm(total=20*len(params), desc="Processing")
        result = tqdm(p.imap(loop_function, params), total=len(params))
        # result = p.map(loop_function,params)
        dfs_new = []
        for df in result:
            # for d in df:
            dfs_new += df

        Length = len(dfs_new)

        bs = Length//5
        for i in range(0, Length, bs):
            experiments_df = pd.concat(dfs_new[i:i+bs],axis=0)
            experiments_df.to_pickle(output_file[:-3]+str(i//bs)+output_file[-3:])


    print("Finished experiments, file saved")