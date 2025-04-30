import numpy as np
import matplotlib.pyplot as plt
from hapi import *
import pandas as pd
#from metpy.calc import mixing_ratio_from_relative_humidity
#from metpy.units import units
import os

def retrieve_weather(filename):
    '''
    This function retrieves the weather information given by Etienne 
    to develop ~realistic weather conditions. 
    
    NEED TO CONFIRM INSTRUMENT ORIGIN AND UNITS

    INPUTS:
    filename

    OUTPUTS:
    weather : an array as [humidity (vmr),pressure (atm), temperature (K)]
    '''

    # We can generate some weather values for our tellurics using weather data from Etienne (confirm instrument origin) 
    data = np.genfromtxt(filename,skip_header=True)

    # Data is [relative humidity (%), pressure (torr), temperature (Celsius)]
    weather = data[:,1:4]
    weather[:,1] /= 760 # torr to atm conversion 
    weather[:,2] += 273 # celsius to K conversion 


    # Convert relative humidity to mxixing ratio 
    p = weather[:,1] * units.atm
    T = weather[:,2] * units.degK

    mr = mixing_ratio_from_relative_humidity(p, T, weather[:,0]/100)
    weather[:,0] = mr
    return weather

class Telluric_Model():

    def __init__(self,T_real,p_real,water_abundance_real,filepath) -> None:
        '''
        T_real = This is the real temperature you want your atmosphere to be (units K)
        p_real = This is the real pressure you want your atmosphere to be (units atm)
        water_abundance_real = This is the real humidity level (units mol fraction) (should be between 0.00-0.04)
        filepath: where your HITRAN data is stored
        '''
        self.T_real = T_real
        self.p_real = p_real

        ## Abundances / Composition this is provided in mol fractions
        # The composition is taken from: https://www.noaa.gov/jetstream/atmosphere 
        self.comp = {'N2':[0.7804],
                    'O2':[0.20846],
                    'CO2':[0.00035*0.984204,0.00035*0.011057],
                    'O3':[0.07/1e6],
                    'CO':[0.1/1e6 * 0.986544,0.1/1e6 * 0.011084],
                    'CH4':[1.7/1e6 * 0.988274,1.7/1e6 * 0.011103],
                    'H2O':[water_abundance_real*0.997317,water_abundance_real*0.002000]}
        
        total = sum(sum(lst) for lst in self.comp.values())

        # Make abundance changes to Nitrogen composition since it is inert, to make sure it equals one
        if total<1:
            self.comp['N2'][0] += 1-total
        elif total>1:
            self.comp['N2'][0] -= total-1

        # Tells HITRAN where to find data
        db_begin(filepath)

        pass

    def real_tellurics(self,strength):
        '''
        Creates the real telluric spectrum you want to apply to create your observations
        
        INPUTS:
        strength: this tells us the level of line intensity we should include should range from 10^{-33} - 10^{-25}
        pathlength: this is the pathlength of the line of sight (units km)

        OUTPUTS:
        lam: wavelength in nm 
        trans: transmittance spectrum
        '''
        # Create Components parameter based on composition

        # Some reason including more isotopes is resulting in tooo many differences
        # comp = [(22,1,self.comp['N2'][0]),
        #         (7,1,self.comp['O2'][0]),
        #         (2,1,self.comp['CO2'][0]),(2,2,self.comp['CO2'][1]),
        #         (3,1,self.comp['O3'][0]),
        #         (5,1,self.comp['CO'][0]),(5,2,self.comp['CO'][1]),
        #         (6,1,self.comp['CH4'][0]),(6,2,self.comp['CH4'][1]),
        #         (1,1,self.comp['H2O'][0]),(1,1,self.comp['H2O'][1])
        #         ]
        
        comp = [(22,1,self.comp['N2'][0]),
                (7,1,self.comp['O2'][0]),
                (2,1,self.comp['CO2'][0]/0.984204),
                (3,1,self.comp['O3'][0]),
                (5,1,self.comp['CO'][0]/0.986544),
                (6,1,self.comp['CH4'][0]/0.988274),
                (1,1,self.comp['H2O'][0]/0.997317),
                ]

        # Create Diluent parameter
        ## Adding this diluent parameter increases the absorption of the bands!! -> also increases the residuals??
        diluent = {'N2':self.comp['N2'][0],
                   'O2':self.comp['O2'][0],
                   'CO2':self.comp['CO2'][0],
                   'O3':self.comp['O3'][0],
                   'CO':self.comp['CO'][0],
                   'CH4':self.comp['CH4'][0],
                   'H2O':self.comp['H2O'][0]}
        
        # Create the absorption coefficients
        wavenumber, absorp = absorptionCoefficient_Voigt(SourceTables=["N2","O2","CO2","O3","CO","CH4",'H2O'],
                                                Components=comp, #Diluent=diluent, 
                                                Environment = {'T':self.T_real,'p':self.p_real},
                                                IntensityThreshold=strength,HITRAN_units=False)

        return wavenumber, absorp

    def correction_model(self,strength,sig_T,sig_p,seed=30):
        '''
        Creates the corrected model telluric spectrum you want to apply to correct your observations

        The main changes made here are:
        1. DEPRECATED: Only consider the most abundant isotope of each molecule 
        2. DEPRECATED: Dont set a seperate diluent parameter, use the default one of HITRAN
        3. Change the T and P values slightly
        4. Only consider stronger tellurics (therefore strength parameter should be higher than that provided in real_tellurics)
        
        INPUTS:
        strength: this tells us the level of line intensity we should include should range from 10^{-33} - 10^{-25}
        sig_T: the sigma for changing the temperature value
        sig_p: the sigma for changing the pressure value 
        seed: Just to set the randomness

        OUTPUTS:
        lam: wavelength in nm 
        trans: transmittance spectrum
        '''
        # Create Components parameter based on composition
        comp = [(22,1,self.comp['N2'][0]),
                (7,1,self.comp['O2'][0]),
                (2,1,self.comp['CO2'][0]/0.984204),
                (3,1,self.comp['O3'][0]),
                (5,1,self.comp['CO'][0]/0.986544),
                (6,1,self.comp['CH4'][0]/0.988274),
                (1,1,self.comp['H2O'][0]/0.997317),
                ]

        # Diluent parameter
        ## Adding this diluent parameter increases the absorption of the bands!! -> also increases the residuals??
        diluent = {'N2':self.comp['N2'][0],
            'O2':self.comp['O2'][0],
            'CO2':self.comp['CO2'][0],
            'O3':self.comp['O3'][0],
            'CO':self.comp['CO'][0],
            'CH4':self.comp['CH4'][0],
            'H2O':self.comp['H2O'][0]}

        np.random.seed(int(seed))
        T_mod = np.random.normal(loc=self.T_real,scale=sig_T)
        np.random.seed(int(seed+3))
        p_mod = np.random.normal(loc=self.p_real,scale=sig_p)

        # Create the absorption coefficients
        wavenumber, absorp = absorptionCoefficient_Voigt(SourceTables=["N2","O2","CO2","O3","CO","CH4",'H2O'],
                                                Components=comp, #Diluent=diluent,
                                                Environment = {'T':T_mod,'p':p_mod},
                                                IntensityThreshold=strength,HITRAN_units=False)

        return wavenumber, absorp, T_mod, p_mod


def coefficient_table_maker(input_values, coef_file_path, molec_file_path):
    '''
    This function is called to make new absorption coefficients for a parameter combination
    that hasn't already been saved. This function will compute the new coefficients and save it
    to the filepath that already stores the pre-existing coefficients. 

    INPUTS:
    input_values: this is an array of the form [correction, humidity, p, T, strength, sigp, sigT]
    file_path: this tells us where to save the newly computed absorption coefficients

    OUTPUTS:
    None, this function simply creates the file with the stored coefficients
    '''
    unique_items = np.unique(input_values, axis=0).shape[0]
    # Print statements to tell the user what's happening
    print("Cacluating new coefficients, using HITRAN API")
    approx_time = unique_items*45/60/60
    print("Have to calculate for "+str(unique_items)+" new atmosphere environments.")
    print("This will take approximately "+str(round(approx_time,2))+" hours.")

    # Extract values from input values
    bool_correction = input_values[:,0]
    humidity = input_values[:,1]
    p = input_values[:,2]
    T = input_values[:, 3]
    strength = input_values[:, 4]
    sigp = input_values[:, 5]
    sigT = input_values[:, 6]
    seed = input_values[:, 7]

    # Setup dataframes
    # This one will be passed back into the function
    new_coef = pd.DataFrame({'Correction?': bool_correction, 'Humidity': humidity,
                        'Pressure': p, 'Temperature': T, 'Strength':strength, 
                        'Sig_p':sigp, 'Sig_T':sigT,'seed':seed})
    new_coef['Wavenumber'] = None
    new_coef['AC'] = None

    #This one will be used to save the new environments
    simply_new = pd.DataFrame({'Correction?': bool_correction, 'Humidity': humidity,
                        'Pressure': p, 'Temperature': T, 'Strength':strength, 
                        'Sig_p':sigp, 'Sig_T':sigT,'seed':seed})
    simply_new['Wavenumber'] = None
    simply_new['AC'] = None

    # Loop through weather values and add the absoprtion coefficients to the dataframe but only those that dont duplicate
    wn_dict = {}
    ac_dict = {}
    count = 0
    for i in range(len(input_values)):
        key = tuple(input_values[i])
        if key not in wn_dict:
            model = Telluric_Model(T_real = T[i],p_real=p[i],water_abundance_real=humidity[i],filepath=molec_file_path)
            if new_coef['Correction?'].iloc[i]:
                wn, ac, T_mod, p_mod = model.correction_model(strength=strength[i], sig_T=sigT[i],sig_p=sigp[i],seed = seed[i])

            else:
                wn, ac= model.real_tellurics(strength=strength[i])
            wn_dict[key] = wn
            ac_dict[key] = ac

            simply_new.at[i,'Wavenumber'] = wn_dict[key]
            simply_new.at[i,'AC'] = ac_dict[key]
            count+=1

        new_coef.at[i,'Wavenumber'] = wn_dict[key] 
        new_coef.at[i,'AC'] = ac_dict[key] 

    # Drop the duplicate rows to save:
    simply_new = simply_new.dropna(subset=['Wavenumber'])
    if os.path.exists(coef_file_path):
        # File exists, so load the existing DataFrame
        existing_df = pd.read_pickle(coef_file_path)
        # Concatenate current_df with existing_df
        concatenated_df = pd.concat([existing_df, simply_new], ignore_index=True)
        # Save the concatenated DataFrame back to file
        concatenated_df.to_pickle(coef_file_path)
        print("Loaded existing DataFrame, concatenated and saved.")
    else:
        # File doesn't exist, so save the current DataFrame as a new file
        simply_new.to_pickle(coef_file_path)
        print("New DataFrame created and saved.")
    wn_dict = 0
    ac_dict = 0
    simply_new = 0
    existing_df = 0
    concatenated_df = 0
    return new_coef

def find_coefficients(input_values, coef_file_path, molec_file_path):
    '''
    This function will take in the input parameters that you want, it will check if they already exist in the file provided. 
    If they all exist in the file provided it will take them and compute the transmission spectra for them and keep passing
    down the input values
    If a couple dont exist, then we call the coefficient table maker for the ones that dont exist 
    this then returns a pd table of the input parameters and the coefficients and the wavenumber

    INPUTS:
    input_values = This is an array of the atmosphere conditions that we want to use for the real tellurics, and correction models 
    coef_file_path = The is the file path where the atmospheric coefficients are stored
    molec_file_path = This is the file path where the HITRAN molecular data is stored 

    OUTPUT:
    coef: pd dataframe with all of the input values and their corresponding wavenumber and coefficients
    '''

    if os.path.exists(coef_file_path):
        # These are the coefficients that have already been calculated
        existing_df = pd.read_pickle(coef_file_path)

        # This is where we will store the coefficients found based on input_values
        found_coef = pd.DataFrame(columns=existing_df.columns)

        #This is where we store the input_values not found
        not_found_indices = []
        not_found = []
        count = 0
        for i, row in enumerate(input_values):
            # If environment exists, add it to found coefficients
            if row[0] == 0:
                exists = existing_df.drop(columns=['Wavenumber','AC','seed']).eq(row[:-1]).all(axis=1)
            else:
                exists = existing_df.drop(columns=['Wavenumber','AC']).eq(row).all(axis=1)

            # If envrionemnt does not exist, add it to not found 
            if not any(exists):
                not_found_indices.append(i)
                not_found.append(row)

            # Otherwise add it to found coefficients
            else:
                if count ==0:
                    found_coef = existing_df[exists].iloc[[0]].copy()
                    count+=1
                else:
                    found_coef = pd.concat([found_coef,existing_df[exists].iloc[[0]]], ignore_index=True)
            

        
        # If everything was found, move on
        if not not_found:
            coef = found_coef.copy()

        # Calculate coefficients of environments not found
        else:
            not_found_coef = coefficient_table_maker(np.array(not_found),coef_file_path,molec_file_path)

            # Create a new dataframe to store the result
            coef = pd.DataFrame()

            ## If there was nothing already in the lookup table
            if len(not_found)==len(input_values):
                coef = not_found_coef.copy()

            ## Following code is to ensure coef is in same order as input_values
            ## We'll always have the correction model before the real tellurics according to
            # the code given before
            else:
                coef = pd.concat([not_found_coef, found_coef],ignore_index=True)


    else:
        coef = coefficient_table_maker(input_values,coef_file_path,molec_file_path)
    found_coef = 0 
    not_found_coef = 0
    existing_df = 0
    return coef



def transmission_spectra(wgrid,coef,pathlengths):
    '''
    This computes the transmission spectra for a given wgrid, atmosphere coefficients, and pathlengths.

    INPUTS:
    wgrid: the wgrid used for the rest of the toy model
    coef: the coefficients generated from hitran (pd dataframe)
    pathlengths: pathlengths in km (will be converted to cm)

    OUTPUTS:
    trans_spectra: pd dataframe of thet transmission spectra for each environment

    '''
    

    # Setup dataframe
    trans_spectra = coef[['Correction?','Humidity','Pressure', 'Temperature', 'Strength', 'Sig_p', 'Sig_T',"seed"]].copy()
    trans_spectra['Wavelength_nm'] = None
    trans_spectra['Transmittance'] = None

    for index, row in coef.iterrows():
        n,T = transmittanceSpectrum(row['Wavenumber'],row['AC'],Environment={'T':row['Temperature'],'l':pathlengths[index%len(pathlengths)]*1e5})

        # Change wavenumber (cm^-1) to nm and ensure the right orientation
        nu = 1/n * 1e7

        # Flip for proper interpolation
        nu = np.flip(nu)
        T = np.flip(T)

        # Now let's interpolate it such that I have the spectrum for the wavelength grid I defined 
        T_interp = np.interp(wgrid,nu,T)

        trans_spectra.at[index,'Wavelength_nm'] = wgrid
        trans_spectra.at[index,'Transmittance'] = T_interp

    return trans_spectra




def tellurics(environments, sigT, sigP, real_S, corr_S, pathlengths, wgrid, coef_filepath, molec_filepath, seeds):
    '''
    This function takes in various arguments and returns the real tellurics applied to observations
    as well as the correction model that we will apply to clean the tellurics. 

    INPUTS:
    environments: This should be an array consisting of [Humidity (vmr), Pressure (atm), Temperature (K)] for N observations
    sigT: the sigma_T for how different the corrected model temperature should be from the real one
    sigP: the sigma_P for how different the corrected model pressure should be from the real one
    real_S: the min strength of the lines for the real tellurics
    corr_S: the min strength of the lines for the corrected model tellurics (should be lower than real S)
    pathlengths: the pathlengths to integrate over for N observations
    wgrid: the wavelength grid used in the toy model
    coef_filepath: the filepath to find the absorbption coefficients
    molec_filepath: the filepath to find the HITRAN line data

    OUTPUTS:
    real_tellurics: the real telluric spectra to apply to N observations
    correction_model_tellurics: the correction model to clean the tellurics for N observations

    '''

    if len(environments) != len(pathlengths):
        raise ValueError("The length of environments should be the same length as pathlengths, the length for both should be the number of observations.")

    if corr_S<real_S:
        raise ValueError("The min strength of the corrected lines should be higher than the min strength of the real lines.")

    #### Create the input values: ####################################
    # Add in correction model column, with corr_S
    modified_array_1 = np.concatenate((np.ones((environments.shape[0], 1)), environments, np.full((environments.shape[0], 1), corr_S)), axis=1)

    # Add in correction model column, with real_S
    modified_array_2 = np.concatenate((np.zeros((environments.shape[0], 1)), environments, np.full((environments.shape[0], 1), real_S)), axis=1)

    # Combine the modified arrays vertically (seperated in half by corrected and real [1,1,1,1,0,0,0,0])
    input = np.vstack((modified_array_1, modified_array_2))

    # Add sig_P and sig_T 
    # Repeat the end_values to match the number of rows in array_2d
    end_values_repeated = np.tile([sigP,sigT,seeds], (input.shape[0], 1))

    # Concatenate the original array with the end_values
    input = np.concatenate((input, end_values_repeated), axis=1)

    ### Compute the transmission spectrum #########################################
    coef = find_coefficients(input,coef_filepath,molec_filepath)
    print("HITRAN Coefficients Done")
    trans = transmission_spectra(wgrid,coef,pathlengths)

    
    ### Seperate into real and corrected #########################################
    correction_model_tellurics = trans.loc[trans['Correction?'] == 1, 'Transmittance'].to_numpy()
    real_tellurics = trans.loc[trans['Correction?'] == 0, 'Transmittance'].to_numpy()
    
    return np.vstack(real_tellurics), np.vstack(correction_model_tellurics)
    

    

