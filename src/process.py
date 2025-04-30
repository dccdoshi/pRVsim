import numpy as np
from hapi import *
from interpolater import *
from convolution import *

class Process():

    def __init__(self, wavegrid, spectra):
        self.wgrid = wavegrid.copy()
        self.spectra = spectra.copy()

    def normalize_spectra(self,const,SNR):
        self.spectra = self.spectra/const

        sig = (1/SNR)*self.spectra

        return self.spectra.copy(), sig.copy()

    def remove_tellurics(self,correct_tellurics,inst_res,inst_wgrid,phoenix_wgrid):
        wgrid, spec = gauss_convolve(phoenix_wgrid, correct_tellurics, inst_res, n_fwhm=7, res_rtol=1e-6, mode='valid', i_plot=True)
        inst_correct_tellurics = cubic_spline_torch(torch.from_numpy(wgrid),torch.from_numpy(spec), torch.from_numpy(inst_wgrid))

        self.spectra /= inst_correct_tellurics
        return self.spectra

    def remove_bad_regions(self):
        '''
        This removes the regions in the spectra that have less than 10% of transmission
        in order to deal with the tellurics correctly. Will also remove the worst telluric region. 
        '''

        # bad_regions = np.where((self.spectra<0.1) | (self.spectra>1.05))
        # self.removed = self.noisy_data.copy()
        if bool:
            self.spectra[bad_regions] = np.nan

            # Define minimum and maximum values
            min_val = 1350 #1320 #1300 #1350
            max_val = 1420 #1500 #1470 #1450 , 1420

            # Find indices where values are between min and max
            worst_tell = np.where((self.wgrid > min_val) & (self.wgrid < max_val))[0]
            self.spectra[:,worst_tell] = np.nan

        return self.spectra


    def correct_tellurics(self, bool, corrector):
        # self.noisy_corrected = self.removed.copy()
        if bool:
            # not_nans = np.where(~np.isnan(self.spectra))
            # self.spectra[not_nans] = self.spectra[not_nans]/corrector[not_nans]

            self.spectra = self.spectra/corrector
            

        return self.spectra
    
