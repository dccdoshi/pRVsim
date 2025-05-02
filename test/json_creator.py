import json, argparse
import numpy as np 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Json Creator')

    parser.add_argument('--param', type=str, help="file name for json attributes")
    args = parser.parse_args()
    param_file = args.param
     
    # Data to be written to the JSON file
    data = {
        "stellar": {
            "order": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
 38, 39, 40, 41, 42, 43, 44, 45, 46,47], #spirou order
            "R_start": 200_000,
            "oversample": 5
        },
        "instrument":{
            "broadening": 70_000,
            "use_inst": True,
            "inst_resolution": [100_000]
        },
        "radial_vel_obs": {
            "Ntemp": [0,5,10,25,50,100],
            "syst_vel": 0, #in km/s
            "Kamp": 5,#in m/s
            "SNR": [10,25,50,100,300,1e5], #SNR for gaussian noise
            "seeds": [1,10,20,30,40,50,60,70,80,90], # seeds for random instances
            "date_seed": [43],
        }
    }

    # Write data to JSON file
    with open(param_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
