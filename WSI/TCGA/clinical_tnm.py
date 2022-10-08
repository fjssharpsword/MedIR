import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from os import listdir
from os.path import isfile, join
import scipy
import openslide
from PIL import Image

def main():
    #load survival information
    root_path = '/data_local/ljjdata/TCGA/clinical/BRCA/'
    sur_data = pd.read_csv(root_path+'survive_info_1133.csv', sep=',')
    #sur_data = sur_data.values #dataframe -> numpy
    pid_list = sur_data['patient_id'].to_list()

    #load json file
    with open(root_path+'clinical_1133.json','r') as cli_file:
        cli_list = json.load(cli_file)

    tnm_list = []
    for pid in pid_list:
        tnm_stage = 'no'
        for cli_dict in cli_list:
            if 'diagnoses' in cli_dict.keys():
                diag_dict = cli_dict['diagnoses'][0]
                if pid+'_diagnosis' == diag_dict['submitter_id']:
                    if 'ajcc_pathologic_stage' in diag_dict.keys():
                        tnm_stage = diag_dict['ajcc_pathologic_stage'].replace('Stage','').strip()
        tnm_list.append(tnm_stage) 
    sur_data['tnm'] = tnm_list
    #print(sur_data['tnm'].value_counts())
    sur_data.to_csv('/data/pycode/MedIR/WSI/data/tcga_brca_clinical.csv', index=False, sep=',')

if __name__ == '__main__':
    main()