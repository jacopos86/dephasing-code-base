import yaml
from pydephasing.log import log
from pydephasing.input_parameters import p
import os
import numpy as np
# restart calculation function
def restart_calculation(restart_file):
    # open file
    try:
        f = open(restart_file)
    except:
        msg = "could not find: " + restart_file
        log.error(msg)
    data = yaml.load(f, Loader=yaml.Loader)
    f.close()
    # extract data
    if 'T2_list' in data:
        T2_list = data['T2_list']
    if 'Delt_list' in data:
        Delt_list = data['Delt_list']
    if 'tauc_list' in data:
        tauc_list = data['tauc_list']
    if 'ic0' in data:
        ic0 = data['ic0']
    # load acf aver data
    acf_data = []
    namef = p.write_dir + "/acf-aver.npy"
    acf_aver = np.load(namef)
    acf_data.append(acf_aver)
    # at res.
    if p.at_resolved:
        namef = p.write_dir + "/acf-atr-aver.npy"
        acf_atr_aver = np.load(namef)
        acf_data.append(acf_atr_aver)
    if p.ph_resolved:
        namef = p.write_dir + "/acf-phr-aver.npy"
        acf_phr_aver = np.load(namef)
        acf_data.append(acf_phr_aver)
    # return data
    return ic0, T2_list, Delt_list, tauc_list, acf_data
# save data on file
def save_data(ic, T2_list, Delt_list, tauc_list, acf_data):
    # write on file
    restart_file = p.write_dir + "/restart_calculation.yml"
    isExist = os.path.isdir(restart_file)
    if isExist:
        os.remove(restart_file)
    # dictionary
    dict = {'T2_list': T2_list, 'Delt_list': Delt_list, 'tauc_list': tauc_list, 'ic0': ic+1}
    # array save
    namef = p.write_dir + "/acf-aver"
    np.save(namef, acf_data[0])
    if p.at_resolved:
        namef = p.write_dir + "/acf-atr-aver"
        np.save(namef, acf_data[1])
    if p.ph_resolved and len(acf_data) == 2:
        namef = p.write_dir + "/acf-phr-aver"
        np.save(namef, acf_data[1])
    elif p.ph_resolved and len(acf_data) == 3:
        namef = p.write_dir + "/acf-phr-aver"
        np.save(namef, acf_data[2])
    # save data
    with open(restart_file, 'w') as out_file:
        yaml.dump(dict, out_file)