import os
import numpy as np
import yaml
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
# restart calculation function
def restart_calculation(restart_file):
    # open file
    try:
        f = open(restart_file)
    except:
        msg = "\t could not find: " + restart_file
        log.error(msg)
    data = yaml.load(f, Loader=yaml.Loader)
    f.close()
    # extract data
    if 'T2i_obj' in data:
        T2i_obj = data['T2i_obj']
    if 'lw_obj' in data:
        lw_obj = data['lw_obj']
    if p.time_resolved:
        if 'Delt_obj' in data:
            Delt_obj = data['Delt_obj']
        if 'tauc_obj' in data:
            tauc_obj = data['tauc_obj']
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
        if p.ph_resolved and p.nphr == 0:
            namef = p.write_dir + "/acf-wql-aver.npy"
            acf_wql_aver = np.load(namef)
            acf_data.append(acf_wql_aver)
        elif p.ph_resolved and p.nphr > 0:
            namef = p.write_dir + "/acf-wql-aver.npy"
            acf_wql_aver = np.load(namef)
            acf_data.append(acf_wql_aver)
            namef = p.write_dir + "/acf-phr-aver.npy"
            acf_phr_aver = np.load(namef)
            acf_data.append(acf_phr_aver)
    else:
        if p.ph_resolved and p.nphr == 0:
            namef = p.write_dir + "/acf-wql-aver.npy"
            acf_wql_aver = np.load(namef)
            acf_data.append(acf_wql_aver)
        elif p.ph_resolved and p.nphr > 0:
            namef = p.write_dir + "/acf-wql-aver.npy"
            acf_wql_aver = np.load(namef)
            acf_data.append(acf_wql_aver)
            namef = p.write_dir + "/acf-phr-aver.npy"
            acf_phr_aver = np.load(namef)
            acf_data.append(acf_phr_aver)
    # return data
    if p.time_resolved:
        restart_data = [ic0, T2i_obj, Delt_obj, tauc_obj, lw_obj, acf_data]
    elif p.w_resolved:
        restart_data = [ic0, T2i_obj, lw_obj, acf_data]
    return restart_data
# save data on file
def save_data(ic, T2_calc_handler, acf_data):
    # write on file
    restart_file = p.write_dir + "/restart_calculation.yml"
    isExist = os.path.isdir(restart_file)
    if isExist:
        os.remove(restart_file)
    # extract objects
    T2i_obj = T2_calc_handler.T2_obj
    lw_obj = T2_calc_handler.lw_obj
    if p.time_resolved:
        tauc_obj = T2_calc_handler.tauc_obj
        Delt_obj = T2_calc_handler.Delt_obj
        # dictionary
        dict = {'T2i_obj': T2i_obj, 'Delt_obj': Delt_obj, 'tauc_obj': tauc_obj, 'lw_obj' : lw_obj, 'ic0': ic+1}
    elif p.w_resolved:
        # dictionary
        dict = {'T2i_obj': T2i_obj, 'lw_obj' : lw_obj, 'ic0': ic+1}
    # array save
    namef = p.write_dir + "/acf-aver"
    np.save(namef, acf_data[0])
    if p.at_resolved:
        namef = p.write_dir + "/acf-atr-aver"
        np.save(namef, acf_data[1])
        if p.ph_resolved and len(acf_data) == 3:
            namef = p.write_dir + "/acf-wql-aver"
            np.save(namef, acf_data[2])
        elif p.ph_resolved and len(acf_data) == 4:
            namef = p.write_dir + "/acf-wql-aver"
            np.save(namef, acf_data[2])
            namef = p.write_dir + "/acf-phr-aver"
            np.save(namef, acf_data[3])
    else:
        if p.ph_resolved and len(acf_data) == 2:
            namef = p.write_dir + "/acf-wql-aver"
            np.save(namef, acf_data[1])
        elif p.ph_resolved and len(acf_data) == 3:
            namef = p.write_dir + "/acf-wql-aver"
            np.save(namef, acf_data[1])
            namef = p.write_dir + "/acf-phr-aver"
            np.save(namef, acf_data[2])
    # save data
    with open(restart_file, 'w') as out_file:
        yaml.dump(dict, out_file)