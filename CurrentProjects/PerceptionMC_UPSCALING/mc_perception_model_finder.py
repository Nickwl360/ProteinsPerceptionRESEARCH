from scipy.optimize import optimize
from mc_perception_frwd_cl import calc_next_state
import global_params as gp
import pandas as pd
import os


rmax = gp.MAXTOP
emax = gp.MAXBOT

def save_data(params, state, p_arr):
    hap,ham,hbp,hbm,hcp,hcm,hdp,hdm,kcoop,kcomp,kdu,kud,kx = params
    a,b,c,d = state
    data_dir ='saved_probability_arrays'
    save_name = f'E{emax}R{rmax}_calced.csv'

    os.makedirs(data_dir,exist_ok=True)
    out_name = os.path.join(data_dir,save_name)


    if not os.path.exists(f'saved_probability_arrays/E{emax}R{rmax}_calced.csv'):
        df =pd.DataFrame({'hap':hap, 'ham':ham, 'hcp':hcp, 'hcm':hcm, 'kcoop':kcoop,'kcomp':kcomp, 'kdu':kdu,'kud':kud,'kx':kx,'a':a,'b':b,'c':c,'d':d,'p_arr':p_arr})
        df.to_csv(out_name,index=False)


    return



def cl_likelyhood(params, data):
    a_data,b_data, c_data, d_data = data
    data_length = len(a_data)

    for i,_ in enumerate(a_data):
        state = (a_data[i],b_data[i],c_data[i], d_data[i])
        # check if state & params -> p_arr is already saved

        #if saved: load
        #else: create, and save

        #add to likelyhood

        #delete


    return