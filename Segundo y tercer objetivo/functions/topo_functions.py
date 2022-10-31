import os
import pickle
import numpy as np

#-------------------------------------------------------------------------------
def TW_data(sbj,time_inf,time_sup):
    # Load data/images----------------------------------------------------------
    path_cwt = os.path.join(os.getcwd(),'data','CWT_CSP_data_mubeta_8_30_Tw_'+str(time_inf)+'s_'+str(time_sup)+'s_subject'+str(sbj)+'_cwt_resized_10.pickle') 
    with open(path_cwt, 'rb') as f:
         X_train_re_cwt, X_test_re_cwt, y_train, y_test = pickle.load(f)
    path_csp = os.path.join(os.getcwd(),'data','CWT_CSP_data_mubeta_8_30_Tw_'+str(time_inf)+'s_'+str(time_sup)+'s_subject'+str(sbj)+'_csp_resized_10.pickle' ) 
    with open(path_csp, 'rb') as f:
         X_train_re_csp, X_test_re_csp, y_train, y_test = pickle.load(f)
    #---------------------------------------------------------------------------
    return X_train_re_cwt, X_train_re_csp, X_test_re_cwt, X_test_re_csp, y_train, y_test
#-------------------------------------------------------------------------------
def norm_data(XF_train_cwt, XF_train_csp, XF_test_cwt, XF_test_csp, n_fb, Ntw, y_train, y_test, fld):
    # orden de las inputs:------------------------------------------------------
    # [CWT_fb1_TW1, CWT_fb2_TW1 --- CWT_fb1_TW2, CWT_fb2_TW2 --- CWT_fb1_TWN, CWT_fb2_TWN] ... [CSP]
    #---------------------------------------------------------------------------
    XT_train_csp = []
    XT_valid_csp = []
    XT_test_csp  = []
    XT_train_cwt = []
    XT_valid_cwt = []
    XT_test_cwt  = []
    for tw in range(Ntw):
        for fb in range(n_fb):
            X_train_cwtf, X_test_cwt = XF_train_cwt[tw][:,fb,:,:].astype(np.uint8), XF_test_cwt[tw][:,fb,:,:].astype(np.uint8)
            X_train_cspf, X_test_csp = XF_train_csp[tw][:,fb,:,:].astype(np.uint8), XF_test_csp[tw][:,fb,:,:].astype(np.uint8)
            #-------------------------------------------------------------------        
            # Normalize data----------------------------------------------------
            X_mean_cwt  = X_train_cwtf.mean(axis=0, keepdims=True)
            X_std_cwt   = X_train_cwtf.std(axis=0, keepdims=True) + 1e-7
            X_train_cwt = (X_train_cwtf - X_mean_cwt) / X_std_cwt
            X_test_cwt  = (X_test_cwt  - X_mean_cwt) / X_std_cwt

            X_mean_csp  = X_train_cspf.mean(axis=0, keepdims=True)
            X_std_csp   = X_train_cspf.std(axis=0, keepdims=True) + 1e-7
            X_train_csp = (X_train_cspf - X_mean_csp) / X_std_csp
            X_test_csp  = (X_test_csp  - X_mean_csp) / X_std_csp
            #-------------------------------------------------------------------
            # set new axis------------------------------------------------------
            X_train_cwt = X_train_cwt[..., np.newaxis]
            X_test_cwt  = X_test_cwt[..., np.newaxis]   
            XT_train_cwt.append(X_train_cwt)
            XT_test_cwt.append(X_test_cwt)
                                
            X_train_csp = X_train_csp[..., np.newaxis]
            X_test_csp  = X_test_csp[..., np.newaxis]   
            XT_train_csp.append(X_train_csp)
            XT_test_csp.append(X_test_csp)
            #-------------------------------------------------------------------
    y_trainF,  y_testF = y_train.reshape((-1,))-1,  y_test.reshape((-1,))-1
    #---------------------------------------------------------------------------
    XT_train = XT_train_cwt + XT_train_csp
    XT_test  = XT_test_cwt  + XT_test_csp
    #---------------------------------------------------------------------------
    return XT_train,  XT_test, y_trainF,  y_testF 
#-------------------------------------------------------------------------------