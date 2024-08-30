import numpy as np
import pandas as pd
#import math
import matplotlib.pyplot as plt
import subprocess
#import matplotlib.cm as cm
import os
import csv
#SaveFigDir='/home/bkh/OnlineCourses/ProlateSpheroid/Figs/'
from settings import LiquidFilledSettings, AirFilledSettings
from IterativeSolvers import IterativeRefinement, PreconditionedIterativeRefinement, ILUPreconditioner, BiCGSTABSolver, GMResSolver

CurrentDir = os.getcwd()
ParentDIR=os.path.split(os.getcwd())[0]


# http://mathworld.wolfram.com/SphericalHankelFunctionoftheFirstKind.html
# Spherical Hankel Function of the First Kind
# The spherical Hankel function of the first kind h1_n(z) is defined by:
#    h1_n(z)=j_n(z)+i*y_n(z)
# where j_n(z) and y_n(z) are the spherical Bessel functions of the first and second kinds.
#%% INPUTs
#settings = LiquidFilledSettings()
settings = AirFilledSettings()

ro_w = settings.ro_w
ro_s = settings.ro_s
c_w = settings.c_w
c_s = settings.c_s
a = settings.a
b = settings.b
d = 2*((a*a-b*b)**0.5)
AspRatio = a / b
kisi0=(1-(b/a)*(b/a))**(-0.5)
DeltaF = settings.delta_f
Theta_i_deg = settings.theta_i_deg

min_freq = settings.min_freq
max_freq = settings.max_freq
freq_vec=np.arange(min_freq,max_freq,DeltaF)

solver = IterativeRefinement('LU')
#solver = PreconditionedIterativeRefinement(ILUPreconditioner(), IterativeRefinement('LU'))
#solver = GMResSolver(ILUPreconditioner())
#solver = BiCGSTABSolver(ILUPreconditioner())

freq_resp_file = ParentDIR+'/temp/'+'ts_vs_freq_{}_a_{}_b_{}_f1_{}_f2_{}_rhos_{:.2f}_IncAngle_{}_{}.csv'.format(settings.prefix, a, b, int(min_freq / 1000),
                                                                                      int(freq_vec[-1] / 1000), ro_s, Theta_i_deg,
                                                                                      solver.solver_name())
frf = open(freq_resp_file, 'wt', newline='', encoding='utf-8')
freq_resp_writer = csv.writer(frf, delimiter=',')
freq_resp_writer.writerow(['Freq_kHz', 'TS'])

# ro_w=1027 #ro_water #1000    # Surronding Water density  "kg/m³"
# ro_s=10 #ro_w*1.05 #ro_scatterer #1*1.24 #14860   # inner sphere Density "kg/m³"
#
# c_w=1500 #c_water #1480     # Surronding Water speed of sound "m/s"
# c_s=343 #1500*1.05 #c_scatterer # 1*343     #  gas speed of sound "m/s"
#
# a=0.03 #MajorSemiAxis
# b=0.01 #MinorSemiAxis
# d=2*((a*a-b*b)**0.5)  # interfocal distance of ellipse
# AspRatio=a/b
# kisi0=(1-(b/a)*(b/a))**(-0.5)   #0.0025  # (m) radius of gas sphere
#
# DeltaF=4000
# min_freq=269000  #freq1 # 1*500  # (Hz) The maximum frequency (Hz) that we want its TS
# max_freq=273100 #freq1 # 1*120000  # (Hz) The maximum frequency (Hz) that we want its TS
# freq_vec=np.arange(min_freq,max_freq,DeltaF)
#
# Theta_i_deg=30.0
Theta_propagate_positive_deg = float(Theta_i_deg)
Theta_propagate_positive=Theta_propagate_positive_deg*np.pi/180   
Theta_i=np.pi-Theta_propagate_positive # Incident angle (See Fig 1 in Masahiko Furusawa, Prolate Spheroidal models for predicting general trends of fish target strength, J. Acoust. Soc. Jpn 1988)
                                       # Also see Silbiger 1963 Eq1

#m=0 #m_order
#n=0 #n_order
#M_order=5
#N_order=5


Theta_s_deg=180-Theta_i_deg

Phi_i=0
Phi_b=np.pi+Phi_i


#%% Write profcn.dat file
'''
# Structure of parameters in "profcn.dat"
#!       line 1:
#!          mmin   : minimum value for m. (integer)
#!          minc   : increment for m. (integer)
#!          mnum   : number of values of m. (integer)
#!          lnum   : number of values of l [l=m, l=m+1,
#!                   ..., l=m+lnum-1]. (integer)
#!
#!       line 2:
#!          ioprad : (integer)
#!                 : =0 if radial functions are not computed
#!                 : =1 if radial functions of only the first kind
#!                      and their first derivatives are computed
#!                 : =2 if radial functions of both kinds and
#!                      their first derivatives are computed.
#!                      Note only radial functions of the first
#!                      kind and their first derivatives can be
#!                      computed if x = 1.0. 
#!
#!          iopang : (integer)
#!                 : =0 if angular functions are not computed
#!                 : =1 if angular functions of the first kind
#!                      are computed
#!                 : =2 if angular functions of the first kind and
#!                      their first derivatives are computed
#!
#!          iopnorm: (integer)
#!                 : =0 if not scaled. The angular functions have
#!                      the same norm as the corresponding associated
#!                      Legendre function [i.e., we use the Meixner-
#!                      Schafke normalization scheme.]
#!                 : =1 if angular functions of the first kind
#!                      (and their first derivatives if computed)
#!                      are scaled by the square root of the
#!                      normalization of the corresponding
#!                      associated Legendre function. The resulting
#!                      scaled angular functions have unity norm.
#!
#!       line 3:
#!          c      : value of the size parameter (= kd/2, where k =
#!                   wavenumber and d = interfocal length) (real(knd))
#!          x1     : value of the radial coordinate x minus one (real(knd))
#!                   (a nominal value of 10.0e0_knd can be entered for x1
#!                   if ioprad = 0). x1 must be greater than 0.0e0_knd
#!                   if radial functions of both the first and second
#!                   kind are computed, i.e., ioprad must = 1 when x1 =
#!                   0.0e0_knd. Also, when x1 = 0.0e0_knd, the radial
#!                   functions of the first kind and their first derivatives
#!                   are equal to 0.0e0_knd unless m = 0. And the radial
#!                   functions of the second kind and their first derivatives
#!                   are infinite for all m.  
#!                  1.0327955589d0 corresponds to the aspect ratio of "4"
#!                  x1= 0.0327955589d0 is used    
#!
#!       line 4:
#!          ioparg : (integer)
#!                 : =0 if both arg1 and darg are angles in degrees
#!                 : =1 if arg1 and darg are dimensionless values of eta
#!
#!          eta1, eta2   : Values for the angle coordinate (dimensionless eta, eta=cos(Theta)) for which angular
#!                   functions are to be computed. (real(knd))
#!
#!          darg   : increment used to calculate additional desired
# mmin, minc, mnum, lnum
# ioprad, iopang, iopnorm
# c, x1
# ! x1 is value of the radial coordinate "x" minus one
                      
# ioparg, eta1, eta2
'''
# Run Param Fortran script
script_path1 = ParentDIR+'/src/'+'Run_param_fort_FromPython.py'
print(script_path1)
# Run Python Script:
subprocess.run(['python', script_path1], capture_output=True, text=True)
#>>>>>>>>>

def write_Inputfile(c_nondimensional,AspectRatio,M_order,N_order,Theta_IncDeg, _parent_dir=ParentDIR):
    mmin=0   # minimum value for m. (integer)
    minc=1   # increment for m. (integer)
    mnum=M_order   # number of values of m. (integer)
    lnum=N_order   # number of values of l [l=m, l=m+1,
           #            ..., l=m+lnum-1]. (integer)
    ioprad=2
    iopang=2
    iopnorm=1
    
#    c=c_nondimensional
    kisi0=(1-(1/AspectRatio)*(1/AspectRatio))**(-0.5) 
    x1=kisi0-1
    
    ioparg=1
    eta1=np.cos(np.pi*Theta_IncDeg/180)
    eta2=np.cos(np.pi-np.pi*Theta_IncDeg/180)   

    file_name = _parent_dir+'/src/'+'profcn.dat'
    
# Writing the NumPy array to the DAT file
    #Arr = np.array([], dtype=np.float64)
    Arr=[[mmin,minc,mnum,lnum],
                    [ioprad,iopang,iopnorm],
                    [c_nondimensional,x1],
                    [ioparg,eta1,eta2]]
    with open(file_name, 'w') as file:
        for row in Arr:
            # Convert each element to string and join them with a space
            line = ','.join(map(str, row))
            # Write the line to the file
            file.write(line + '\n')

#How to use this function:
'''
c_nondimensional=10
Theta_IncDeg=Theta_i_deg
write_Inputfile(c_nondimensional,AspRatio,M_order,N_order,Theta_IncDeg)
'''

#%%
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||   Read the FORTRAN Outputs ||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def infunc_content_Rad_SWF(_parent_dir=ParentDIR):

    DirFortran=_parent_dir+'/src/'
    fort20File=DirFortran+'fort.20'
    '''
    fort.20 contains  
    x, c, m  followed by 
    l, r1c, ir1e, r1dc, ir1de
    l+1, r1c, ir1e, r1dc, ir1de
    .
    .
    .
    '''
    with open(fort20File) as myFile1:
         content_Rad_SWF = myFile1.readlines()
    
    return content_Rad_SWF


def infunc_content_Ang_SWF(_parent_dir=ParentDIR):

    DirFortran=_parent_dir+'/src/'
    fort30File=DirFortran+'fort.30'
    '''
    fort.30 contains  
    x, c, m  followed by 
    l, 
    r1c, ir1e, r1dc, ir1de
    l+1, r1c, ir1e, r1dc, ir1de
    .
    .
    .
    '''
    with open(fort30File) as myFile1:
         content_Ang_SWF = myFile1.readlines()
         
    return content_Ang_SWF
 
    
def infunc_content_dr_values(_parent_dir=ParentDIR):

    DirFortran=_parent_dir+'/src/'
    fort70File=DirFortran+'fort.70'
    '''
    fort.70 contains  
    x, c, m  followed by 
    l, dmlf , idmlfe followed by, d(l-m)
    l=m, enr(d(2k+2+ix)/d(2k+ix))
    l=m+1, ...
    .
    .
    .
    l=m+lnum-1
    
    again
    
    x, c, m+1
    .
    .
    .
     
    '''
    with open(fort70File) as myFile1:
         content_dr_values = myFile1.readlines()
             
    return content_dr_values     

#%%

def func_Generate_Partition_Matrices(content_Rad_SWF_hs, content_dr_values_hs, 
                       content_Rad_SWF_hw, content_dr_values_hw,
                       Theta_i_deg,
                       _m, N_order_Fortran, _N_col_start, _N_col_end, _N_row_start, _N_row_end):
    
    Theta_s_deg=180-Theta_i_deg
    
    if _m == 0:
       Em=1
    else:
       Em=2
       
    Part_C_Beta_D_Matrix_Arg=np.zeros((_N_row_end - _N_row_start, _N_col_end - _N_col_start), dtype = 'complex')
    Part_C_Beta_D_Matrix_Exp=np.zeros((_N_row_end - _N_row_start, _N_col_end - _N_col_start), dtype = 'float')
    
    Part_C_Beta_F_Mat_Arg = np.zeros((_N_row_end - _N_row_start, _N_col_end - _N_col_start), dtype = 'complex')
    Part_C_Beta_F_Mat_Exp = np.zeros((_N_row_end - _N_row_start, _N_col_end - _N_col_start), dtype = 'float')
    
    Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg=np.zeros((1, _N_row_end - _N_row_start), dtype = 'complex')
    Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp=np.zeros((1, _N_row_end - _N_row_start), dtype = 'float')

    # This section calculates right partition *************************** 
    ll=-1
    for l in range(_m + _N_col_start, _N_col_end + _m): # l index Loop: l=m,1,2,...,N : This fills the columns in Matrices
        
        ## PSWF values ==================================================================================================
        R_ml_hs_kisi0=func_Rad_SWF_ArgExp(content_Rad_SWF_hs, _m, l , N_order_Fortran) #func_Adelman_Rmn(DirPSWF, AspRatio, m, l, hs, Prec) # Prolate spheroidal radial function of the first kind >>  R1(m,n,hw,kisi0)
        R1_ml_hs_kisi0=R_ml_hs_kisi0[0]
        dR1_ml_hs_kisi0=R_ml_hs_kisi0[1]

        dml_r_hs_ArgExp=func_dr_values_ArgExp(content_dr_values_hs, _m, l , N_order_Fortran) #func_Adelman_dr_mn_of_c(DirPSWF, m, l, hs, Prec)

        
        nn=-1
        ll=ll+1
        
        for n in range(_m + _N_row_start , _N_row_end + _m):  # n index Loop: n=m,1,2,...,N : This fills the rows in Matrices
            nn=nn+1
            ## PSWF values ==================================================================================================
#                Smn_hw_eta_is=func_Ang_SWF(content_Ang_SWF_hw, m, n , N_order) #func_Adelman_Smn_Const_Eta_fof_c(DirPSWF, Theta_i_deg, m, n, hw, Prec)  # Prolate spheroidal angular function of the first kind >> S(m,n,hw,cos(Theta_i))
#                Smn_hw_eta_i=Smn_hw_eta_is[0]
#                Smn_hw_eta_s=Smn_hw_eta_is[1]
#                Nmn_hw=func_Adelman_Nmn_of_c(DirPSWF, m, n, hw, Prec)
            Nmn_hw=1
            dmn_r_hw_ArgExp=func_dr_values_ArgExp(content_dr_values_hw, _m, n , N_order_Fortran)# func_Adelman_dr_mn_of_c(DirPSWF, m, n, hw, Prec)
            
            R_mn_hw_kisi0=func_Rad_SWF_ArgExp(content_Rad_SWF_hw, _m, n , N_order_Fortran) #func_Adelman_Rmn(DirPSWF, AspRatio, m, n, hw, Prec) # Prolate spheroidal radial function of the first kind >>  R1(m,n,hw,kisi0)
          
            eta_i=np.cos(np.pi*Theta_i_deg/180)
            Smn_hw_eta_i=func_Smn_eta_c_from_dr_ArgExp(_m, n, eta_i, dmn_r_hw_ArgExp)
            
            eta_s=np.cos(np.pi*Theta_s_deg/180)
            Smn_hw_eta_s=func_Smn_eta_c_from_dr_ArgExp(_m, n, eta_s, dmn_r_hw_ArgExp)
            
            R1_mn_hw_kisi0 = R_mn_hw_kisi0[0] # [Arg, Exp]
            dR1_mn_hw_kisi0 = R_mn_hw_kisi0[1]
            R2_mn_hw_kisi0 = R_mn_hw_kisi0[2]
            dR2_mn_hw_kisi0 = R_mn_hw_kisi0[3]
            R3_mn_hw_kisi0 = Sum_ArgExp(R1_mn_hw_kisi0, Multi_ArgExp([1j,0], R2_mn_hw_kisi0))
            dR3_mn_hw_kisi0 = Sum_ArgExp(dR1_mn_hw_kisi0, Multi_ArgExp([1j,0], dR2_mn_hw_kisi0))
            ## ===============================================================================================================

               
            # Calculate Coefficient ------------------------------------------
            C_n_m=Multi_ArgExp(Get_ArgExp((1j**n)*Em),Smn_hw_eta_i) #(1j**n)*Em*Smn_hw_eta_i/Nmn_hw
#                 D_nl_m = (ro_s/ro_w)*dR3_mn_hw_kisi0*R1_ml_hs_kisi0 - R3_mn_hw_kisi0*dR1_ml_hs_kisi0
            D_nl_m = Multi_ArgExp(Get_ArgExp(ro_s/ro_w), Multi_ArgExp(dR3_mn_hw_kisi0, R1_ml_hs_kisi0)) 
            D_nl_m = Sum_ArgExp(D_nl_m, Multi_ArgExp([-1,0], Multi_ArgExp(R3_mn_hw_kisi0, dR1_ml_hs_kisi0)))
            
#                F_nl_m=(ro_s/ro_w)*dR1_mn_hw_kisi0*R1_ml_hs_kisi0 - R1_mn_hw_kisi0*dR1_ml_hs_kisi0
            F_nl_m = Multi_ArgExp(Get_ArgExp(ro_s/ro_w), Multi_ArgExp(dR1_mn_hw_kisi0, R1_ml_hs_kisi0 )) 
            F_nl_m = Sum_ArgExp(F_nl_m, Multi_ArgExp([-1,0], Multi_ArgExp(R1_mn_hw_kisi0, dR1_ml_hs_kisi0))) 
            
            Beta_nl_m=func_Integrate_SmnEta1_SmlEta2_ArgExp(_m, n, l, dmn_r_hw_ArgExp, dml_r_hs_ArgExp)


            # Filling the l_th column
#               #   C_Beta_D_Matrix[nn][ll]=C_n_m*Beta_nl_m*D_nl_m
            TEMPvar=Multi_ArgExp(C_n_m, Multi_ArgExp(Beta_nl_m, D_nl_m))
            Part_C_Beta_D_Matrix_Arg[nn][ll]=TEMPvar[0]
            Part_C_Beta_D_Matrix_Exp[nn][ll]=TEMPvar[1]
            
           
            # We will save C_Beta_F_Mat[nn][ll] and then summation of column-wise summation provides C_Beta_F_Vector
            Minus_C_n_m=Multi_ArgExp([-1,0], C_n_m)
            Beta_nl_m_x_F_nl_m=Multi_ArgExp(Beta_nl_m, F_nl_m)
            TEMPvar = Multi_ArgExp(Minus_C_n_m, Beta_nl_m_x_F_nl_m)
            Part_C_Beta_F_Mat_Arg[nn][ll] = TEMPvar[0]
            Part_C_Beta_F_Mat_Exp[nn][ll] = TEMPvar[1]
            
            if ll==0:
                Em_x_Smn_hw_eta_i=Multi_ArgExp(Get_ArgExp(Em), Smn_hw_eta_i)
                Smn_hw_eta_s_X_cosmpi=Multi_ArgExp(Smn_hw_eta_s, Get_ArgExp(np.cos(m*np.pi)))
                
                TEMPvar=Multi_ArgExp(Em_x_Smn_hw_eta_i, Smn_hw_eta_s_X_cosmpi)
                Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg[0][nn]=TEMPvar[0]
                Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp[0][nn]=TEMPvar[1]
 
    return [Part_C_Beta_D_Matrix_Arg,
            Part_C_Beta_D_Matrix_Exp,
            Part_C_Beta_F_Mat_Arg,
            Part_C_Beta_F_Mat_Exp,
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg,
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp] 


def Estimate_fmn_from_CBetaD_Mat_CBetaFvec(C_Beta_D_Matrix_Arg,
                                           C_Beta_D_Matrix_Exp,
                                           C_Beta_F_Vector_Arg,
                                           C_Beta_F_Vector_Exp,
                                           Smn_hw_CosThetai_x_CosThetas_vec_Arg,
                                           Smn_hw_CosThetai_x_CosThetas_vec_Exp):
    
    # Create a matrix with all elements 10. we want to have the 10 ** numbers
    Matrix10_Exps = (10 + 0*C_Beta_D_Matrix_Exp) ** C_Beta_D_Matrix_Exp
    # Multiply the matrices element-wise
    C_Beta_D_Matrix = np.multiply(C_Beta_D_Matrix_Arg , Matrix10_Exps)
    
    
    Vector10=10 + 0*C_Beta_F_Vector_Exp
    # Multiply the vectors element-wise
    C_Beta_F_Vector = np.multiply(C_Beta_F_Vector_Arg , (Vector10**C_Beta_F_Vector_Exp))
#   # ----------------------------------------------------------------

    A_mn_vec = solver.solve(C_Beta_D_Matrix.T, C_Beta_F_Vector.T)
    #A_mn_vec = gmresPrecond(C_Beta_D_Matrix.T, C_Beta_F_Vector.T, tol=1e-5)
    A_mn_vec = A_mn_vec.T
    #A_mn_vec = func_Solve_Ax_b(C_Beta_D_Matrix, C_Beta_F_Vector)

    Vec10 = 10 + 0 * Smn_hw_CosThetai_x_CosThetas_vec_Exp
    Smn_hw_CosThetai_x_CosThetas_vec = Smn_hw_CosThetai_x_CosThetas_vec_Arg * (Vec10**Smn_hw_CosThetai_x_CosThetas_vec_Exp)
    fb_m=(2/(1j*kw))*A_mn_vec@np.transpose(Smn_hw_CosThetai_x_CosThetas_vec)
#             
    return fb_m

def Calculate_fbm(content_Rad_SWF_hs, 
                            content_dr_values_hs, 
                           content_Rad_SWF_hw, 
                           content_dr_values_hw,
                           Theta_i_deg,
                           _m, 
                           _N_order_Fortran,
                           _N_order):
    
    N_col_start = 0
    N_col_end = _N_order
    N_row_start = 0
    N_row_end = _N_order
    
    [C_Beta_D_Matrix_Arg, 
     C_Beta_D_Matrix_Exp,
     C_Beta_F_Mat_Arg, 
     C_Beta_F_Mat_Exp,
     Smn_hw_CosThetai_x_CosThetas_vec_Arg, 
     Smn_hw_CosThetai_x_CosThetas_vec_Exp] = func_Generate_Partition_Matrices(content_Rad_SWF_hs, content_dr_values_hs, 
                   content_Rad_SWF_hw, content_dr_values_hw,
                   Theta_i_deg,
                   _m, N_order_Fortran, 
                   N_col_start, N_col_end, N_row_start, N_row_end)
     
    C_Beta_F_Vector_Arg=np.zeros((1,C_Beta_F_Mat_Arg.shape[1]), dtype = 'complex')
    C_Beta_F_Vector_Exp=np.zeros((1,C_Beta_F_Mat_Arg.shape[1]), dtype = 'float')
        
    for ss_col in range(0,C_Beta_F_Mat_Arg.shape[1]):
        Summation_C_Beta_F=np.zeros((2),dtype='float') 
        for ss_row in range(0,C_Beta_F_Mat_Arg.shape[0]):
            Summation_C_Beta_F=Sum_ArgExp(Summation_C_Beta_F, [C_Beta_F_Mat_Arg[ss_row][ss_col], C_Beta_F_Mat_Exp[ss_row][ss_col]])     

        C_Beta_F_Vector_Arg[0][ss_col] = Summation_C_Beta_F[0]
        C_Beta_F_Vector_Exp[0][ss_col] = Summation_C_Beta_F[1]
    
    _f_bm=Estimate_fmn_from_CBetaD_Mat_CBetaFvec(C_Beta_D_Matrix_Arg,
                                          C_Beta_D_Matrix_Exp,
                                          C_Beta_F_Vector_Arg,
                                          C_Beta_F_Vector_Exp,
                                          Smn_hw_CosThetai_x_CosThetas_vec_Arg,
                                          Smn_hw_CosThetai_x_CosThetas_vec_Exp) 
    
    return [_f_bm]   
 


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#%%
#import sys
#sys.path.insert(0, '..')
#sys.path.insert(0, ParentDIR)

from funcs.FUNC_ReadFortranOutput import func_Rad_SWF, func_Ang_SWF, func_dr_values,\
     func_Smn_eta_c_from_dr, func_Smn_eta_c_from_dr_log, func_Rad_SWF_ArgExp, func_dr_values_ArgExp,\
     func_Smn_eta_c_from_dr_ArgExp, Multi_ArgExp, Sum_ArgExp, Get_ArgExp, func_Integrate_SmnEta1_SmlEta2_ArgExp

#sys.path.pop(0)

Amn_Vec=np.array([],dtype=complex)
Pmn_R_at1m_Vec=np.array([],dtype=complex) # ths is the scattered pressure at farfield (@1m) for m,n contribution


f_backscatter=[0]*len(freq_vec)
TS = f_backscatter

f_b=[]
for ii in range(0,np.size(freq_vec,0)): # Frequency Loop
    f=freq_vec[ii]
    w=2*np.pi*f
    kw=w/c_w
    ks=w/c_s
    hw=d*kw/2
    hs=d*ks/2
    
    N_at_m0_for_hs = 0.75 * hs + np.ceil(10-7*np.sin(Theta_i_deg*np.pi/180)) 
       
    m = -1
    if c_s < 500:
       M_order = 0.5 * (c_s/c_w) * (np.sin(Theta_i_deg*np.pi/180)**3) *  hs  + 8
    else: 
       M_order = 0.3 * (c_s/c_w) * (np.sin(Theta_i_deg*np.pi/180)**3) *  hs  + 6

    
    M_order = int(np.ceil(M_order))
    
  
    M_order_Fortran = M_order
    N_order_Fortran = int(np.ceil(N_at_m0_for_hs))
    
    # Write the Fortran's input file
    # hs:-------------------------------------------------------------------
    c_nondimensional=hs
    Theta_IncDeg=Theta_i_deg
    write_Inputfile(c_nondimensional,AspRatio, M_order_Fortran, N_order_Fortran ,Theta_IncDeg)
    
#    import os
#    parent_dir = os.path.split(os.getcwd())[0]
    script_path = ParentDIR+'/src/'+'Run_profcn_II_fort_FromPython.py'
    print(script_path)
    # Run Fortran:
    subprocess.run(['python', script_path], capture_output=True, text=True)
    #>>>>>>>>>
    
    
    
    content_Rad_SWF_hs=infunc_content_Rad_SWF()
#    content_Ang_SWF_hs=infunc_content_Ang_SWF()
    content_dr_values_hs=infunc_content_dr_values()
    # -------------------------------------------------------------------------
    
    
    # hw:-------------------------------------------------------------------
    c_nondimensional=hw
    Theta_IncDeg=Theta_i_deg
    write_Inputfile(c_nondimensional,AspRatio,M_order_Fortran,N_order_Fortran,Theta_IncDeg)
   
    # Run Fortran:
    subprocess.run(['python', script_path], capture_output=True, text=True)
    #>>>>>>>>>
    content_Rad_SWF_hw=infunc_content_Rad_SWF()
#    content_Ang_SWF_hw=infunc_content_Ang_SWF()
    content_dr_values_hw=infunc_content_dr_values()
    # -------------------------------------------------------------------------
    
    f_mn_contribution_VEC = []
    f_b_sum = 0
    last_f_bs=-100000
    CHECK_fbm_value = True
    
    
    while CHECK_fbm_value:
        m = m + 1   

        
        if hs > 60 :
            N_order = N_at_m0_for_hs - 0.033*(m**2) - (80/N_at_m0_for_hs)*(m)
        else: 
            N_order = N_at_m0_for_hs - 1*(m) 
            
        N_order = int(np.ceil(N_order))
          
        # Estimate fbm_1 for the initial selected N_order
        [f_bm] = Calculate_fbm(content_Rad_SWF_hs, 
                                    content_dr_values_hs, 
                                   content_Rad_SWF_hw, 
                                   content_dr_values_hw,
                                   Theta_i_deg,
                                   m, 
                                   N_order_Fortran,
                                   N_order)
        
#    for m in range(0,M_order):  # m index Loop: m=0,1,2,...,M
        print('m: ',str(m), ' of ',str(M_order), ', N: ', str(N_order))        
       

        f_b_sum=f_b_sum+f_bm
        
        if (np.abs(last_f_bs-f_bm) < 1E-5) or (m >= M_order_Fortran):
           CHECK_fbm_value = False
           
        last_f_bs = f_bm
        
    f_b=np.append(f_b,f_b_sum)
    freq_resp_writer.writerow(['{:.2f}'.format(f/1000), np.float64(20*np.log10(np.abs(f_b_sum)))])
    frf.flush()
    print('freq:',f,' of',max_freq,' TS: ',20*np.log10(np.abs(f_b_sum)))
    
#print(f_mn_contribution_VEC) 
    
#plt.plot(np.arange(0,len(f_mn_contribution_VEC),1), np.real(f_mn_contribution_VEC),marker='o',color=[0,0,1])  
#plt.plot(np.arange(0,len(f_mn_contribution_VEC),1), np.imag(f_mn_contribution_VEC),marker='<',color=[1,0,1])  
#
#f_mn_contribution_VEC_data = pd.DataFrame({'f_mn_term_real': np.real(f_mn_contribution_VEC), 'f_mn_term_imag':np.imag(f_mn_contribution_VEC)})
#f_mn_file = 'f_mn_contribution_ros_{}_f_{}_M_{}_N_{}_precond_NewN_Order.csv'.format(int(ro_s), int(min_freq / 1000), M_order, N_order)
#f_mn_contribution_VEC_data.to_csv(f_mn_file, index=False)


TS=20*np.log10(np.abs(f_b))

# results from COMSOL model for comarison 
#ParentDIR+'/ValidationCOMSOL/'

#COMSOL1 = pd.read_csv(ParentDIR+'/ValidationCOMSOL/'+'SB_Prol_B2cm_AspR4_Cont1p05_Theta5_38_300kHz.csv')
#COMSOL1 = pd.read_csv(ParentDIR+'/ValidationCOMSOL/'+'SB_Prol_B1cm_AspR2_ro10_c343_Theta15_38_300kHz.csv')
#COMSOL3 = pd.read_csv(ParentDIR+'/ValidationCOMSOL/'+'TS200kHz_Prol_B2cm_AspR4_Cont1p05_Theta90_400_420kHz.csv')
#COMSOL4 = pd.read_csv(ParentDIR+'/ValidationCOMSOL/'+'TS200kHz_Prol_B2cm_AspR4_Cont1p05_Theta90_580_620kHz.csv')

FIG=plt.figure(figsize=(14,7)) 
#plt.plot(COMSOL1.to_numpy()[:,0]/1000,COMSOL1.to_numpy()[:,1],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution')  
#plt.plot(COMSOL2[COMSOL2.columns[0]]/1000,COMSOL2[COMSOL2.columns[1]],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution') 
#plt.plot(COMSOL3[COMSOL3.columns[0]]/1000,COMSOL3[COMSOL3.columns[1]],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution') 
#plt.plot(COMSOL4[COMSOL4.columns[0]]/1000,COMSOL4[COMSOL4.columns[1]],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution') 
plt.plot(freq_vec[0:len(TS)]/1000, TS, marker='o',markersize=4, color=[1,0,0])

ts_data = pd.DataFrame({'Freq_kHz': freq_vec[0:len(TS)]/1000, 'TS':TS})
ts_file = 'ts_f_a_{}_b_{}_f1_{}_f2_{}_M_{}_N_{}_rhos_{:.2f}_IncAngle_{}.csv'.format(a, b, int(min_freq / 1000),
                        int(freq_vec[ii-1] / 1000), M_order, N_order, ro_s, Theta_i_deg)
ts_data.to_csv(ParentDIR+'/temp/'+ts_file, index=False)
#


#print('total compute time = {}s'.format(time.time() - t0))
