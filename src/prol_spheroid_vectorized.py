import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import csv
from scipy.special import lpmn


from funcs.FUNC_read_fortran_output_vectorized import func_Rad_SWF_ArgExp, func_dr_values_ArgExp, \
    func_Smn_eta_c_from_dr_ArgExp, Multi_ArgExp, Sum_ArgExp, Get_ArgExp, func_Integrate_SmnEta1_SmlEta2_ArgExp, \
    func_Rad_SWF_ArgExp_vectorized, func_dr_values_ArgExp_vectorized, func_Smn_eta_c_from_dr_ArgExp_vectorized, func_Integrate_SmnEta1_SmlEta2_ArgExp_vectorized,\
    Get_ArgExp_array, Multi_ArgExp_array, Sum_ArgExp_array, Sum_ArgExp_over_rows

# http://mathworld.wolfram.com/SphericalHankelFunctionoftheFirstKind.html
# Spherical Hankel Function of the First Kind
# The spherical Hankel function of the first kind h1_n(z) is defined by:
#    h1_n(z)=j_n(z)+i*y_n(z)
# where j_n(z) and y_n(z) are the spherical Bessel functions of the first and second kinds.


class ProlateSpheroid:

    def __init__(self, settings, solver):

        self.prefix = settings.prefix
        self.ro_w = settings.ro_w
        self.ro_s = settings.ro_s
        self.c_w = settings.c_w
        self.c_s = settings.c_s
        self.a = settings.a
        self.b = settings.b
        self.d = 2*((self.a*self.a-self.b*self.b)**0.5)
        self.AspRatio = self.a / self.b
        delta_f = settings.delta_f
        self.Theta_i_deg = settings.theta_i_deg

        self.min_freq = settings.min_freq
        self.max_freq = settings.max_freq
        self.freq_vec=np.arange(self.min_freq,self.max_freq,delta_f)

        self.solver = solver
        self.precision_fbs = settings.precision_fbs


    def run(self, ts_file_name):
        ParentDIR=os.path.split(os.getcwd())[0]
        freq_resp_file = os.path.join(ParentDIR, 'temp', ts_file_name)
        frf = open(freq_resp_file, 'wt', newline='', encoding='utf-8')
        freq_resp_writer = csv.writer(frf, delimiter=',')
        freq_resp_writer.writerow(['Freq_kHz', 'TS'])

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

        script_path1 = os.path.join(ParentDIR, 'src', 'Run_param_fort_FromPython.py')
        print(script_path1)
        # Run Python Script:
        subprocess.run(['python3', script_path1], capture_output=True, text=True)
        #>>>>>>>>>
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #%%

        f_b=[]
        for ii in range(0,np.size(self.freq_vec,0)): # Frequency Loop
            f=self.freq_vec[ii]
            w=2*np.pi*f
            kw=w/self.c_w
            ks=w/self.c_s
            hw=self.d*kw/2
            hs=self.d*ks/2

            N_at_m0_for_hs = np.ceil(0.75 * hs + 8 - 5*np.sin(self.Theta_i_deg*np.pi/180))
            if self.precision_fbs < 1e-5:
                N_at_m0_for_hs = np.ceil(0.75 * hs + 12 - 7*np.sin(self.Theta_i_deg*np.pi/180))


            m = -1
            if self.c_s < 500:
                M_order = 0.5 * (self.c_s/self.c_w) * (np.sin(self.Theta_i_deg*np.pi/180)**3) *  hs  + 8
            else:
                M_order = 0.3 * (self.c_s/self.c_w) * (np.sin(self.Theta_i_deg*np.pi/180)**3) *  hs  + 6

            M_order = int(np.ceil(M_order))

            M_order_Fortran = M_order
            N_order_Fortran = int(np.ceil(N_at_m0_for_hs))

            # Write the Fortran's input file
            # hs:-------------------------------------------------------------------
            c_nondimensional=hs
            Theta_IncDeg=self.Theta_i_deg
            write_Inputfile(c_nondimensional,self.AspRatio, M_order_Fortran, N_order_Fortran ,Theta_IncDeg, ParentDIR)


            #    import os
            #    parent_dir = os.path.split(os.getcwd())[0]
            
            
            script_path = os.path.join(ParentDIR, 'src', 'Run_profcn_II_fort_FromPython.py')
            
            
            #print(script_path)
            # Run Fortran:
            subprocess.run(['python3', script_path], capture_output=True, text=True)
            #>>>>>>>>>

            content_Rad_SWF_hs=infunc_content_Rad_SWF(ParentDIR)
            #    content_Ang_SWF_hs=infunc_content_Ang_SWF()
            content_dr_values_hs=infunc_content_dr_values(ParentDIR)
            # -------------------------------------------------------------------------

            # hw:-------------------------------------------------------------------
            c_nondimensional=hw
            Theta_IncDeg=self.Theta_i_deg
            write_Inputfile(c_nondimensional,self.AspRatio,M_order_Fortran,N_order_Fortran,Theta_IncDeg, ParentDIR)
            



            # Run Fortran:
            subprocess.run(['python3', script_path], capture_output=True, text=True)
            #>>>>>>>>>
            content_Rad_SWF_hw=infunc_content_Rad_SWF(ParentDIR)
            #    content_Ang_SWF_hw=infunc_content_Ang_SWF()
            content_dr_values_hw=infunc_content_dr_values(ParentDIR)
            # -------------------------------------------------------------------------

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
                [f_bm] = Calculate_fbm_vectorized(content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw,
                        content_dr_values_hw, self.Theta_i_deg, m, N_order_Fortran, N_order, self.solver,
                        self.ro_s, self.ro_w, kw)
                
                # Fallback if vectorized calculation fails
                if np.array_equal(f_bm, np.array([[0.+0.j]])):
                    print("This is f_bm", f_bm)
                    print("f_bm is zero, solvers failed, using non vectorized code")
                    [f_bm] = Calculate_fbm(content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw,
                        content_dr_values_hw, self.Theta_i_deg, m, N_order_Fortran, N_order, self.solver,
                        self.ro_s, self.ro_w, kw)

                #    for m in range(0,M_order):  # m index Loop: m=0,1,2,...,M
                print('m: ',str(m), ' of ',str(M_order), ', N: ', str(N_order))

                f_b_sum=f_b_sum+f_bm

                if m==0:
                    f_b_max = np.abs(f_b_sum)
                else:
                    f_b_max = np.max([np.abs(f_b_sum), np.abs(f_b_sum)])
                
#                if (np.abs(last_f_bs-f_bm) < self.precision_fbs) or (m >= (M_order_Fortran-1)):
                if (np.abs(last_f_bs-f_bm) < f_b_max*(1E-2)) or (m >= (M_order_Fortran-1)):
                    CHECK_fbm_value = False

                last_f_bs = f_bm

            f_b=np.append(f_b,f_b_sum)
            freq_resp_writer.writerow(['{:.2f}'.format(f/1000), np.float64(20*np.log10(np.abs(f_b_sum)))])
            frf.flush()

            print('freq:',f,' of',self.max_freq,' TS: ',20*np.log10(np.abs(f_b_sum)))
            


        TS=20*np.log10(np.abs(f_b))


        # TS plot
        plt.figure(figsize=(14,7))
        #plt.plot(COMSOL1.to_numpy()[:,0]/1000,COMSOL1.to_numpy()[:,1],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution')
        #plt.plot(COMSOL2[COMSOL2.columns[0]]/1000,COMSOL2[COMSOL2.columns[1]],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution')
        #plt.plot(COMSOL3[COMSOL3.columns[0]]/1000,COMSOL3[COMSOL3.columns[1]],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution')
        #plt.plot(COMSOL4[COMSOL4.columns[0]]/1000,COMSOL4[COMSOL4.columns[1]],color='k',dashes=[3,2],linewidth=2.5,label='Comsol Solution')
        plt.plot(self.freq_vec[0:len(TS)]/1000, TS, marker='o',markersize=4, color=[1,0,0])
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('Target Strength [dB re 1m^2]')
        plt.show()
        

    def compute_far_field_pattern(self, freq_Hz, num_angles_per_half_plane):
        ParentDIR=os.path.split(os.getcwd())[0]
        script_path1 = ParentDIR+'/src/'+'Run_param_fort_FromPython.py'
        print(script_path1)
        # Run Python Script:
        subprocess.run(['python3', script_path1], capture_output=True, text=True)

        f = freq_Hz

        w=2*np.pi*f
        kw=w/self.c_w
        ks=w/self.c_s
        hw=self.d*kw/2
        hs=self.d*ks/2

        N_at_m0_for_hs = np.ceil(0.75 * hs + 8 - 3*np.sin(self.Theta_i_deg*np.pi/180))
        if self.precision_fbs < 1e-5:
            N_at_m0_for_hs = np.ceil(0.75 * hs + 12 - 7*np.sin(self.Theta_i_deg*np.pi/180))

        if self.c_s < 500:
            M_order = 0.5 * (self.c_s/self.c_w) * (np.sin(self.Theta_i_deg*np.pi/180)**3) *  hs  + 8
        else:
            M_order = 0.3 * (self.c_s/self.c_w) * (np.sin(self.Theta_i_deg*np.pi/180)**3) *  hs  + 6

        M_order = int(np.ceil(M_order))


        M_order_Fortran = M_order
        N_order_Fortran = int(np.ceil(N_at_m0_for_hs))

        # Write the Fortran's input file
        # hs:-------------------------------------------------------------------
        c_nondimensional=hs
        Theta_IncDeg=self.Theta_i_deg
        write_Inputfile(c_nondimensional,self.AspRatio, M_order_Fortran, N_order_Fortran ,Theta_IncDeg, ParentDIR)

        #    import os
        #    parent_dir = os.path.split(os.getcwd())[0]
        script_path = ParentDIR+'/src/'+'Run_profcn_II_fort_FromPython.py'
        print(script_path)
        # Run Fortran:
        subprocess.run(['python3', script_path], capture_output=True, text=True)
        #>>>>>>>>>

        content_Rad_SWF_hs=infunc_content_Rad_SWF(ParentDIR)
        #    content_Ang_SWF_hs=infunc_content_Ang_SWF()
        content_dr_values_hs=infunc_content_dr_values(ParentDIR)
        # -------------------------------------------------------------------------

        # hw:-------------------------------------------------------------------
        c_nondimensional=hw
        Theta_IncDeg=self.Theta_i_deg
        write_Inputfile(c_nondimensional,self.AspRatio,M_order_Fortran,N_order_Fortran,Theta_IncDeg, ParentDIR)

        # Run Fortran:
        subprocess.run(['python3', script_path], capture_output=True, text=True)
        #>>>>>>>>>
        content_Rad_SWF_hw=infunc_content_Rad_SWF(ParentDIR)
        #    content_Ang_SWF_hw=infunc_content_Ang_SWF()
        content_dr_values_hw=infunc_content_dr_values(ParentDIR)
        # -------------------------------------------------------------------------
        m = -1
        CHECK_fbm_value = True
        angles = np.zeros(num_angles_per_half_plane * 2, dtype=np.float64)
        pattern = np.zeros(num_angles_per_half_plane * 2, dtype=np.complex128)
        last_f_bs = -100000 * np.ones(num_angles_per_half_plane * 2)
        while CHECK_fbm_value:
            m = m + 1

            if hs > 60 :
                N_order = N_at_m0_for_hs - 0.033*(m**2) - (80/N_at_m0_for_hs)*(m)
            else:
                N_order = N_at_m0_for_hs - 1*(m)

            N_order = int(np.ceil(N_order))
            

            angles_m, pattern_m = calculate_fbs(num_angles_per_half_plane, content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw,
                                                content_dr_values_hw, self.Theta_i_deg, m, N_order_Fortran, N_order, self.solver,
                                                self.ro_s, self.ro_w, kw)
            pattern += pattern_m
            angles = angles_m

            print('m: ',str(m), ' of ',str(M_order), ', N: ', str(N_order))

            if np.max(np.abs(last_f_bs-pattern_m)) < self.precision_fbs or m >= M_order_Fortran-1:
                CHECK_fbm_value = False

            last_f_bs = pattern_m

        return angles, np.absolute(pattern)


def write_Inputfile(c_nondimensional,AspectRatio,M_order,N_order,Theta_IncDeg, _parent_dir):
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

    file_name = os.path.join(_parent_dir, 'src', 'profcn.dat')
    
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
def infunc_content_Rad_SWF(_parent_dir):

    fort20File=os.path.join(_parent_dir, 'src', 'fort.20')
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
    
    return np.array(content_Rad_SWF)

    
def infunc_content_dr_values(_parent_dir):

    fort70File=os.path.join(_parent_dir, 'src', 'fort.70')
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
             
    return np.array(content_dr_values)    

#%%

def func_Generate_Partition_Matrices(content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw, content_dr_values_hw,
                                     Theta_i_deg, _m, N_order_Fortran, _N_col_start, _N_col_end, _N_row_start,
                                     _N_row_end, ro_s, ro_w):
    
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
        
        for _n in range(_m + _N_row_start , _N_row_end + _m):  # n index Loop: n=m,1,2,...,N : This fills the rows in Matrices
            nn=nn+1
            ## PSWF values ==================================================================================================
#                Smn_hw_eta_is=func_Ang_SWF(content_Ang_SWF_hw, m, n , N_order) #func_Adelman_Smn_Const_Eta_fof_c(DirPSWF, Theta_i_deg, m, n, hw, Prec)  # Prolate spheroidal angular function of the first kind >> S(m,n,hw,cos(Theta_i))
#                Smn_hw_eta_i=Smn_hw_eta_is[0]
#                Smn_hw_eta_s=Smn_hw_eta_is[1]
#                Nmn_hw=func_Adelman_Nmn_of_c(DirPSWF, m, n, hw, Prec)
            Nmn_hw=1
            dmn_r_hw_ArgExp=func_dr_values_ArgExp(content_dr_values_hw, _m, _n , N_order_Fortran)# func_Adelman_dr_mn_of_c(DirPSWF, m, n, hw, Prec)
            
            R_mn_hw_kisi0=func_Rad_SWF_ArgExp(content_Rad_SWF_hw, _m, _n , N_order_Fortran) #func_Adelman_Rmn(DirPSWF, AspRatio, m, n, hw, Prec) # Prolate spheroidal radial function of the first kind >>  R1(m,n,hw,kisi0)
          
            eta_i=np.cos(np.pi*Theta_i_deg/180)
            Nr = len(dmn_r_hw_ArgExp)
            Pmn_eta_i = lpmn(_m, _n + Nr - 1, eta_i)[0][_m]
            Smn_hw_eta_i=func_Smn_eta_c_from_dr_ArgExp(_m, _n, eta_i, dmn_r_hw_ArgExp, Pmn_eta_i)
            
            
            eta_s=np.cos(np.pi*Theta_s_deg/180)
            Nr = len(dmn_r_hw_ArgExp)
            Pmn_eta_s = lpmn(_m, _n + Nr - 1, eta_s)[0][_m]
            Smn_hw_eta_s=func_Smn_eta_c_from_dr_ArgExp(_m, _n, eta_s, dmn_r_hw_ArgExp, Pmn_eta_s)
            
            R1_mn_hw_kisi0 = R_mn_hw_kisi0[0] # [Arg, Exp]
            dR1_mn_hw_kisi0 = R_mn_hw_kisi0[1]
            R2_mn_hw_kisi0 = R_mn_hw_kisi0[2]
            dR2_mn_hw_kisi0 = R_mn_hw_kisi0[3]
            R3_mn_hw_kisi0 = Sum_ArgExp(R1_mn_hw_kisi0, Multi_ArgExp([1j,0], R2_mn_hw_kisi0))
            dR3_mn_hw_kisi0 = Sum_ArgExp(dR1_mn_hw_kisi0, Multi_ArgExp([1j,0], dR2_mn_hw_kisi0))
            ## ===============================================================================================================

               
            # Calculate Coefficient ------------------------------------------
            C_n_m=Multi_ArgExp(Get_ArgExp((1j**_n)*Em),Smn_hw_eta_i) #(1j**n)*Em*Smn_hw_eta_i/Nmn_hw
#                 D_nl_m = (ro_s/ro_w)*dR3_mn_hw_kisi0*R1_ml_hs_kisi0 - R3_mn_hw_kisi0*dR1_ml_hs_kisi0
            D_nl_m = Multi_ArgExp(Get_ArgExp(ro_s/ro_w), Multi_ArgExp(dR3_mn_hw_kisi0, R1_ml_hs_kisi0))
            D_nl_m = Sum_ArgExp(D_nl_m, Multi_ArgExp([-1,0], Multi_ArgExp(R3_mn_hw_kisi0, dR1_ml_hs_kisi0)))
            
#                F_nl_m=(ro_s/ro_w)*dR1_mn_hw_kisi0*R1_ml_hs_kisi0 - R1_mn_hw_kisi0*dR1_ml_hs_kisi0
            F_nl_m = Multi_ArgExp(Get_ArgExp(ro_s/ro_w), Multi_ArgExp(dR1_mn_hw_kisi0, R1_ml_hs_kisi0 )) 
            F_nl_m = Sum_ArgExp(F_nl_m, Multi_ArgExp([-1,0], Multi_ArgExp(R1_mn_hw_kisi0, dR1_ml_hs_kisi0))) 
            
            Beta_nl_m=func_Integrate_SmnEta1_SmlEta2_ArgExp(_m, _n, l, dmn_r_hw_ArgExp, dml_r_hs_ArgExp)


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
                Smn_hw_eta_s_X_cosmpi=Multi_ArgExp(Smn_hw_eta_s, Get_ArgExp(np.cos(_m*np.pi)))
                
                TEMPvar=Multi_ArgExp(Em_x_Smn_hw_eta_i, Smn_hw_eta_s_X_cosmpi)
                Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg[0][nn]=TEMPvar[0]
                Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp[0][nn]=TEMPvar[1]
    
    if _m == 0:
        print("\nNon-vectorized matrix stats:")
        print("First few matrix values:")
        print("C_Beta_D_Matrix_Arg[0:3, 0:3]:\n", Part_C_Beta_D_Matrix_Arg[0:3, 0:3])
        print("C_Beta_D_Matrix_Exp[0:3, 0:3]:\n", Part_C_Beta_D_Matrix_Exp[0:3, 0:3])
    return [Part_C_Beta_D_Matrix_Arg,
            Part_C_Beta_D_Matrix_Exp,
            Part_C_Beta_F_Mat_Arg,
            Part_C_Beta_F_Mat_Exp,
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg,
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp] 

def func_Generate_Partition_Matrices_vectorized(content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw, content_dr_values_hw,
                                                Theta_i_deg, m, N_order_Fortran, N_col_start, N_col_end, N_row_start,
                                                N_row_end, ro_s, ro_w):
    Theta_s_deg = 180 - Theta_i_deg
    
    if m == 0:
       Em=1
    else:
       Em=2
    
    # Create arrays for l and n
    l_array = np.arange(m + N_col_start, N_col_end + m)
    n_array = np.arange(m + N_row_start, N_row_end + m)

    num_rows = N_row_end - N_row_start
    num_cols = N_col_end - N_col_start

    # Prepare matrices
    if num_rows < 0 or num_cols < 0:
        num_rows = 0
        num_cols = 0
    
    
    Part_C_Beta_D_Matrix_Arg = np.zeros((num_rows, num_cols), dtype='complex')
    Part_C_Beta_D_Matrix_Exp = np.zeros((num_rows, num_cols), dtype='float')
    Part_C_Beta_F_Mat_Arg = np.zeros((num_rows, num_cols), dtype='complex')
    Part_C_Beta_F_Mat_Exp = np.zeros((num_rows, num_cols), dtype='float')

    Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg = np.zeros((1, num_rows), dtype='complex')
    Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp = np.zeros((1, num_rows), dtype='float')

    # Check if l_array or n_array are empty
    if len(l_array) == 0 or len(n_array) == 0:
        # Return empty matrices this is only done for m that creates empty arrays
        # This is the same as in the non vectorized version where when N_order is less than or equal to m, the loops over l and n don't execute any iterations. 
        return [Part_C_Beta_D_Matrix_Arg,
                Part_C_Beta_D_Matrix_Exp,
                Part_C_Beta_F_Mat_Arg,
                Part_C_Beta_F_Mat_Exp,
                Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg,
                Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp]


    # Precompute values for l
    R1_ml_hs_kisi0_array, dR1_ml_hs_kisi0_array, _, _ = func_Rad_SWF_ArgExp_vectorized(content_Rad_SWF_hs, m, l_array, N_order_Fortran)
    dml_r_hs_ArgExp_list = func_dr_values_ArgExp_vectorized(content_dr_values_hs, m, l_array, N_order_Fortran)
    
    # Precompute values for n
    R1_mn_hw_kisi0_array, dR1_mn_hw_kisi0_array, R2_mn_hw_kisi0_array, dR2_mn_hw_kisi0_array = func_Rad_SWF_ArgExp_vectorized(content_Rad_SWF_hw, m, n_array, N_order_Fortran)
    dmn_r_hw_ArgExp_list = func_dr_values_ArgExp_vectorized(content_dr_values_hw, m, n_array, N_order_Fortran)
    
    eta_i = np.cos(np.pi * Theta_i_deg / 180)
    eta_s = np.cos(np.pi * Theta_s_deg / 180)
    
    
    Smn_hw_eta_i_array = func_Smn_eta_c_from_dr_ArgExp_vectorized(m, n_array, eta_i, dmn_r_hw_ArgExp_list)
    Smn_hw_eta_s_array = func_Smn_eta_c_from_dr_ArgExp_vectorized(m, n_array, eta_s, dmn_r_hw_ArgExp_list)
    
    
    R3_mn_hw_kisi0_array = Sum_ArgExp_array(R1_mn_hw_kisi0_array, Multi_ArgExp_array(np.array([1j, 0]), R2_mn_hw_kisi0_array))
    dR3_mn_hw_kisi0_array = Sum_ArgExp_array(dR1_mn_hw_kisi0_array, Multi_ArgExp_array(np.array([1j, 0]), dR2_mn_hw_kisi0_array))
    

    
    arg_e_ro = Get_ArgExp(ro_s / ro_w)
    arg_e_m_pi = Get_ArgExp(np.cos(m * np.pi))
    
    # Vectorized over n and l
    for ll, (l, R1_ml_hs_kisi0, dR1_ml_hs_kisi0, dml_r_hs_ArgExp) in enumerate(zip(l_array, R1_ml_hs_kisi0_array, dR1_ml_hs_kisi0_array, dml_r_hs_ArgExp_list)):
        # For each l, process all n
        # Extract dml_r_hs_ArgExp once per l
        # Prepare arrays for current l
        R1_ml_hs_kisi0 = np.array(R1_ml_hs_kisi0)
        dR1_ml_hs_kisi0 = np.array(dR1_ml_hs_kisi0)
        dml_r_hs_ArgExp = np.array(dml_r_hs_ArgExp)
        
        # Replicate R1_ml_hs_kisi0 and dR1_ml_hs_kisi0 to match n_array size
        R1_ml_hs_kisi0_repl = np.tile(R1_ml_hs_kisi0, (len(n_array), 1))
        dR1_ml_hs_kisi0_repl = np.tile(dR1_ml_hs_kisi0, (len(n_array), 1))
        dml_r_hs_ArgExp_repl = np.tile(dml_r_hs_ArgExp[np.newaxis, ...], (len(n_array), 1, 1))
             
        Beta_nl_m_array = np.array([
        func_Integrate_SmnEta1_SmlEta2_ArgExp_vectorized(m, n, l, dmn_r_hw_ArgExp, dml_r_hs_ArgExp)
        for n, dmn_r_hw_ArgExp in zip(n_array, dmn_r_hw_ArgExp_list)])


        
        D_nl_m_array = Multi_ArgExp_array(arg_e_ro, Multi_ArgExp_array(dR3_mn_hw_kisi0_array, R1_ml_hs_kisi0_repl))
        D_nl_m_array = Sum_ArgExp_array(D_nl_m_array, Multi_ArgExp_array([-1, 0], Multi_ArgExp_array(R3_mn_hw_kisi0_array, dR1_ml_hs_kisi0_repl)))
        
  
        
        F_nl_m_array = Multi_ArgExp_array(arg_e_ro, Multi_ArgExp_array(dR1_mn_hw_kisi0_array, R1_ml_hs_kisi0_repl))
        F_nl_m_array = Sum_ArgExp_array(F_nl_m_array, Multi_ArgExp_array([-1, 0], Multi_ArgExp_array(R1_mn_hw_kisi0_array, dR1_ml_hs_kisi0_repl)))
        
        
        Em_array = Em

        n_powers = 1j ** n_array
        C_n_m_array = Multi_ArgExp_array(Get_ArgExp_array(n_powers * Em_array), Smn_hw_eta_i_array)
        
        # Compute Part_C_Beta_D_Matrix
        TEMPvar = Multi_ArgExp_array(C_n_m_array, Multi_ArgExp_array(Beta_nl_m_array, D_nl_m_array))

        Part_C_Beta_D_Matrix_Arg[:, ll] = TEMPvar[:, 0]
        Part_C_Beta_D_Matrix_Exp[:, ll] = TEMPvar[:, 1].real

        # Compute Part_C_Beta_F_Mat 
        Minus_C_n_m_array = Multi_ArgExp_array([-1, 0], C_n_m_array)
        Beta_nl_m_x_F_nl_m_array = Multi_ArgExp_array(Beta_nl_m_array, F_nl_m_array)
        TEMPvar = Multi_ArgExp_array(Minus_C_n_m_array, Beta_nl_m_x_F_nl_m_array)
        

        Part_C_Beta_F_Mat_Arg[:, ll] = TEMPvar[:, 0]
        Part_C_Beta_F_Mat_Exp[:, ll] = TEMPvar[:, 1].real
        

        if ll == 0:            
            Em_arg_exp = Get_ArgExp(Em)
            Em_arg_exp_array = np.tile(Em_arg_exp, (len(n_array), 1))
            Em_x_Smn_hw_eta_i_array = Multi_ArgExp_array(Em_arg_exp_array, Smn_hw_eta_i_array)
                       
            arg_e_m_pi_array = np.tile(arg_e_m_pi, (len(n_array), 1))
            Smn_hw_eta_s_X_cosmpi_array = Multi_ArgExp_array(Smn_hw_eta_s_array, arg_e_m_pi_array)
                        
            TEMPvar_array = Multi_ArgExp_array(Em_x_Smn_hw_eta_i_array, Smn_hw_eta_s_X_cosmpi_array)
            

            Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg[0, :] = TEMPvar_array[:, 0]
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp[0, :] = TEMPvar_array[:, 1].real
    
    return [Part_C_Beta_D_Matrix_Arg,
            Part_C_Beta_D_Matrix_Exp,
            Part_C_Beta_F_Mat_Arg,
            Part_C_Beta_F_Mat_Exp,
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Arg,
            Part_Smn_hw_CosThetai_x_CosThetas_vec_Exp]


def solve_amn(C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp, C_Beta_F_Vector_Arg,
              C_Beta_F_Vector_Exp, solver, m, N_order):
    # Create matrix of 10^exponents
    Matrix10_Exps = (10 + 0*C_Beta_D_Matrix_Exp) ** C_Beta_D_Matrix_Exp
    C_Beta_D_Matrix = np.multiply(C_Beta_D_Matrix_Arg, Matrix10_Exps)

    Vector10 = 10 + 0*C_Beta_F_Vector_Exp
    C_Beta_F_Vector = np.multiply(C_Beta_F_Vector_Arg, (Vector10**C_Beta_F_Vector_Exp))

    # Try direct solve first
    try:
        A_mn_vec = solver.solve(C_Beta_D_Matrix.T, C_Beta_F_Vector.T)
    except RuntimeError as e:
        # If direct solve fails, try numpy's lstsq which is more stable for ill-conditioned matrices
        print(f"LU decomposition failed, trying least squares for m={m}, N_order={N_order}")
        try:
            A_mn_vec = np.linalg.lstsq(C_Beta_D_Matrix.T, C_Beta_F_Vector.T, rcond=None)[0]
            print("Least squares solution succeeded")
        except np.linalg.LinAlgError:
            print(f"All solution attempts failed for m={m}, N_order={N_order}")
            A_mn_vec = np.zeros_like(C_Beta_F_Vector.T)

    return A_mn_vec.T




def Estimate_fmn_from_CBetaD_Mat_CBetaFvec(C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp, C_Beta_F_Vector_Arg,
                                           C_Beta_F_Vector_Exp, Smn_hw_CosThetai_x_CosThetas_vec_Arg,
                                           Smn_hw_CosThetai_x_CosThetas_vec_Exp, solver, kw,  m, N_order):
    
    if C_Beta_D_Matrix_Arg.size == 0 or C_Beta_F_Vector_Arg.size == 0:
        # Return fb_m as zero
        return 0.0
    
    A_mn_vec = solve_amn(C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp, C_Beta_F_Vector_Arg,
                         C_Beta_F_Vector_Exp, solver, m, N_order)

    Vec10 = 10 + 0 * Smn_hw_CosThetai_x_CosThetas_vec_Exp
    Smn_hw_CosThetai_x_CosThetas_vec = Smn_hw_CosThetai_x_CosThetas_vec_Arg * (Vec10**Smn_hw_CosThetai_x_CosThetas_vec_Exp)
    fb_m=(2/(1j*kw))*A_mn_vec@np.transpose(Smn_hw_CosThetai_x_CosThetas_vec)

    return fb_m



def Calculate_fbm_vectorized(content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw, content_dr_values_hw, Theta_i_deg, _m,
                  _N_order_Fortran, _N_order, solver, ro_s, ro_w, kw):

    N_col_start = 0
    N_col_end = _N_order
    N_row_start = 0
    N_row_end = _N_order



    [C_Beta_D_Matrix_Arg,
     C_Beta_D_Matrix_Exp,
     C_Beta_F_Mat_Arg,
     C_Beta_F_Mat_Exp,
     Smn_hw_CosThetai_x_CosThetas_vec_Arg,
     Smn_hw_CosThetai_x_CosThetas_vec_Exp] = func_Generate_Partition_Matrices_vectorized(
        content_Rad_SWF_hs, content_dr_values_hs,
        content_Rad_SWF_hw, content_dr_values_hw,
        Theta_i_deg, _m, _N_order_Fortran,
        N_col_start, N_col_end, N_row_start,
        N_row_end, ro_s, ro_w)
    

    # Check if matrices are empty
    if C_Beta_D_Matrix_Arg.size == 0 or C_Beta_F_Mat_Arg.size == 0:
        # Return fbm as zero for this m since this is for the case where we return empty arrays for this one m.
        return [0.0]

    ######### Vectorized this aswell from the old Calculate_fbm ########
    # Stack the matrices into AE_matrix
    AE_matrix = np.stack((C_Beta_F_Mat_Arg, C_Beta_F_Mat_Exp), axis=2)

    # Sum over rows
    Argu_Number, Expon = Sum_ArgExp_over_rows(AE_matrix)

    # Prepare the result arrays
    C_Beta_F_Vector_Arg = Argu_Number[np.newaxis, :]
    C_Beta_F_Vector_Exp = Expon[np.newaxis, :]

    
    _f_bm = Estimate_fmn_from_CBetaD_Mat_CBetaFvec(
        C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp,
        C_Beta_F_Vector_Arg, C_Beta_F_Vector_Exp,
        Smn_hw_CosThetai_x_CosThetas_vec_Arg,
        Smn_hw_CosThetai_x_CosThetas_vec_Exp, solver, kw, _m, _N_order)


    return [_f_bm]


def Calculate_fbm(content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw, content_dr_values_hw, Theta_i_deg, _m,
                  _N_order_Fortran, _N_order, solver, ro_s, ro_w, kw):
    
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
                                                                              Theta_i_deg, _m, _N_order_Fortran,
                                                                              N_col_start, N_col_end, N_row_start,
                                                                              N_row_end, ro_s, ro_w)
     
    C_Beta_F_Vector_Arg=np.zeros((1,C_Beta_F_Mat_Arg.shape[1]), dtype = 'complex')
    C_Beta_F_Vector_Exp=np.zeros((1,C_Beta_F_Mat_Arg.shape[1]), dtype = 'float')
        
    for ss_col in range(0,C_Beta_F_Mat_Arg.shape[1]):
        Summation_C_Beta_F=np.zeros((2),dtype='float') 
        for ss_row in range(0,C_Beta_F_Mat_Arg.shape[0]):
            Summation_C_Beta_F=Sum_ArgExp(Summation_C_Beta_F, [C_Beta_F_Mat_Arg[ss_row][ss_col], C_Beta_F_Mat_Exp[ss_row][ss_col]])     

        C_Beta_F_Vector_Arg[0][ss_col] = Summation_C_Beta_F[0]
        C_Beta_F_Vector_Exp[0][ss_col] = Summation_C_Beta_F[1]
    
    _f_bm= Estimate_fmn_from_CBetaD_Mat_CBetaFvec(C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp, C_Beta_F_Vector_Arg,
                                                  C_Beta_F_Vector_Exp, Smn_hw_CosThetai_x_CosThetas_vec_Arg,
                                                  Smn_hw_CosThetai_x_CosThetas_vec_Exp, solver, kw, _m, _N_order)
    
    return [_f_bm]

def calculate_fbs(num_angles_per_half_plane, content_Rad_SWF_hs, content_dr_values_hs, content_Rad_SWF_hw, content_dr_values_hw, Theta_i_deg, m,
                  N_order_Fortran, N_order, solver, ro_s, ro_w, kw):
    N_col_start = 0
    N_col_end = N_order
    N_row_start = 0
    N_row_end = N_order

    [C_Beta_D_Matrix_Arg,
     C_Beta_D_Matrix_Exp,
     C_Beta_F_Mat_Arg,
     C_Beta_F_Mat_Exp,
     _,
     _] = func_Generate_Partition_Matrices(content_Rad_SWF_hs, content_dr_values_hs,
                                                                              content_Rad_SWF_hw, content_dr_values_hw,
                                                                              Theta_i_deg, m, N_order_Fortran,
                                                                              N_col_start, N_col_end, N_row_start,
                                                                              N_row_end, ro_s, ro_w)

    C_Beta_F_Vector_Arg=np.zeros((1,C_Beta_F_Mat_Arg.shape[1]), dtype = 'complex')
    C_Beta_F_Vector_Exp=np.zeros((1,C_Beta_F_Mat_Arg.shape[1]), dtype = 'float')

    for ss_col in range(0,C_Beta_F_Mat_Arg.shape[1]):
        Summation_C_Beta_F=np.zeros((2),dtype='float')
        for ss_row in range(0,C_Beta_F_Mat_Arg.shape[0]):
            Summation_C_Beta_F=Sum_ArgExp(Summation_C_Beta_F, [C_Beta_F_Mat_Arg[ss_row][ss_col], C_Beta_F_Mat_Exp[ss_row][ss_col]])

        C_Beta_F_Vector_Arg[0][ss_col] = Summation_C_Beta_F[0]
        C_Beta_F_Vector_Exp[0][ss_col] = Summation_C_Beta_F[1]

    # amn = solve_amn(C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp, C_Beta_F_Vector_Arg,
    #                 C_Beta_F_Vector_Exp, solver)

    solve_amn(C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp, C_Beta_F_Vector_Arg,
              C_Beta_F_Vector_Exp, solver, m, N_order)
    
    amn = 0

    if m == 0:
        Em=1
    else:
        Em=2

    eta_i=np.cos(np.pi*Theta_i_deg/180)
    Em_x_Smn_hw_eta_i = []
    nn=-1
    for n in range(m + N_row_start , N_row_end + m):  # n index Loop: n=m,1,2,...,N : This fills the rows in Matrices
        nn=nn+1
        dmn_r_hw_ArgExp=func_dr_values_ArgExp(content_dr_values_hw, m, n , N_order_Fortran)# func_Adelman_dr_mn_of_c(DirPSWF, m, n, hw, Prec)

        Smn_hw_eta_i_arg_exp=func_Smn_eta_c_from_dr_ArgExp(m, n, eta_i, dmn_r_hw_ArgExp)
        Smn_hw_eta_i = Smn_hw_eta_i_arg_exp[0]*10**Smn_hw_eta_i_arg_exp[1]
        Em_x_Smn_hw_eta_i.append(Em * Smn_hw_eta_i)
    Em_x_Smn_hw_eta_i = np.asarray(Em_x_Smn_hw_eta_i)

    angles = np.zeros(num_angles_per_half_plane * 2, dtype=np.float64)
    pattern = np.zeros(num_angles_per_half_plane * 2, dtype=np.complex128)
    cos_phi_m = np.cos(0) # phi = 0
    counter = 0
    for theta_s in np.linspace(start=0, num=num_angles_per_half_plane, stop=180, endpoint=False):
        vec = compute_pattern_vec(amn, theta_s, content_dr_values_hw, Em_x_Smn_hw_eta_i, N_order_Fortran, N_row_start, N_row_end, m, cos_phi_m)
        pattern[counter] = (2/(1j*kw))*amn@np.transpose(vec)
        angles[counter] = theta_s
        counter += 1
    cos_phi_m = np.cos(np.pi * m) # phi = 180
    for theta_s in np.linspace(start=180, num=num_angles_per_half_plane, stop=0, endpoint=False):
        vec = compute_pattern_vec(amn, theta_s, content_dr_values_hw, Em_x_Smn_hw_eta_i, N_order_Fortran, N_row_start, N_row_end, m, cos_phi_m)
        pattern[counter] = (2/(1j*kw))*amn@np.transpose(vec)
        angles[counter] = 360 - theta_s
        counter += 1
    return angles, pattern


def compute_pattern_vec(amn, theta_s, content_dr_values_hw, Em_x_Smn_hw_eta_i, N_order_Fortran, N_row_start, N_row_end, m, cos_phi_m):
    vec = np.zeros(N_row_end, dtype=np.complex128)
    for n in range(m + N_row_start , N_row_end + m):
        #  add Smn(cos(theta_s)*cos(m*phi_s) to Em_x_Smn_hw_eta_i[n]
        eta_s=np.cos(np.pi*theta_s/180)
        dmn_r_hw_ArgExp=func_dr_values_ArgExp(content_dr_values_hw, m, n , N_order_Fortran)       
        smn_s_arg_exp = func_Smn_eta_c_from_dr_ArgExp(m, n, eta_s, dmn_r_hw_ArgExp)
        smn_hw_eta_s = smn_s_arg_exp[0]*10**smn_s_arg_exp[1]
        vec[n-m] = (Em_x_Smn_hw_eta_i[n - m] * smn_hw_eta_s * cos_phi_m)
    return vec
