import numpy as np
from scipy.special import lpmn


def func_dr_values(content1, m, l , lnum):

    import numpy as np

    data_of_line=content1[m*(int(2*lnum)+1)+2*(l-m)+1].split('\t')
    d_l_minus_m=float(data_of_line[0].split()[1])*10**(float(data_of_line[0].split()[2]))
    
    data_of_line=content1[m*(int(2*lnum)+1)+2*(l-m)+2].split('\t')[0].split()
    # Convert each string to float using a list comprehension
    enr_vec = [float(value) for value in data_of_line[1:-1]]
    
    
    if ((l-m) % 2)==0:
        ix=0
    else:
        ix=1
    
    dr_vec=np.zeros(ix+1+2*len(enr_vec),dtype='float')  
    print(d_l_minus_m)
    dr_vec[l-m]=d_l_minus_m
    
    # from d[l-m] to d[l-m-2], then to d[l-m-4], ... , d[0]
    for ii in range(l-m-2,0-1,-2):
        enr_Indx=int(np.floor(ii/2))
        dr_vec[ii]=round(dr_vec[ii+2]/enr_vec[enr_Indx],17)
    
    # from d[l-m] to d[l-m+2], then to d[l-m+4], ... , d[l-m+2*L]
    for ii in range(l-m,2*len(enr_vec),+2):
        enr_Indx=int(np.floor((ii)/2))
        dr_vec[ii+2]=round(dr_vec[ii]*enr_vec[enr_Indx],17)

    return dr_vec


def Get_ArgExp(Num): # Receives a number and returns its argument and exponet Num=Argx10**(Exponent)  
    if np.abs(Num==0):
       Expon=0
       Argu_Number=0
    else:
       Expon=int(np.log10(np.abs(Num)))  # get the exponent of argument multiplications
       Argu_Number= Num / (10**Expon)
    return [Argu_Number, Expon]

def Multi_ArgExp(AE1,AE2):
    # Input:
    # AE1: a 1x2 number in the format of [Argument, exponent]
    # AE2: a 1x2 number in the format of [Argument, exponent]
    # Output 
    if np.abs(np.abs(AE1[0]*AE2[0])==0):
       Expon=0
       Argu_Number=0
    else:
       Expon_ofArgs=int(np.log10(np.abs(AE1[0]*AE2[0])))  # get the exponent of argument multiplications
       Argu_Number= AE1[0]*AE2[0] / (10**Expon_ofArgs)
       Expon= Expon_ofArgs +  AE1[1] + AE2[1]
    
    return [Argu_Number, Expon]


def Sum_ArgExp(AE1,AE2):
    # Input:
    # AE1: a 1x2 number in the format of [Argument, exponent]
    # AE2: a 1x2 number in the format of [Argument, exponent]
       
    MaxExp=0.5 * (AE1[1] + AE2[1] + np.abs(AE1[1] - AE2[1]) ) # selects maximum of exponent
    Argu_SumNum=AE1[0]*(10**(AE1[1]-MaxExp)) + AE2[0]*(10**(AE2[1]-MaxExp))
    
    if np.abs(np.abs(Argu_SumNum)==0):
       Expon=0
       Argu_Number=0
    else:
       Expon_ofArgs=int(np.log10(np.abs(Argu_SumNum)))  # get the exponent of argument multiplications
       Argu_Number= Argu_SumNum / (10**Expon_ofArgs)
       Expon= Expon_ofArgs +  MaxExp
    
    return [Argu_Number, Expon]

def func_dr_values_ArgExp(content1, m, l , lnum):

    import numpy as np

    data_of_line=content1[m*(int(2*lnum)+1)+2*(l-m)+1].split('\t')
    d_l_minus_m=float(data_of_line[0].split()[1])*10**(float(data_of_line[0].split()[2]))
    
    data_of_line=content1[m*(int(2*lnum)+1)+2*(l-m)+2].split('\t')[0].split()
    # Convert each string to float using a list comprehension
    enr_vec = [float(value) for value in data_of_line[1:-1]]
    
    #Disregard last part of enr_vec which is zero:------------------------
    ZeroInds = np.where(np.abs(enr_vec) > 1E-20)[0]
    if ZeroInds[-1] < len(enr_vec):
       enr_vec = enr_vec[0:(ZeroInds[-1]+1)]
    #---------------------------------------------------------------------
    
    if ((l-m) % 2)==0:
        ix=0
    else:
        ix=1
    
    dr_vec=np.zeros(ix+1+2*len(enr_vec),dtype='float') 
    dr_vec[l-m]=d_l_minus_m
    
    dr_vec_argExp=np.zeros((ix+1+2*len(enr_vec),2),dtype='float')  
    if np.isnan(dr_vec[l-m]):
       dr_vec[l-m] = 0 
    [dr_vec_argExp[l-m,0],dr_vec_argExp[l-m,1]]=Get_ArgExp(dr_vec[l-m])
    
    # from d[l-m] to d[l-m-2], then to d[l-m-4], ... , d[0]
    for ii in range(l-m-2,0-1,-2):
        enr_Indx=int(np.floor(ii/2))
        if np.abs(enr_vec[enr_Indx]) < 1E-30:
            enr_vec[enr_Indx]=1E-30
        dr_vec[ii]=round(dr_vec[ii+2]/enr_vec[enr_Indx],17)
        if np.isnan(dr_vec[ii]):
            dr_vec[ii] = 0 
        [dr_vec_argExp[ii,0],dr_vec_argExp[ii,1]]=Get_ArgExp(dr_vec[ii])        

        
        
    # from d[l-m] to d[l-m+2], then to d[l-m+4], ... , d[l-m+2*L]
    for ii in range(l-m,2*len(enr_vec),+2):
        enr_Indx=int(np.floor((ii)/2))
        dr_vec[ii+2]=round(dr_vec[ii]*enr_vec[enr_Indx],17)
        if np.isnan(dr_vec[ii]):
            dr_vec[ii] = 0 
        [dr_vec_argExp[ii+2,0],dr_vec_argExp[ii+2,1]]=Get_ArgExp(dr_vec[ii+2])
        
   
    return dr_vec_argExp

def func_Ang_SWF(content1, m, l , lnum):

    data_of_line_eta_i=content1[m*(3*int(lnum)+1)+3*(l-m)+2].split('\t')
    data_of_line_eta_s=content1[m*(3*int(lnum)+1)+3*(l-m)+3].split('\t')

    S1_eta_i=float(data_of_line_eta_i[0].split()[1])*10**(float(data_of_line_eta_i[0].split()[2]))
    S1_eta_s=float(data_of_line_eta_s[0].split()[3])*10**(float(data_of_line_eta_s[0].split()[4]))

    return [S1_eta_i , S1_eta_s]


def func_Rad_SWF(content1, m, l , lnum):
    
    data_of_line=content1[m*(int(lnum)+1)+(l-m)+1].split('\t')
    r1=float(data_of_line[0].split()[1])*10**(float(data_of_line[0].split()[2]))
    dr1=float(data_of_line[0].split()[3])*10**(float(data_of_line[0].split()[4]))
    r2=float(data_of_line[0].split()[5])*10**(float(data_of_line[0].split()[6]))
    dr2=float(data_of_line[0].split()[7])*10**(float(data_of_line[0].split()[8]))
    
    return [r1 , dr1 , r2 , dr2]

def func_Rad_SWF_ArgExp(content1, m, l , lnum):
    
    data_of_line=content1[m*(int(lnum)+1)+(l-m)+1].split('\t')
    r1=[float(data_of_line[0].split()[1]), float(data_of_line[0].split()[2])]
    dr1=[float(data_of_line[0].split()[3]), float(data_of_line[0].split()[4])]
    r2=[float(data_of_line[0].split()[5]), float(data_of_line[0].split()[6])]
    dr2=[float(data_of_line[0].split()[7]), float(data_of_line[0].split()[8])]
    
    return r1 , dr1 , r2 , dr2

def Factorial_Mterms_ArgExp(NN,MMterms):
    Argu_Number = 1
    Expon = 0
    for ii in range(0, MMterms):
        Number = NN - ii

        Expon_Number = int(np.log10(np.abs(Argu_Number*Number)))
        Argu_Number = Argu_Number * Number / (10**Expon_Number)

        Expon = Expon + Expon_Number

    return Argu_Number, Expon


def func_Smn_eta_c_from_dr(m,n,eta,dmn_r_vec):

    
    def Factorial_Mterms(NN,MMterms):
        Fact=1.0
        for ii in range(0,MMterms):
          Fact=Fact*(NN-ii)
        return Fact
    
    #.................................................................................................................
    #............    Activate this part if dmn are scaled by Flemmer approach     ............
    #.................................................................................................................
    #Scale dr as eq (3) in Van Buren 2002
    #SUM=0
    #if ((n-m) % 2) == 0:
    #    for rr in range(0, int(0.5*len(dmn_r)-1) ):
    #        r=2*rr
    #        Coeff=2*Factorial_Mterms((r+2*m),2*m)/(2*(r+m)+1)
    #        SUM=SUM+Coeff*dmn_r[r]*dmn_r[r]
    #
    #if ((n-m) % 2) == 1:
    #    for rr in range(0, int(0.5*len(dmn_r)-1) ):
    #        r=2*rr+1
    #        Coeff=2*Factorial_Mterms((r+2*m),2*m)/(2*(r+m)+1)
    #        SUM=SUM+Coeff*dmn_r[r]*dmn_r[r]
    #        
    #RHS=2*Factorial_Mterms((n+m),(n+m))/( (2*n+1)*Factorial_Mterms((n-m),(n-m)) ) 
    #
    #Scale=(RHS/SUM)**0.5
    #dmn_r=Scale*dmn_r
    dmn_r=dmn_r_vec
    from scipy.special import lpmn
    Nr=len(dmn_r)
    Pmn_eta=( lpmn(m,m+int(Nr),eta)[0] )[m]
    
    
    S_mn_c1_eta=0
    if ((n-m) % 2) == 0:
       for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r])*(Pmn_eta[m+2*r])
      
    if (n-m) % 2 == 1:
        for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r+1])*(Pmn_eta[m+2*r+1])
            
    RHS=2*Factorial_Mterms((n+m),(n+m))/( (2*n+1)*Factorial_Mterms((n-m),(n-m)) )
    
    Nmn=RHS
    S_mn_c1_eta=S_mn_c1_eta/((Nmn)**0.5)
    return S_mn_c1_eta

def func_Smn_eta_c_from_dr_log(m,n,eta,dmn_r_vec):
    import numpy as np
    def log_meixner_norm(m, n):
        def log_factorial(NN, MMterms):
            log = 0.0
            for ii in range(0,MMterms):
                log += np.log(NN-ii)
            return log
        return np.log(2.0) + log_factorial((n+m), (n+m)) - np.log((2*n+1)) - log_factorial((n-m), (n-m))


    #.................................................................................................................
    #............    Activate this part if dmn are scaled by Flemmer approach     ............
    #.................................................................................................................
    #Scale dr as eq (3) in Van Buren 2002
    #SUM=0
    #if ((n-m) % 2) == 0:
    #    for rr in range(0, int(0.5*len(dmn_r)-1) ):
    #        r=2*rr
    #        Coeff=2*Factorial_Mterms((r+2*m),2*m)/(2*(r+m)+1)
    #        SUM=SUM+Coeff*dmn_r[r]*dmn_r[r]
    #
    #if ((n-m) % 2) == 1:
    #    for rr in range(0, int(0.5*len(dmn_r)-1) ):
    #        r=2*rr+1
    #        Coeff=2*Factorial_Mterms((r+2*m),2*m)/(2*(r+m)+1)
    #        SUM=SUM+Coeff*dmn_r[r]*dmn_r[r]
    #
    #RHS=2*Factorial_Mterms((n+m),(n+m))/( (2*n+1)*Factorial_Mterms((n-m),(n-m)) )
    #
    #Scale=(RHS/SUM)**0.5
    #dmn_r=Scale*dmn_r
    dmn_r=dmn_r_vec
    from scipy.special import lpmn
    Nr=len(dmn_r)
    Pmn_eta=( lpmn(m,m+int(Nr),eta)[0] )[m]



    S_mn_c1_eta=0
    if ((n-m) % 2) == 0:
        for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r])*(Pmn_eta[m+2*r])

    if (n-m) % 2 == 1:
        for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r+1])*(Pmn_eta[m+2*r+1])

    sign_smn = np.sign(S_mn_c1_eta)
    abs_smn = np.abs(S_mn_c1_eta)
    log_norm = log_meixner_norm(m, n)
    log_smn = np.where(abs_smn > 0.0, np.log(abs_smn), 0) - 0.5 * log_norm
    return np.where(abs_smn > 0.0, sign_smn * np.exp(log_smn), 0)



def func_Smn_eta_c_from_dr_ArgExp(m, n, eta, dmn_r_vec, Pmn_eta):
    Nr = len(dmn_r_vec)
    S_mn_c1_eta_arg = 0.0
    S_mn_c1_eta_exp = 0

    for r in range(Nr):
        idx = 2 * r if (n - m) % 2 == 0 else 2 * r + 1
        if idx >= Nr:
            break
        term_arg, term_exp = Multi_ArgExp(dmn_r_vec[idx], Get_ArgExp(Pmn_eta[m + idx]))
        S_mn_c1_eta_arg, S_mn_c1_eta_exp = Sum_ArgExp((S_mn_c1_eta_arg, S_mn_c1_eta_exp), (term_arg, term_exp))

    RHS_arg, RHS_exp = Multi_ArgExp(Get_ArgExp(2 / (2 * n + 1)), Factorial_Mterms_ArgExp(n + m, 2 * m))
    Nmn_arg, Nmn_exp = RHS_arg, RHS_exp
    Nmn_sqrt_arg = Nmn_arg ** -0.5
    Nmn_sqrt_exp = -0.5 * Nmn_exp

    S_mn_c1_eta_arg, S_mn_c1_eta_exp = Multi_ArgExp((S_mn_c1_eta_arg, S_mn_c1_eta_exp), (Nmn_sqrt_arg, Nmn_sqrt_exp))
    

    return S_mn_c1_eta_arg, S_mn_c1_eta_exp

def func_Integrate_SmnEta1_SmlEta2_ArgExp(_m, _n, _l, _dmn_r_h1_vec, _dml_r_h2_vec):

    
    Sum_r_terms=np.zeros((2),dtype='float')
    R_terms=170
    for rr in range(0,min(int(len(_dmn_r_h1_vec)/2), int(len(_dml_r_h2_vec)/2), R_terms)):
        if ((_n + _m) % 2)==0:
            rr=2*rr
        else:
            rr=2*rr+1
             
        FactorielTerm = Factorial_Mterms_ArgExp(rr+2*_m,2*_m)
        Coefficient=Multi_ArgExp(FactorielTerm, Get_ArgExp(1/(rr+_m+0.5))) 
        dmn_r_h1_x_dml_r_h2 = Multi_ArgExp((_dmn_r_h1_vec[rr, :]), (_dml_r_h2_vec[rr, :]))
        Sum_r_terms = Sum_ArgExp(Sum_r_terms, Multi_ArgExp(Coefficient, dmn_r_h1_x_dml_r_h2))
        
    
#    Integrate_SmnEta1_SmlEta2_ArgExp=Sum_r_terms
    Mmn=Multi_ArgExp( Get_ArgExp(2/(2*_n+1)), Factorial_Mterms_ArgExp((_n+_m),2*_m) )
    Mml=Multi_ArgExp( Get_ArgExp(2/(2*_l+1)), Factorial_Mterms_ArgExp((_l+_m),2*_m) )
    MmnMml=Multi_ArgExp([Mmn[0], Mmn[1]], [Mml[0], Mml[1]])
    Integrate_SmnEta1_SmlEta2_ArgExp = Multi_ArgExp(Sum_r_terms, [MmnMml[0]**(-0.5), MmnMml[1]*(-0.5)])
    return Integrate_SmnEta1_SmlEta2_ArgExp


####################### ------------------- Vectorized functions------------------- #######################

def func_Rad_SWF_ArgExp_vectorized(content, m, l_array, lnum):
    indices = m * (int(lnum) + 1) + (l_array - m) + 1
    data_lines = [content[idx].split('\t') for idx in indices]
    
    r1_list = np.array([[float(line[0].split()[1]), float(line[0].split()[2])] for line in data_lines])
    dr1_list = np.array([[float(line[0].split()[3]), float(line[0].split()[4])] for line in data_lines])
    r2_list = np.array([[float(line[0].split()[5]), float(line[0].split()[6])] for line in data_lines])
    dr2_list = np.array([[float(line[0].split()[7]), float(line[0].split()[8])] for line in data_lines])
    
    return r1_list, dr1_list, r2_list, dr2_list

def func_dr_values_ArgExp_vectorized(content, m, l_array, lnum):
    indices = m * (int(2 * lnum) + 1) + 2 * (l_array - m) + 1
    d_l_minus_m_list = []
    enr_vec_list = []
    for idx in indices:
        # Read d_l_minus_m
        data_line = content[idx].split('\t')
        d_l_minus_m = float(data_line[0].split()[1]) * 10 ** (float(data_line[0].split()[2]))
        d_l_minus_m_list.append(d_l_minus_m)
        # Read enr_vec
        data_line = content[idx + 1].split('\t')[0].split()
                
        enr_vec = []
        
        ###---------------------------------------------###
        # Done to fix errors where the data from the fortan output is not in scientiffic notation,
        # This seems to be the case when numbers become smaller then xxe-100 or smaller, 
        # I suspect it is because the exponent becomes 3 digits and then the "e" dissapears.
        # https://stackoverflow.com/questions/24004824/for-three-digit-exponents-fortran-drops-the-e-in-the-output

        for value in data_line[1:-1]:
            try:
                # Attempt to convert the value directly
                parsed_value = float(value)
            except ValueError:
                # If conversion fails, assume scientific notation without 'e'
                # For example "0.1326-306" 
                # Split the string into the base and exponent parts manually
                base_part = value[:-4]  # All characters except the last 4
                exponent_part = value[-4:]  # Last 4 characters, assuming it's like "-306"

                # Reconstruct with scientific notation format, manualy adding the e
                parsed_value = float(f"{base_part}e{exponent_part}")


            enr_vec.append(parsed_value)
        
        # Disregard last part of enr_vec which is zero
        ZeroInds = np.where(np.abs(enr_vec) > 1E-20)[0]
        if len(ZeroInds) > 0:
            enr_vec = enr_vec[:ZeroInds[-1] + 1]
        else:
            # All elements are zero
            enr_vec = []
        
        enr_vec_list.append(enr_vec)
    
    # Now process dr_vec_argExp for each l
    dr_vec_argExp_list = []
    for d_l_minus_m, enr_vec, l in zip(d_l_minus_m_list, enr_vec_list, l_array):
        if ((l - m) % 2) == 0:
            ix = 0
        else:
            ix = 1
        
        dr_vec = np.zeros(ix + 1 + 2 * len(enr_vec), dtype='float')
        dr_vec_argExp = np.zeros((ix + 1 + 2 * len(enr_vec), 2), dtype='float')
        dr_vec[l - m] = d_l_minus_m
        if np.isnan(dr_vec[l - m]):
            dr_vec[l - m] = 0
        dr_vec_argExp[l - m, :] = Get_ArgExp(dr_vec[l - m])
        
        # Backward recursion
        for ii in range(l - m - 2, -1, -2):
            enr_Indx = int(np.floor(ii / 2))
            if np.abs(enr_vec[enr_Indx]) < 1E-18:
                enr_vec[enr_Indx] = 1E-18
            dr_vec[ii] = round(dr_vec[ii + 2] / enr_vec[enr_Indx], 17)
            if np.isnan(dr_vec[ii]):
                dr_vec[ii] = 0
            dr_vec_argExp[ii, :] = Get_ArgExp(dr_vec[ii])
        
        # Forward recursion
        for ii in range(l - m, 2 * len(enr_vec), +2):
            enr_Indx = int(np.floor(ii / 2))
            dr_vec[ii + 2] = round(dr_vec[ii] * enr_vec[enr_Indx], 17)
            if np.isnan(dr_vec[ii + 2]):
                dr_vec[ii + 2] = 0
            dr_vec_argExp[ii + 2, :] = Get_ArgExp(dr_vec[ii + 2])
        
        dr_vec_argExp_list.append(dr_vec_argExp)
    
    return dr_vec_argExp_list  # List of arrays




## Improved vectorized version that gets rid of the loop over r, much faster ##
def func_Smn_eta_c_from_dr_ArgExp_vectorized(m, n_array, eta, dmn_r_vec_list):
    S_mn_c1_eta_list = []
    
    for n, dmn_r_vec in zip(n_array, dmn_r_vec_list):
        Nr = dmn_r_vec.shape[0]
        max_order = m + Nr - 1
        
        # Compute associated Legendre functions up to required order
        Pmn_eta_full=( lpmn(m,m+int(Nr),eta)[0] )[m]
        
        # Determine indices based on whether (n - m) is even or odd
        if (n - m) % 2 == 0:
            idx = np.arange(0, Nr, 2)
        else:
            idx = np.arange(1, Nr, 2)
        
        # Extract relevant terms from dmn_r_vec and Pmn_eta
        dmn_r_vec_selected = dmn_r_vec[idx, :] 
        Pmn_eta_selected = Pmn_eta_full[m + idx]
        
        # Add numerical stability checks to prevent singular matrices
        valid_mask = (np.abs(dmn_r_vec_selected[:, 0]) > 1e-300) & (np.abs(Pmn_eta_selected) > 1e-300)
        if not np.any(valid_mask):
            S_mn_c1_eta_list.append([0.0, 0.0])
            continue
            
        # Filter out small values
        dmn_r_vec_selected = dmn_r_vec_selected[valid_mask]
        Pmn_eta_selected = Pmn_eta_selected[valid_mask]

        Pmn_eta_argexp = Get_ArgExp_array(Pmn_eta_selected) 
        
        terms = Multi_ArgExp_array(dmn_r_vec_selected, Pmn_eta_argexp)  # Shape: (num_terms, 2)
        
        S_mn_c1_eta = Sum_ArgExp_over_array(terms)
        
        Mmn = Multi_ArgExp(Get_ArgExp(2 / (2 * n + 1)), Factorial_Mterms_ArgExp(n + m, 2 * m))
        Mmn_sqrt_inv = [Mmn[0] ** (-0.5), Mmn[1] * (-0.5)]
        
        S_mn_c1_eta = Multi_ArgExp(S_mn_c1_eta, Mmn_sqrt_inv)
        S_mn_c1_eta_list.append(S_mn_c1_eta)
    
    return np.array(S_mn_c1_eta_list)



## Uses logarithms to speed up computations ##
def func_Integrate_SmnEta1_SmlEta2_ArgExp_vectorized(m, n, l, dmn_r_h1_vec, dml_r_h2_vec):
    from scipy.special import gammaln
    R_terms = 170
    max_rr = min(len(dmn_r_h1_vec) // 2, len(dml_r_h2_vec) // 2, R_terms)
    rr = np.arange(0, max_rr)
    if ((n + m) % 2) == 0:
        rr_indices = 2 * rr
    else:
        rr_indices = 2 * rr + 1

    rr_index_plus_2m = rr_indices + 2 * m

    # Compute logarithm of factorial terms
    ln_factorial_terms = gammaln(rr_index_plus_2m + 1) - gammaln(rr_index_plus_2m - 2 * m + 1)
    log10_factorial_terms = ln_factorial_terms / np.log(10)

    denom = rr_indices + m + 0.5
    log10_coefficients = log10_factorial_terms - np.log10(denom)

    arg1 = dmn_r_h1_vec[rr_indices, 0]
    exp1 = dmn_r_h1_vec[rr_indices, 1]
    arg2 = dml_r_h2_vec[rr_indices, 0]
    exp2 = dml_r_h2_vec[rr_indices, 1]


    sign1 = np.sign(arg1)
    sign2 = np.sign(arg2)
    sign_product = sign1 * sign2

    # Create masks for non-zero arguments
    valid_arg1 = arg1 != 0
    valid_arg2 = arg2 != 0
    valid_indices = valid_arg1 & valid_arg2

    # If no valid indices, return zero
    if not np.any(valid_indices):
        return np.array([0.0, 0.0])

    # Filter arrays based on valid indices
    arg1_abs = np.abs(arg1[valid_indices])
    arg2_abs = np.abs(arg2[valid_indices])
    exp1_valid = exp1[valid_indices]
    exp2_valid = exp2[valid_indices]
    sign_product_valid = sign_product[valid_indices]
    log10_coefficients_valid = log10_coefficients[valid_indices]


    log10_arg1 = np.log10(arg1_abs) + exp1_valid
    log10_arg2 = np.log10(arg2_abs) + exp2_valid


    total_log10 = log10_arg1 + log10_arg2 + log10_coefficients_valid

    # Sum the values in linear space, considering signs
    max_log10 = np.max(total_log10)
    scaled_values = sign_product_valid * 10 ** (total_log10 - max_log10)
    sum_scaled = np.sum(scaled_values)

    # If the sum is zero, return zero to avoid log10(0)
    if sum_scaled == 0:
        return np.array([0.0, 0.0])

    sign_sum = np.sign(sum_scaled)
    abs_sum = np.abs(sum_scaled)
    log10_abs_sum = np.log10(abs_sum) + max_log10
    Expon_ofArgs = np.floor(log10_abs_sum)
    Argu_Number = sign_sum * 10 ** (log10_abs_sum - Expon_ofArgs)
    Expon = Expon_ofArgs

    # Compute Mmn and Mml using logarithms
    ln_Mmn = np.log(2 / (2 * n + 1)) + gammaln(n + m + 1) - gammaln(n - m + 1)
    ln_Mml = np.log(2 / (2 * l + 1)) + gammaln(l + m + 1) - gammaln(l - m + 1)
    ln_MmnMml = ln_Mmn + ln_Mml
    log10_MmnMml_sqrt_inv = -0.5 * ln_MmnMml / np.log(10)

    # Final multiplication in log space
    final_log10 = np.log10(np.abs(Argu_Number)) + Expon + log10_MmnMml_sqrt_inv
    final_sign = np.sign(Argu_Number)
    final_exponent = np.floor(final_log10)
    final_arg = final_sign * 10 ** (final_log10 - final_exponent)

    return np.array([final_arg, final_exponent])


####################### ------------------- Array functions ------------------- #######################

def Get_ArgExp_array(Num_array):
    Num_array = np.array(Num_array)
    
    Expon = np.zeros_like(Num_array.real)
    Argu_Number = np.zeros_like(Num_array)


    nonzero = Num_array != 0
    Expon[nonzero] = np.floor(np.log10(np.abs(Num_array[nonzero])))
    Argu_Number[nonzero] = Num_array[nonzero] / (10.0 ** Expon[nonzero])
    return np.stack([Argu_Number, Expon], axis=-1)


def Multi_ArgExp_array(AE1_array, AE2_array):
    AE1_array = np.array(AE1_array)
    AE2_array = np.array(AE2_array)
    Argu_Product = (AE1_array[..., 0] * AE2_array[..., 0])
    Expon_Product = AE1_array[..., 1] + AE2_array[..., 1]
    
    nonzero = Argu_Product != 0
    
    Expon_ofArgs = np.zeros_like(Argu_Product.real)
    Expon_ofArgs[nonzero] = np.floor(np.log10(np.abs(Argu_Product[nonzero])))
    
    Argu_Number = np.zeros_like(Argu_Product)

    # Setter null on non-nonzero (example where Argu_Product is zero)
    Argu_Number[~nonzero] = 0
    Expon_ofArgs[~nonzero] = 0

    Argu_Number[nonzero] = Argu_Product[nonzero] / (10.0 ** Expon_ofArgs[nonzero])

    Expon = Expon_ofArgs + Expon_Product.real
    return np.stack([Argu_Number, Expon], axis=-1)




def Sum_ArgExp_array(AE1_array, AE2_array):
    AE1_array = np.array(AE1_array)
    AE2_array = np.array(AE2_array)
    
    MaxExp = 0.5 * (AE1_array[..., 1].real + AE2_array[..., 1].real + np.abs(AE1_array[..., 1].real - AE2_array[..., 1].real))
    
    Argu_SumNum = AE1_array[..., 0] * (10.0 ** (AE1_array[..., 1] - MaxExp)) + AE2_array[..., 0] * (10.0 ** (AE2_array[..., 1] - MaxExp))
    
    nonzero = Argu_SumNum != 0
    Expon_ofArgs = np.zeros_like(Argu_SumNum.real)
    Expon_ofArgs[nonzero] = np.floor(np.log10(np.abs(Argu_SumNum[nonzero]))).astype(int)
    
    Argu_Number = np.zeros_like(Argu_SumNum)
    
    Argu_Number[nonzero] = Argu_SumNum[nonzero] / (10.0 ** Expon_ofArgs[nonzero])
    
    Expon = Expon_ofArgs + MaxExp.real
    return np.stack([Argu_Number, Expon], axis=-1)



def Sum_ArgExp_over_array(AE_array):
    MaxExp = np.max(AE_array[:, 1].real)
    Argu_SumNum = AE_array[:, 0] * (10.0 ** (AE_array[:, 1].real - MaxExp))
    Argu_Sum = np.sum(Argu_SumNum)
    if Argu_Sum == 0:
        return np.array([0.0, 0.0])
    else:
        Expon_ofArgs = np.floor(np.log10(np.abs(Argu_Sum)))
        Argu_Number = Argu_Sum / (10.0 ** Expon_ofArgs)
        Expon = Expon_ofArgs + MaxExp
        return np.array([Argu_Number, Expon])


def Sum_ArgExp_over_rows(AE_matrix):
    """
    Sum over rows of argument-exponent pairs for each column.

    Parameters:
    - AE_matrix: NumPy array of shape (num_rows, num_cols, 2)
                 The last dimension holds [Argument, Exponent].

    Returns:
    - Argu_Number: NumPy array of shape (num_cols,)
                   The arguments of the summed values.
    - Expon: NumPy array of shape (num_cols,)
             The exponents of the summed values.
    """
    # Extract exponents and compute maximum exponent per column
    MaxExp = np.max(AE_matrix[:, :, 1].real, axis=0)

    # Compute exponent differences
    Exponent_diffs = AE_matrix[:, :, 1].real - MaxExp[np.newaxis, :]

    # Scale arguments and sum over rows
    Argu_SumNum = AE_matrix[:, :, 0] * (10.0 ** Exponent_diffs)
    Argu_Sum = np.sum(Argu_SumNum, axis=0)

    # Handle zeros to avoid log10(0)
    nonzero = Argu_Sum != 0
    Expon_ofArgs = np.zeros_like(Argu_Sum.real)
    Expon_ofArgs[nonzero] = np.floor(np.log10(np.abs(Argu_Sum[nonzero])))

    # Normalize arguments
    Argu_Number = np.zeros_like(Argu_Sum, dtype='complex128')
    Argu_Number[nonzero] = Argu_Sum[nonzero] / (10.0 ** Expon_ofArgs[nonzero])

    # Compute final exponents
    Expon = Expon_ofArgs + MaxExp

    return Argu_Number, Expon