import numpy as np

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
    #    print(ii)
        enr_Indx=int(np.floor(ii/2))
    #    print(enr_Indx)
        dr_vec[ii]=round(dr_vec[ii+2]/enr_vec[enr_Indx],17)
    
    # from d[l-m] to d[l-m+2], then to d[l-m+4], ... , d[l-m+2*L]
    for ii in range(l-m,2*len(enr_vec),+2):
    #    print(ii)
        enr_Indx=int(np.floor((ii)/2))
    #    print(enr_Indx)
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
    #    print(ii)
        enr_Indx=int(np.floor(ii/2))
        if np.abs(enr_vec[enr_Indx]) < 1E-18:
            enr_vec[enr_Indx]=1E-18
    #    print(enr_Indx)
        dr_vec[ii]=round(dr_vec[ii+2]/enr_vec[enr_Indx],17)
        if np.isnan(dr_vec[ii]):
            dr_vec[ii] = 0 
        [dr_vec_argExp[ii,0],dr_vec_argExp[ii,1]]=Get_ArgExp(dr_vec[ii])        

        
        
    # from d[l-m] to d[l-m+2], then to d[l-m+4], ... , d[l-m+2*L]
    for ii in range(l-m,2*len(enr_vec),+2):
    #    print(ii)
        enr_Indx=int(np.floor((ii)/2))
#        if np.abs(enr_vec[enr_Indx]) < 1E-16:
#            enr_vec[enr_Indx]=1E-16
    #    print(enr_Indx)
        dr_vec[ii+2]=round(dr_vec[ii]*enr_vec[enr_Indx],17)
        if np.isnan(dr_vec[ii]):
            dr_vec[ii] = 0 
        [dr_vec_argExp[ii+2,0],dr_vec_argExp[ii+2,1]]=Get_ArgExp(dr_vec[ii+2])
        
   
    return dr_vec_argExp

def func_Ang_SWF(content1, m, l , lnum):

#    import numpy as np

#    data_of_line_m=content1[m*(3*int(lnum)+1)].split('\t')
#    data_of_line_l=content1[m*(3*int(lnum)+1)+3*(l-m)+1].split('\t')
    data_of_line_eta_i=content1[m*(3*int(lnum)+1)+3*(l-m)+2].split('\t')
    data_of_line_eta_s=content1[m*(3*int(lnum)+1)+3*(l-m)+3].split('\t')
#    print(m,l,data_of_line_eta_i)
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
#    return f"{Argu_Number}E{Expon}"

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
    
   
    #    print(Pmn_eta.shape)
    
    #    print(m+int(0.5*Nr),Pmn_eta.shape)
    
    S_mn_c1_eta=0
    if ((n-m) % 2) == 0:
       for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r])*(Pmn_eta[m+2*r])
#            print('term: ',(dmn_r[2*r])*(Pmn_eta[m+2*r]))
    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r])
    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r],S_mn_c1_eta)
       
    if (n-m) % 2 == 1:
        for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r+1])*(Pmn_eta[m+2*r+1])
#            print('term: ',(dmn_r[2*r+1])*(Pmn_eta[m+2*r+1]))
            
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


    #    print(Pmn_eta.shape)

    #    print(m+int(0.5*Nr),Pmn_eta.shape)

    S_mn_c1_eta=0
    if ((n-m) % 2) == 0:
        for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r])*(Pmn_eta[m+2*r])
    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r])
    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r],S_mn_c1_eta)

    if (n-m) % 2 == 1:
        for r in range( 0,int(0.5*Nr)):
            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r+1])*(Pmn_eta[m+2*r+1])

    sign_smn = np.sign(S_mn_c1_eta)
    abs_smn = np.abs(S_mn_c1_eta)
    log_norm = log_meixner_norm(m, n)
    log_smn = np.where(abs_smn > 0.0, np.log(abs_smn), 0) - 0.5 * log_norm
    return np.where(abs_smn > 0.0, sign_smn * np.exp(log_smn), 0)


#def func_Smn_eta_c_from_dr_Stable(m,n,eta,dmn_r_vec):
#
#
#    
#    def Factorial_Mterms_Stable(NN,MMterms):
#        Argu_Number = 1
#        Expon = 0
#        for ii in range(0, MMterms):
#            Number = NN - ii
#    
#            Expon_Number = int(np.log10(Argu_Number*Number))
#            Argu_Number = Argu_Number * Number / (10**Expon_Number)
#    
#            Expon = Expon + Expon_Number
#    
#        return Argu_Number, Expon
#        
#    #.................................................................................................................
#    #............    Activate this part if dmn are scaled by Flemmer approach     ............
#    #.................................................................................................................
#    #Scale dr as eq (3) in Van Buren 2002
#    #SUM=0
#    #if ((n-m) % 2) == 0:
#    #    for rr in range(0, int(0.5*len(dmn_r)-1) ):
#    #        r=2*rr
#    #        Coeff=2*Factorial_Mterms((r+2*m),2*m)/(2*(r+m)+1)
#    #        SUM=SUM+Coeff*dmn_r[r]*dmn_r[r]
#    #
#    #if ((n-m) % 2) == 1:
#    #    for rr in range(0, int(0.5*len(dmn_r)-1) ):
#    #        r=2*rr+1
#    #        Coeff=2*Factorial_Mterms((r+2*m),2*m)/(2*(r+m)+1)
#    #        SUM=SUM+Coeff*dmn_r[r]*dmn_r[r]
#    #        
#    #RHS=2*Factorial_Mterms((n+m),(n+m))/( (2*n+1)*Factorial_Mterms((n-m),(n-m)) ) 
#    #
#    #Scale=(RHS/SUM)**0.5
#    #dmn_r=Scale*dmn_r
#    dmn_r=dmn_r_vec
#    from scipy.special import lpmn
#    Nr=len(dmn_r)
#    Pmn_eta=( lpmn(m,m+int(Nr),eta)[0] )[m]
#    
#   
#    #    print(Pmn_eta.shape)
#    
#    #    print(m+int(0.5*Nr),Pmn_eta.shape)
#    
#    S_mn_c1_eta=0
#    if ((n-m) % 2) == 0:
#       for r in range( 0,int(0.5*Nr)):
#            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r])*(Pmn_eta[m+2*r])
#    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r])
#    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r],S_mn_c1_eta)
#       
#    if (n-m) % 2 == 1:
#        for r in range( 0,int(0.5*Nr)):
#            S_mn_c1_eta=S_mn_c1_eta + (dmn_r[2*r+1])*(Pmn_eta[m+2*r+1])
#    
#    #    RHS=2*Factorial_Mterms_Stable((n+m),(n+m))/( (2*n+1)*Factorial_Mterms_Stable((n-m),(n-m)) )
#    RHS_nom=Factorial_Mterms_Stable((n+m),(n+m))
#    RHS_denom=( Factorial_Mterms_Stable((n-m),(n-m)) )
#    
#    Arg=RHS_nom[0]/RHS_denom[0]
#    Exp=(RHS_nom[1]-RHS_denom[1])
#    RHS=(2/(2*n+1))*float(f"{Arg}E{Exp}")
#    
#    Nmn=RHS
#    S_mn_c1_eta=S_mn_c1_eta/((Nmn)**0.5)
#    return S_mn_c1_eta


def func_Smn_eta_c_from_dr_ArgExp(m,n,eta,dmn_r_vec):

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
    

    #    print(Pmn_eta.shape)
    #    print(m+int(0.5*Nr),Pmn_eta.shape)
    
    S_mn_c1_eta=np.zeros((2),dtype='float')
    if ((n-m) % 2) == 0:
       for r in range( 0,int(0.5*Nr)):
#            print('r >>> ',r, S_mn_c1_eta, Multi_ArgExp( (dmn_r[2*r, :]),Get_ArgExp(Pmn_eta[m+2*r]) )) 
           # Eq1 in (Van Buren & Boisvert 2002, 2004)
            S_mn_c1_eta=Sum_ArgExp(S_mn_c1_eta , Multi_ArgExp( (dmn_r[2*r,:]),Get_ArgExp(Pmn_eta[m+2*r]) ) )
    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r])
    #            print(2*r,dr_of_line[2*r],Pmn_eta[m][m+2*r],S_mn_c1_eta)
       
    if (n-m) % 2 == 1:
       for r in range( 0,int(0.5*Nr)):
#            print('r >>> ',r, S_mn_c1_eta, Multi_ArgExp( (dmn_r[2*r+1, :]),Get_ArgExp(Pmn_eta[m+2*r+1]) ))
           # Eq1 in (Van Buren & Boisvert 2002, 2004)
            S_mn_c1_eta=Sum_ArgExp(S_mn_c1_eta ,  Multi_ArgExp( (dmn_r[2*r+1, :]),Get_ArgExp(Pmn_eta[m+2*r+1]) ) )
    
    #    RHS=2*Factorial_Mterms_Stable((n+m),(n+m))/( (2*n+1)*Factorial_Mterms_Stable((n-m),(n-m)) )
    RHS=Multi_ArgExp( Get_ArgExp(2/(2*n+1)), Factorial_Mterms_ArgExp((n+m),2*m) )
   
    Nmn=RHS
#    S_mn_c1_eta=S_mn_c1_eta/((Nmn)**0.5)
    S_mn_c1_eta = Multi_ArgExp( S_mn_c1_eta, [Nmn[0]**(-0.5), Nmn[1]*(-0.5) ]  )
    return S_mn_c1_eta

def func_Integrate_SmnEta1_SmlEta2_ArgExp(_m, _n, _l, _dmn_r_h1_vec, _dml_r_h2_vec):

    
    Sum_r_terms=np.zeros((2),dtype='float')
    R_terms=111
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

#%%

def func_Initiate_Matrices(content_Rad_SWF_hs, content_dr_values_hs, 
                       content_Rad_SWF_hw, content_dr_values_hw,
                       Theta_i_deg,
                       _m, Init_N_order, N_order_Fortran):
    
    Theta_s_deg=180-Theta_i_deg
    
    if _m ==0:
       Em=1
    else:
       Em=2
       
    C_Beta_D_Matrix_Arg=np.zeros((Init_N_order, Init_N_order), dtype = 'complex')
    C_Beta_D_Matrix_Exp=np.zeros((Init_N_order, Init_N_order), dtype = 'float')
    
    C_Beta_F_Mat_Arg = np.zeros((Init_N_order, Init_N_order), dtype = 'complex')
    C_Beta_F_Mat_Exp = np.zeros((Init_N_order, Init_N_order), dtype = 'float')
    
    Smn_hw_CosThetai_x_CosThetas_vec_Arg=np.zeros((1, Init_N_order), dtype = 'complex')
    Smn_hw_CosThetai_x_CosThetas_vec_Exp=np.zeros((1, Init_N_order), dtype = 'float')
    
    ll=-1
    for l in range(_m, Init_N_order+ _m): # l index Loop: l=m,1,2,...,N
        
        ## PSWF values ==================================================================================================
        R_ml_hs_kisi0=func_Rad_SWF_ArgExp(content_Rad_SWF_hs, _m, l , N_order_Fortran) #func_Adelman_Rmn(DirPSWF, AspRatio, m, l, hs, Prec) # Prolate spheroidal radial function of the first kind >>  R1(m,n,hw,kisi0)
        R1_ml_hs_kisi0=R_ml_hs_kisi0[0]
        dR1_ml_hs_kisi0=R_ml_hs_kisi0[1]

        dml_r_hs_ArgExp=func_dr_values_ArgExp(content_dr_values_hs, _m, l , N_order_Fortran) #func_Adelman_dr_mn_of_c(DirPSWF, m, l, hs, Prec)

        
        nn=-1
        ll=ll+1
        
        for n in range(_m, Init_N_order+ _m):  # n index Loop: n=m,1,2,...,N
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
           
#                print('m,n,l,F_nl_m : ',m,n,l, F_nl_m )
     
#                print('m,l,n >> ',m,l,n, C_n_m, Beta_nl_m, D_nl_m, F_nl_m)
            # Filling the l_th column
#               #   C_Beta_D_Matrix[nn][ll]=C_n_m*Beta_nl_m*D_nl_m
            TEMPvar=Multi_ArgExp(C_n_m, Multi_ArgExp(Beta_nl_m, D_nl_m))
            C_Beta_D_Matrix_Arg[nn][ll]=TEMPvar[0]
            C_Beta_D_Matrix_Exp[nn][ll]=TEMPvar[1]
            
            
            # Summation for the C_Beta_F vector
            #  C_Beta_F_Vector[0][ll]=C_Beta_F_Vector[0,ll]+(-1*C_n_m)*Beta_nl_m*F_nl_m
#                Minus_C_n_m=Multi_ArgExp([-1,0], C_n_m)
#                Beta_nl_m_x_F_nl_m=Multi_ArgExp(Beta_nl_m, F_nl_m)
#                Summation_C_Beta_F=Sum_ArgExp(Summation_C_Beta_F, Multi_ArgExp(Minus_C_n_m, Beta_nl_m_x_F_nl_m))
            
           
            # We will save C_Beta_F_Mat[nn][ll] and then summation of column-wise summation provides C_Beta_F_Vector
            Minus_C_n_m=Multi_ArgExp([-1,0], C_n_m)
            Beta_nl_m_x_F_nl_m=Multi_ArgExp(Beta_nl_m, F_nl_m)
            TEMPvar = Multi_ArgExp(Minus_C_n_m, Beta_nl_m_x_F_nl_m)
            C_Beta_F_Mat_Arg[nn][ll] = TEMPvar[0]
            C_Beta_F_Mat_Exp[nn][ll] = TEMPvar[1]
            
#                if ll==0:
#                    Smn_hw_CosThetai_x_CosThetas_vec[0][nn]=(Em/Nmn_hw)*Smn_hw_eta_i*Smn_hw_eta_s*np.cos(m*np.pi)
            if ll==0:
                Em_x_Smn_hw_eta_i=Multi_ArgExp(Get_ArgExp(Em), Smn_hw_eta_i)
                Smn_hw_eta_s_X_cosmpi=Multi_ArgExp(Smn_hw_eta_s, Get_ArgExp(np.cos(_m*np.pi)))
                
                TEMPvar=Multi_ArgExp(Em_x_Smn_hw_eta_i, Smn_hw_eta_s_X_cosmpi)
                Smn_hw_CosThetai_x_CosThetas_vec_Arg[0][nn]=TEMPvar[0]
                Smn_hw_CosThetai_x_CosThetas_vec_Exp[0][nn]=TEMPvar[1]
                
        # Filling the l_th element of Force vector:
#            C_Beta_F_Vector_Arg[0][ll]=Summation_C_Beta_F[0]
#            C_Beta_F_Vector_Exp[0][ll]=Summation_C_Beta_F[1]
        
   
    
    return [C_Beta_D_Matrix_Arg, C_Beta_D_Matrix_Exp,
             C_Beta_F_Mat_Arg, C_Beta_F_Mat_Exp,
             Smn_hw_CosThetai_x_CosThetas_vec_Arg,
             Smn_hw_CosThetai_x_CosThetas_vec_Exp]
    
    

def func_Generate_Partition_Matrices(content_Rad_SWF_hs, content_dr_values_hs, 
                       content_Rad_SWF_hw, content_dr_values_hw,
                       Theta_i_deg,
                       _m, N_order_Fortran, N_order_Init, N_order_New):
    
    Theta_s_deg=180-Theta_i_deg
    
    if _m==0:
       Em=1
    else:
       Em=2
       
    Right_C_Beta_D_Matrix_Arg=np.zeros((N_order_Init, N_order_New - N_order_Init), dtype = 'complex')
    Right_C_Beta_D_Matrix_Exp=np.zeros((N_order_Init, N_order_New - N_order_Init), dtype = 'float')
    
    Right_C_Beta_F_Mat_Arg = np.zeros((N_order_Init, N_order_New - N_order_Init), dtype = 'complex')
    Right_C_Beta_F_Mat_Exp = np.zeros((N_order_Init, N_order_New - N_order_Init), dtype = 'float')
    
    Lower_C_Beta_D_Matrix_Arg=np.zeros((N_order_New - N_order_Init, N_order_New), dtype = 'complex')
    Lower_C_Beta_D_Matrix_Exp=np.zeros((N_order_New - N_order_Init, N_order_New), dtype = 'float')
    
    Lower_C_Beta_F_Mat_Arg = np.zeros((N_order_New - N_order_Init, N_order_New), dtype = 'complex')
    Lower_C_Beta_F_Mat_Exp = np.zeros((N_order_New - N_order_Init, N_order_New), dtype = 'float')
    
    Added_Smn_hw_CosThetai_x_CosThetas_vec_Arg=np.zeros((1, N_order_New - N_order_Init), dtype = 'complex')
    Added_Smn_hw_CosThetai_x_CosThetas_vec_Exp=np.zeros((1, N_order_New - N_order_Init), dtype = 'float')

    # This section calculates right partition *************************** 
    ll=-1
    for l in range(_m + N_order_Init, N_order_New + _m): # l index Loop: l=m,1,2,...,N : This fills the columns in Matrices
        
        ## PSWF values ==================================================================================================
        R_ml_hs_kisi0=func_Rad_SWF_ArgExp(content_Rad_SWF_hs, _m, l , N_order_Fortran) #func_Adelman_Rmn(DirPSWF, AspRatio, m, l, hs, Prec) # Prolate spheroidal radial function of the first kind >>  R1(m,n,hw,kisi0)
        R1_ml_hs_kisi0=R_ml_hs_kisi0[0]
        dR1_ml_hs_kisi0=R_ml_hs_kisi0[1]

        dml_r_hs_ArgExp=func_dr_values_ArgExp(content_dr_values_hs, _m, l , N_order_Fortran) #func_Adelman_dr_mn_of_c(DirPSWF, m, l, hs, Prec)

        
        nn=-1
        ll=ll+1
        
        for n in range(_m , N_order_Init + _m):  # n index Loop: n=m,1,2,...,N : This fills the rows in Matrices
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
            
#            eta_s=np.cos(np.pi*Theta_s_deg/180)
#            Smn_hw_eta_s=func_Smn_eta_c_from_dr_ArgExp(_m, n, eta_s, dmn_r_hw_ArgExp)
            
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
            Right_C_Beta_D_Matrix_Arg[nn][ll]=TEMPvar[0]
            Right_C_Beta_D_Matrix_Exp[nn][ll]=TEMPvar[1]
            
           
            # We will save C_Beta_F_Mat[nn][ll] and then summation of column-wise summation provides C_Beta_F_Vector
            Minus_C_n_m=Multi_ArgExp([-1,0], C_n_m)
            Beta_nl_m_x_F_nl_m=Multi_ArgExp(Beta_nl_m, F_nl_m)
            TEMPvar = Multi_ArgExp(Minus_C_n_m, Beta_nl_m_x_F_nl_m)
            Right_C_Beta_F_Mat_Arg[nn][ll] = TEMPvar[0]
            Right_C_Beta_F_Mat_Exp[nn][ll] = TEMPvar[1]
            

    # This section calculates lower partition *************************** 
    ll=-1
    for l in range(_m, N_order_New + _m): # l index Loop: l=m,1,2,...,N: This fills the columns in Matrices
        
        ## PSWF values ==================================================================================================
        R_ml_hs_kisi0=func_Rad_SWF_ArgExp(content_Rad_SWF_hs, _m, l , N_order_Fortran) #func_Adelman_Rmn(DirPSWF, AspRatio, m, l, hs, Prec) # Prolate spheroidal radial function of the first kind >>  R1(m,n,hw,kisi0)
        R1_ml_hs_kisi0=R_ml_hs_kisi0[0]
        dR1_ml_hs_kisi0=R_ml_hs_kisi0[1]

        dml_r_hs_ArgExp=func_dr_values_ArgExp(content_dr_values_hs, _m, l , N_order_Fortran) #func_Adelman_dr_mn_of_c(DirPSWF, m, l, hs, Prec)

        
        nn=-1
        ll=ll+1
        
        for n in range(_m + N_order_Init, N_order_New + _m):  # n index Loop: n=m,1,2,...,N : This fills the rows in Matrices
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
            Lower_C_Beta_D_Matrix_Arg[nn][ll]=TEMPvar[0]
            Lower_C_Beta_D_Matrix_Exp[nn][ll]=TEMPvar[1]
            
           
            # We will save C_Beta_F_Mat[nn][ll] and then summation of column-wise summation provides C_Beta_F_Vector
            Minus_C_n_m=Multi_ArgExp([-1,0], C_n_m)
            Beta_nl_m_x_F_nl_m=Multi_ArgExp(Beta_nl_m, F_nl_m)
            TEMPvar = Multi_ArgExp(Minus_C_n_m, Beta_nl_m_x_F_nl_m)
            Lower_C_Beta_F_Mat_Arg[nn][ll] = TEMPvar[0]
            Lower_C_Beta_F_Mat_Exp[nn][ll] = TEMPvar[1]
            
            if ll==0:
                Em_x_Smn_hw_eta_i=Multi_ArgExp(Get_ArgExp(Em), Smn_hw_eta_i)
                Smn_hw_eta_s_X_cosmpi=Multi_ArgExp(Smn_hw_eta_s, Get_ArgExp(np.cos(m*np.pi)))
                
                TEMPvar=Multi_ArgExp(Em_x_Smn_hw_eta_i, Smn_hw_eta_s_X_cosmpi)
                Added_Smn_hw_CosThetai_x_CosThetas_vec_Arg[0][nn]=TEMPvar[0]
                Added_Smn_hw_CosThetai_x_CosThetas_vec_Exp[0][nn]=TEMPvar[1]
            
    return [Right_C_Beta_D_Matrix_Arg,
            Right_C_Beta_D_Matrix_Exp,
            Right_C_Beta_F_Mat_Arg,
            Right_C_Beta_F_Mat_Exp,
            Lower_C_Beta_D_Matrix_Arg,
            Lower_C_Beta_D_Matrix_Exp,
            Lower_C_Beta_F_Mat_Arg,
            Lower_C_Beta_F_Mat_Exp,
            Added_Smn_hw_CosThetai_x_CosThetas_vec_Arg,
            Added_Smn_hw_CosThetai_x_CosThetas_vec_Exp] 