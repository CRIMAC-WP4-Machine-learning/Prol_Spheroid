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