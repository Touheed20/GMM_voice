import numpy as np
import math
from decimal import*
from mpmath import*

e = math.exp(1)
NoM = 8

def InitializeGamma():

    gamma = np.zeros(shape=(8,X.shape[0]),dtype='float128')
    gamma = np.random.uniform(0,1,8*X.shape[0]).reshape(8,X.shape[0])
    summ = np.sum(gamma,axis=0)
    print summ.shape
##    print gamma[0]
    gamma = np.divide(gamma,summ)
##    print gamma[0]
##    print gamma[1]
    return gamma

def CalcGamma(MeanVec,VarVec,wVec,X):

    gamma = np.zeros(shape=(8,X.shape[0]))
    for a in range(X.shape[0]):
        summ = 0
        for i in range(NoM):
            power = np.square(np.subtract(X[a],MeanVec[i]))
##            print 'poweris \n',power
##            print 'v o\n',VarVec[i]
            denp = np.multiply(-2,VarVec[i])
            if(denp.any==0):
                denp = np.add(denp,0.000001)
##            print denp
            power = np.divide(power,denp)
##            print power
            power = np.sum(power)
##            print 'power is \n',power

            power = math.exp(power)
            #print power
            prodVarVec = np.prod(VarVec[i])
            den = np.float128(1.0)
            den = 1/(2*math.pi)**(NoM/2)*math.sqrt(prodVarVec)
            if(den==0):
                den = den+0.0000001
##            print den
            gamma[i][a] = wVec[i]*np.divide(power,den)
##            print gamma
            summ = summ + gamma[i][a]
##            print gamma[i][a]
        for i in range(NoM):
            gamma[i][a] = gamma[i][a]/summ
        
    return gamma

def Calc_N_m(gamma,X):

    N_m = np.zeros(shape=(8))

    for i in range(8):
        N_m[i] = np.sum(gamma,axis=1)[i]

    return N_m

def Calc_Weights(N_m,N,wVec):

    wVec = np.divide(N_m,N)

    return wVec
    
def CalcMean(gamma,X,MeanVec):

    MeanVec = np.matrix(gamma)*np.matrix(X)
    
    for i in range(NoM):
        den = np.sum(gamma,axis=1)[i]
        MeanVec[i] = np.divide(MeanVec[i],den)

    return MeanVec

def CalcVariance(gamma,X,MeanVec,VarVec):

    for i in range(NoM):
        num = np.square(np.subtract(X,MeanVec[i]))
        num = np.matrix(gamma[i])*np.matrix(num)
        den = np.sum(gamma,axis=1)[i]
        VarVec[i] = np.divide(num,den)

    return VarVec

def Likelihood(MeanVec,VarVec,wVec,X):
    
    LLH = 0
    for a in range(X.shape[0]):
        summ = 0
        for i in range(NoM):
            power = np.square(np.subtract(X[a],MeanVec[i]))
##            print 'poweris \n',power
##            print 'v o\n',VarVec[i]
            denp = np.multiply(-2,VarVec[i])
##            print denp
            power = np.divide(power,denp)
##            print power
            power = np.sum(power)
##            print 'power is \n',power

            power = exp(power)
##            print power
            prodVarVec = np.prod(VarVec[i])
            den = 1/(2*math.pi)**(NoM/2)*np.sqrt(prodVarVec)
##            print den
            sigma = wVec[i]*np.divide(power,den)
##            print gamma
            summ = summ + sigma
##            print gamma[i][a]
        LLH = LLH + math.log(summ)
        
    return LLH

if __name__ == "__main__":

    try1 = [[1,2,3,4],[1,2,5,7]]

    trainfiles = ['faem0_d.csv','fajw0_d.csv','fpas0_d.csv','fkfb0_d.csv','fsah0_d.csv','megj0_d.csv','mfxv0_d.csv','mklr0_d.csv','mkls0_d.csv','mpgl0_d.csv',]

    G_MeanVec = np.zeros(shape=(10,NoM,60))
    G_VarVec = np.ones(shape=(10,NoM,60))
    G_wVec = np.ones(shape=(10,NoM))
    
    Mean = np.genfromtxt('UBMMean.txt',delimiter = ',')
    Var = np.genfromtxt('UBMVar.txt',delimiter = ',')
    Wgt = np.genfromtxt('UBMWgt.txt',delimiter = ',')

    UBM_MeanVec = np.reshape(Mean,(8,60))
    UBM_VarVec = np.reshape(Var,(8,60))
    UBM_wVec = np.reshape(Wgt,(8))
    

    for person in range(10):
        X =  np.genfromtxt(trainfiles[person])
        MeanVec = np.copy(UBM_MeanVec)
        VarVec = np.copy(UBM_VarVec)
        wVec = np.copy(UBM_wVec)
        
        N = X.shape[0]
        #gamma = InitializeGamma()
        

        for i in range(10):
            gamma = CalcGamma(MeanVec,VarVec,wVec,X)
            N_m = Calc_N_m(gamma,X)
            MeanVec = CalcMean(gamma,X,MeanVec)
            VarVec = CalcVariance(gamma,X,MeanVec,VarVec)
            wVec = Calc_Weights(N_m,N,wVec)
            
        print np.sum(gamma,axis=0)  
##            LLH = Likelihood(MeanVec,VarVec,wVec,X)
##            print LLH
##            print wVec

        G_MeanVec[person] = MeanVec
        G_VarVec[person] = VarVec
        G_wVec[person] = wVec

    G_MeanVec = np.asarray(G_MeanVec)
    G_VarVec = np.asarray(G_VarVec)
    G_wVec = np.asarray(G_wVec)
    G_MeanVec.tofile('UBM_Ad_Mean.txt',sep=',',format='%10.5f')
    G_VarVec.tofile('UBM_Ad_Var.txt',sep=',',format='%10.5f')
    G_wVec.tofile('UBM_Ad_Wgt.txt',sep=',',format='%10.5f')



    
        

