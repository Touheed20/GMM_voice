import numpy as np
import math
from mpmath import*

NoM = 8

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
##            print 'summ is', summ
        LLH = LLH + log(summ)
        
    return LLH

if __name__ == "__main__":

    LLH = np.zeros(10)
    Mean = np.genfromtxt('Split_Mean.txt',delimiter = ',')
    Var = np.genfromtxt('Split_Var.txt',delimiter = ',')
    Wgt = np.genfromtxt('Split_Wgt.txt',delimiter = ',')
    print np.shape(Mean)

    pass_no = 0
    fail_no = 0

    testcases = ['faem0_t1.csv','faem0_t2.csv','faem0_t3.csv','fajw0_t1.csv',
                 'fajw0_t2.csv','fajw0_t3.csv','fpas0_t1.csv','fpas0_t2.csv',
                 'fpas0_t3.csv','fkfb0_t1.csv','fkfb0_t2.csv','fkfb0_t3.csv',
                 'fsah0_t1.csv','fsah0_t2.csv','fsah0_t3.csv','megj0_t1.csv',
                 'megj0_t2.csv','megj0_t3.csv','mfxv0_t1.csv','mfxv0_t2.csv',
                 'mfxv0_t3.csv','mklr0_t1.csv','mklr0_t2.csv','mklr0_t3.csv',
                 'mkls0_t1.csv','mkls0_t2.csv','mkls0_t3.csv','mpgl0_t1.csv',
                 'mpgl0_t2.csv','mpgl0_t3.csv']

    Mean = np.reshape(Mean,(10,8,60))
    Var = np.reshape(Var,(10,8,60))
    Wgt = np.reshape(Wgt,(10,8))
    
    for test in range(len(testcases)): 

        X = np.genfromtxt(testcases[test])
        
        for i in range(10):
            MeanVec = np.copy(Mean[i])
            VarVec = np.copy(Var[i])
            wVec = np.copy(Wgt[i])

            LLH[i] = Likelihood(MeanVec,VarVec,wVec,X)
            print LLH[i]
            
        print 'least iss ',np.argmax(LLH)   
        if(np.argmax(LLH)==test/3):
            pass_no=pass_no+1
            print'pass'
        else:
            fail_no=fail_no+1
            print 'fail'

    print 'total number of test cases are 30'
    print 'number of test cases passed : ',pass_no
    print 'number of test cases failed : ',fail_no
    print 'there'

