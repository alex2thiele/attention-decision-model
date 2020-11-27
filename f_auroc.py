'''

calculates AUROC

@author: Alex Thiele

alex.thiele@ncl.ac.uk

returns auroc

'''

import numpy
import random as pyrandom
import array as arr
from numpy.random import rand as rand
from numpy.random import randn as randn
from scipy.signal import lfilter
from matplotlib import pyplot as plt


def make_f_auroc(arr1,arr2): 	
#arr1=[12.0, 21.0, 23.0, 6.0, 29.0, 52.0, 42.0, 6.0, 51.0, 12.0]
#arr2=[36.0, 27.0, 39.0, 15.0, 41.0, 23.0, 7.0, 15.0, 28.0, 26.0]
    nb = 500
    nu1 = len(arr1)
    nu2 = len(arr2)


    maxr = max([max(arr1), max(arr2)])
    minr = min([min(arr1), min(arr2)])
 #print(minr)
#print(maxr)
    ib1=minr-(maxr-minr)/nb
#print(ib1)
    ib2=maxr+(maxr-minr)/nb
    #print(ib2)
    ib = numpy.linspace(ib1,  ib2,nb)
    #ib = numpy.linspace(0, 1,100)
    #print(ib)
    [n1,bins] = numpy.histogram(arr1,ib)#/nu1#probability distr of 1
    [n2,bins] = numpy.histogram(arr2,ib)#/nu2#probability distr of 2
    n1=arr.array('d',n1)
    n2=arr.array('d',n2)
    for x in range(0,len(n1)):
        n1[x]=n1[x]/nu1
        #print(n1[x])
    for x in range(0,len(n2)):
        n2[x]=n2[x]/nu1
        #print(n2[x])
    
    
    
    #print(n22)
    mk = len(n1)
    auroc0 =0.0;
    if mk>0:
        for jj in range (0,mk):
            sumN2=0
            for kk in range(jj+1,mk):
                #print(kk)
                sumN2=sumN2+n2[kk]
            auroc0 = auroc0 + n1[jj]*sumN2+(n1[jj]*n2[jj])/2; #the term (n1(jj)*n2(jj))/2 is to account for when the bin for 1 and 2 are the same
    
    if mk==0:
        auroc0=0.5;
    print(auroc0)
    return auroc0    