#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division,absolute_import
import os
from optparse import OptionParser
from influ.F2 import F2
from influ.DMD import DMD
from influ.utils import *
from influ.Kalman import Kalman
import time
from tqdm import tqdm
import pandas as pd

def main():
    """ Infer from command line"""
    parser = OptionParser(conflict_handler="resolve")
    
    #specify model  
    parser.add_option('--method', type='string', default = 'LS', dest='method' ,help='specify model: LS, TLS, F2, EM')    
    parser.add_option('-i', '--infile', dest = 'infile_name',metavar='PATH/TO/FILE', help='read in counts and totcounts information')
    parser.add_option('-o', '--outfile', dest = 'outfile_name', metavar='PATH/TO/FILE', help='write results of the inference')
    parser.add_option('--seed', type='int',metavar='N', dest='seed', default = None, help='set seed for inference')
    (options, args) = parser.parse_args()

    #set seed
    if options.seed is not None: np.random.seed(options.seed)

    #Check that the method is specified properly
    methods=['LS', 'TLS', 'F2', 'EM']
    assert options.method in methods
    
    #load file
    file=pd.read_csv(options.infile_name)
    
    # convert to counts, tot_counts arrays
    times=file.time.max()+1
    lineages=file.lineage.max()+1
    nd=file.dimension.max()+1
    counts=np.zeros((nd,lineages,times))
    tot_counts=np.zeros((nd,lineages,times))
    for i in range(nd):
        df=file.loc[file.dimension==i]
        for t in range(times):
            df_=df.loc[df.time==t]
            assert all((df_.lineage==np.arange(lineages)).values)
            counts[i,:,t]=df_.counts.values
            tot_counts[i,:,t]=df_.tot_counts.values
            
    we=0.9 # for now hardcode it      
    # initialize classes
    if options.method=='F2':
        method=F2(counts,tot_counts)
        method.infer()
    elif options.method=='TLS':
        method=DMD(counts,tot_counts)
        method.infer(we=we)
    elif options.method=='LS':
        method=DMD(counts,tot_counts)
        method.infer()
    elif options.method=='EM':
        method=Kalman(counts+1,tot_counts)
        method.infer()
    print(method.A)
            
if __name__ == '__main__': main()
