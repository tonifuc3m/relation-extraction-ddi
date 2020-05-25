#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:34:10 2020

@author: antonio
Utils
"""
import argparse
import itertools

def combine_params(param, indexes):
    combinations = list(itertools.product(*param))
    param_combinations = [{k:v for k, v in zip(indexes, combination)} for combination in combinations]
    
    # If there is not fc layer, set its size and dropout to zero
    if 'fc' in indexes:
        for p in param_combinations:
            if p["fc"]==False:
                p["fc_size"] = 0
                p["fc_d"] = 0
        
    # remove duplicates
    return [dict(t) for t in {tuple(d.items()) for d in param_combinations}]


def argparser():
    '''
    DESCRIPTION: Parse command line arguments
    '''
    
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.add_argument("-i", "--input", required = True, dest = "input", 
                        help = "absolute path to input directory")
    parser.add_argument("-o", "--output", required = True, dest = "output", 
                        help = "absolute path to output directory")
    parser.add_argument("-f", "--outfig", required = False, default='class-weights.png', dest = "outfig", 
                        help = "absolute path to output class weight figure")
    args = parser.parse_args()
    
    in_path = args.input
    out_path = args.output
    out_path_figure = args.outfig
    
    return in_path, out_path, out_path_figure

