# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:40:06 2020

@author: smill

https://www.youtube.com/watch?v=SKq-pmkekTk&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    """
    param xx : xx
    return : xx : xx
    -------
    """    
    Lesson_2()    

# ----------------------------
# keyword(s): Lesson_1
def Lesson_1():
    """
    param n/a : n/a
    return n/a : n/a
    -------
    """    
    print(torch.__version__)    

# ----------------------------
# keyword(s): Lesson_2
def Lesson_2():
    """
    param w : weight value
    param x : input value    
    param y : given output value
    param y_pred : predicted output value 
    return :  loss value
    -------
    """    
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w = 1.0    
    
    def forward(x):
        """   
        return :  predicted output value
        -------
        """    
        return x * w    
    
    def loss(x, y):
        """   
        return :  loss value
        -------
        """        
        y_pred = forward(x)
        return (y_pred - y)**2    
    
    w_list = []
    mse_list = []    
    for w in np.arange(0.0, 4.1, 0.1):
        print ("w=", w)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            l = loss(x_val, y_val)
            l_sum += l
            print("/t", x_val, y_val, y_pred_val, l)
        print("MSE=",l_sum / 3)
        w_list.append(w)
        mse_list.append(l_sum / 3)
    plt.plot(w_list, mse_list)
    plt.ylabel("Loss")
    plt.xlabel("w")
    plt.show()    
    
if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()