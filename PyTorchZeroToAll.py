# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:40:06 2020

@author: smill

https://www.youtube.com/watch?v=SKq-pmkekTk&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

def main():
    """
    param xx : xx
    return : xx : xx
    -------
    """    
    Lesson_4()    

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
# keyword(s): Lesson_4
def Lesson_4():
    """
    param w : weight value
    param x : input value    
    param y : given output value
    param y_pred : predicted output value 
    return :  n/a
    -------
    """    
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w = Variable(torch.Tensor([1.0]), requires_grad=True)
    
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
   
    # Before training
    print("Prediction (before training)", "4 hours of study:", forward(4))    
    
    # Training loop
    for epoch in range(10):
        for x_val, y_val in zip(x_data, y_data):
            l = loss(x_val, y_val)
            l.backward()
            print("\tgrad: ", x_val, y_val, w.grad.data[0])
            w.data -= 0.01 * w.grad.data
            # manually zero gradiaents afer updating weights
            w.grad.data.zero_()
            
    print("Progress:", epoch, l.data[0])
    
    # After training
    print("Prediction (after training)", "4 hours of study:", forward(4))    
    
if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()