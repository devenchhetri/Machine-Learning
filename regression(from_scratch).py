import numpy as np                                                                  #importing required packages and modules
from numpy import mean
import matplotlib.pyplot as plt
import random

def create_data(count, variance, step, corr=False):                                 #function to generate random data
    value = 1
    y_vals = []
    for i in range(count):
        y = value + random.randrange(-variance, variance)                           #using randrange function to generate y values
        y_vals.append(y)
        if corr and corr == 'pos':                                                  #checking if correlation is set to positive or negative
            value += step
        elif corr and corr == 'neg':
            value -= step
    x_vals = [i for i in range(len(y_vals))]                                        #setting x to 1 to len(y)

    return np.array(x_vals, dtype=np.float64), np.array(y_vals, dtype=np.float64)

def slope_intercept(x_vals, y_vals):                                                #function to calculate slope and intercept
    num = (mean(x_vals) * mean(y_vals)) - mean(x_vals * y_vals)
    den = ((mean(x_vals)) * (mean(x_vals))) - mean(x_vals*x_vals)
    m = num/den

    c = mean(y_vals) - m*mean(x_vals)
    return m, c

def coef(y_orig, y_line):                                                            #function to calclualte r-square value
    y_mean_line = [mean(y_orig) for y in y_orig]                                     #higher  r-square value is better, meaning the regression line fits data well
    sq_err_y = sum((y_mean_line-y_orig)**2)
    sq_err_regr = sum((y_line-y_orig)**2)
    return 1-(sq_err_regr/sq_err_y)

x_vals, y_vals = create_data(50, 30, 3, corr='pos')                                  #calling function to create data

m, c = slope_intercept(x_vals, y_vals)                                               #calling function to calculate m and c values (y=mx+c)

model_line = [(m*x + c) for x in x_vals]                                             #regression line

x_new = 8
y_new = (m*x_new) + c                                                                #checking for a new single point

r_sq = coef(y_vals, model_line)                                                      #calculating r-square values to see accuracy
print(r_sq)

plt.scatter(x_vals, y_vals)                                                          #plotting data points and modeled regression line with matplotlib
plt.scatter(x_new, y_new, color='r')
plt.plot(x_vals, model_line)
plt.show()

