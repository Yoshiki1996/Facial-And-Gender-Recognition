'''This project was mediated through Michael Guerzhoy and is not 
to be copy and used for educational purposes which can lead to Academic Integrity'''

# Importing Modules
from pylab import *
from _collections import defaultdict

# Numpy Modules
import numpy as np
from numpy.linalg import norm

# Matplotlib Modules
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg

# Scipy Modules
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
from scipy.misc import imsave

# Modules for reading images and for gradient creation
import random
import time
import os
import urllib

def rgb2gray(rgb):
    
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

# Two actors used for linear-regression using one decision boundary
act = ['Bill Hader','Steve Carell'] 

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

# Empty arrays for TRAINING,TEST and VALIDATION 
TRAININGSET = {} 
TRAININGSET2 = []
TESTSET = {}
TESTSET2 = []
VALIDATIONSET = {}
VALIDATIONSET2 = []

for a in act:
    
    name = a.split()[1].lower()
    i = 0
    TRAININGSET2.append([])
    VALIDATIONSET2.append([]) 
    TESTSET2.append([]) 
    j = len(TRAININGSET2) - 1 
    
    for line in open("faces_subset.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
    
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            #print(line.split()[4])

            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+filename):
                continue 
            
            try: 
                im = imread("uncropped/"+filename) 
                k = line.split()[5]
                k2 = k.replace(',',' ')
            
                ''' cropping the image ''' 
           
            
                im = im[int(k2.split()[1]):int(k2.split()[3]),int(k2.split()[0]):int(k2.split()[2])]
                im = imresize(im,(32,32))
                im = rgb2gray(im)
            
                imsave("cropped/"+filename,im)
                
                ### MAKING OF TRAINING,TEST AND VAL. SET ###
            
                # TRAINING SET: (100 images per actor)
           
                if (i < 100):
                    
                    TRAININGSET['filename'] = im
                    TRAININGSET2[j] = TRAININGSET2[j] + [np.reshape(TRAININGSET['filename'],(1,1024))]
                    
                # VALIDATION SET: (100 images per actor: No image is the same as the Training set)
                elif (100 < i < 111):
                    VALIDATIONSET['filename'] = im
                    VALIDATIONSET2[j] = VALIDATIONSET2[j] + [np.reshape(VALIDATIONSET['filename'],(1,1024))]
        
                
                # TEST SET: (100 images per actor: No image is the same as Training and Val set)
                elif (111 < i < 122):
                    TESTSET['filename'] = im 
                    TESTSET2[j] = TESTSET2[j] + [np.reshape(TESTSET['filename'],(1,1024))]
                    
                else:
                    pass 
            
                print(filename)
                #print(i) 
                #print('TRAININGSET'+ ' ' + name)
                i += 1
            except:
                pass

###
# Making of x and y and initial theta and alpha values

x_temp = []
x_tempVAL = []
x_tempTEST = [] 
y_temp1 = np.ones((100,1))
y_temp2 = np.zeros((100,1))
x_p3 = np.array((200,1024))
x_p3VAL=np.array((20,1024))
x_p3TEST=np.array((20,1024)) 

y_p3 = np.array((200,1)) 
y_p3 = np.vstack((y_temp1,y_temp2))

theta0 = np.zeros((1,1025))
h = 1e-4

for i in range(0,len(TRAININGSET2)):
    for j in range(0,len(TRAININGSET2[i])):
        x_temp = x_temp + [TRAININGSET2[i][j]]
        #print(x_temp) 

for i in range(0,len(VALIDATIONSET2)):
    for j in range(0,len(VALIDATIONSET2[i])):
        x_tempVAL = x_tempVAL + [VALIDATIONSET2[i][j]]
        #print(x_tempVAL) 

for i in range(0,len(TESTSET2)):
    for j in range(0,len(TESTSET2[i])):
        x_tempTEST = x_tempTEST + [TESTSET2[i][j]]
        #print(x_tempTEST) 


#Matrix of size(200,1024) containing the elements of the images 
x_p3 = np.vstack(x_temp) 
x_p3VAL = np.vstack(x_tempVAL)
x_p3TEST = np.vstack(x_tempTEST)
x_ones = np.ones((200,1))
x_onesVAL = np.ones((20,1)) 
x_onesTEST = np.ones((20,1)) 
x_p3 = np.hstack((x_ones,x_p3))
x_p3VAL = np.hstack((x_onesVAL,x_p3VAL)) 
x_p3TEST = np.hstack((x_onesTEST,x_p3TEST))
# Confirming if dimensions is what is needed
# print(x_p3)
# print(x_p3.shape)
# print(x_p3VAL) 
# print(x_p3VAL.shape)
# print(x_p3TEST)
# print(x_p3TEST.shape)
###

#PART 3: COMPARISON BETWEEN BILL HADER AND STEVE CAROLL
def f(x, y, theta):
    return (sum( (y.T - dot(theta,x.T)) ** 2)/(2*len(x)))

def df(x, y, theta):
    return -(1/200.)*sum((y.T-dot(theta, x.T))*x.T, 1)

###        
#Using The Derivative of the Cost Function to Evaluate Grad. Descent To evaluate Theta.
#J(theta0,...,thetaN) - alpha*grad(J(theta0,...,thetaN)) 
###

def grad_descent(df, x, y, init_t, alpha):
    '''Evaulating Gradient Descent''' 
    EPS = 1e-10 #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        #print "Iter", iter
        #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
        #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

def classifier(act,x,y,init_t,alpha,df,grad_descent,x_set):
    '''what we want to do is compute the value of the percentage between 
       how well the theta was tuned in order for us to obtain the proper actor. 
       if the hyptothesis gives a value greater than 0.5, it will be equal to an 
       output of one. Otherwise, it will be zero.'''
       
    a = 0
    b = 0 
    theta = grad_descent(df, x, y, init_t, alpha)
    for i in range(0,len(x_set)):
        if i < (len(x_set)/2) and float(dot(theta,x_set[i].T)) > 0.5:
            a += 1 
        else:
            if i > (len(x_set)/2) and float(dot(theta,x_set[i].T)) < 0.5:
                b += 1
    print(act[0], float(a)/(len(x_set)/2))
    print(act[1], float(b)/(len(x_set)/2))
    return
    
#NOTE: UNCOMMENT THE SECTIONS BELOW TO OBTAIN THE RESULTS FOR THE VALIDATION AND TEST SETS. 

# classifier(act,x_p3,y_p3,theta0,h,df,grad_descent,x_p3VAL)
# classifier(act,x,y,init_t,alpha,df,grad_descent,x_p3TEST)
    
# #PART 4:
# #As we already have the theta's stored through the Grad.Descent Function, we will assign the variable 
# #And save the image within the working directory folder named part 4 
# t_100 = grad_descent(df,x_p3,y_p3,theta0,h)
# t_100 = np.delete(t_100,0)
# t_100 = np.reshape(t_100,(32,32)) 
# imsave("part4/"+filename,t_100) 
# #To create the array which contains the four images, two per actor, we will assign it to a
# #variable called t_2 
# t_2 = np.array((4,1025))
# y_p3_2 = np.array((4,1)) 
# t_2 = np.vstack((x_p3[0],x_p3[1],x_p3[100],x_p3[101])) 
# y_temp1_2 = np.ones((2,1))
# y_temp2_2 = np.zeros((2,1))
# y_p3_2 = np.vstack((y_temp1_2,y_temp2_2))
# t_2 = grad_descent(df,t_2,y_p3_2,theta0,h)
# t_2 = np.delete(t_2,0) 
# t_2 = np.reshape(t_2,(32,32)) 
# imsave("part4/"+filename,t_2) 

# PART 5: Demonstration of overfitting 
# First three actors are female (assign value y =1). The remaining three actors are of male 
# assign values y=0 

actP5 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']


TRAININGSET_P5 = {} 
TRAININGSET2_P5 = []
TESTSET_P5 = {}
TESTSET2_P5 = []
VALIDATIONSET_P5 = {}
VALIDATIONSET2_P5 = []

for a in actP5:
    
    name = a.split()[1].lower()
    i = 0
    TRAININGSET2_P5.append([])
    VALIDATIONSET2_P5.append([]) 
    TESTSET2_P5.append([]) 
    j = len(TRAININGSET2_P5) - 1 
    
    for line in open("faces_subset.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
    
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            #print(line.split()[4])

            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+filename):
                continue 
            
            try: 
                im = imread("uncropped/"+filename) 
                k = line.split()[5]
                k2 = k.replace(',',' ')
            
                ''' cropping the image ''' 
           
            
                im = im[int(k2.split()[1]):int(k2.split()[3]),int(k2.split()[0]):int(k2.split()[2])]
                im = imresize(im,(32,32))
                im = rgb2gray(im)
            
                ### PART 2: MAKING OF TRAINING,TEST AND VAL. SET ###
            
                # TRAINING SET: (100 images per actor)
           
                if (i < 100):
                    TRAININGSET_P5['filename'] = im
                    TRAININGSET2_P5[j] = TRAININGSET2_P5[j] + [np.reshape(TRAININGSET_P5['filename'],(1,1024))]

                    
                # VALIDATION SET: (100 images per actor: (No image is the same as the Training set) 
                elif (100 < i < 111):
                    VALIDATIONSET_P5['filename'] = im
                    VALIDATIONSET2_P5[j] = VALIDATIONSET2_P5[j] + [np.reshape(VALIDATIONSET_P5['filename'],(1,1024))]
                    #print(VALIDATIONSET2[j]) 
                
                # TESTSET: (100 images per actor: No image is the same as Training and Val set)
                
                elif (111 < i < 122):
                    TESTSET_P5['filename'] = im 
                    TESTSET2_P5[j] = TESTSET2_P5[j] + [np.reshape(TESTSET_P5['filename'],(1,1024))]
                    
                else:
                    pass 
            
                print(filename)
                i += 1
                
            except:
                 pass

#making of x and y and initial theta and alpha values
x_temp_P5 = []
x_tempVAL_P5 = []
x_tempTEST_P5 = [] 
y_temp1_P5 = np.ones((300,1))
y_temp2_P5 = np.zeros((300,1))
x_p5= np.array((600,1024))
x_p5VAL=np.array((60,1024))
x_p5TEST=np.array((60,1024)) 

y_p5 = np.array((600,1)) 
y_p5 = np.vstack((y_temp1_P5,y_temp2_P5))

theta0 = np.zeros((1,1025))
h = 1e-4

for i in range(0,len(TRAININGSET2_P5)):
    for j in range(0,len(TRAININGSET2_P5[i])):
        x_temp_P5 = x_temp_P5 + [TRAININGSET2_P5[i][j]]
        #print(x_temp) 

for i in range(0,len(VALIDATIONSET2_P5)):
    for j in range(0,len(VALIDATIONSET2_P5[i])):
        x_tempVAL_P5 = x_tempVAL_P5 + [VALIDATIONSET2_P5[i][j]]
        #print(x_tempVAL) 

for i in range(0,len(TESTSET2_P5)):
    for j in range(0,len(TESTSET2_P5[i])):
        x_tempTEST_P5 = x_tempTEST_P5 + [TESTSET2_P5[i][j]]
        #print(x_tempTEST) 


#Matrix of size(200,1024) containing the elements of the images 
x_p5 = np.vstack(x_temp_P5) 
x_p5VAL = np.vstack(x_tempVAL_P5)
x_p5TEST = np.vstack(x_tempTEST_P5)
x_ones_P5 = np.ones((600,1))
x_onesVAL_P5 = np.ones((60,1)) 
x_onesTEST_P5 = np.ones((60,1)) 
x_p5 = np.hstack((x_ones_P5,x_p5))
x_p5VAL = np.hstack((x_onesVAL_P5,x_p5VAL)) 
x_p5TEST = np.hstack((x_onesTEST_P5,x_p5TEST))
print(x_p5)
print(x_p5.shape)
print(x_p5VAL) 
print(x_p5VAL.shape)
print(x_p5TEST)
print(x_p5TEST.shape)

#For act_test
TRAININGSET_P5_2 = {} 
TRAININGSET2_P5_2 = []
for a in act_test:
    name = a.split()[1].lower()
    #print(a)
    #print(name)
    i = 0
    TRAININGSET2_P5_2.append([])
    j = len(TRAININGSET2_P5_2) - 1 
    for line in open("faces_subsetP5.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
    
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            #print(line.split()[4])

            timeout(testfile.retrieve, (line.split()[4], "uncropped2/"+filename), {}, 30)
            if not os.path.isfile("uncropped2/"+filename):
                continue 
            
            try: 
                im = imread("uncropped2/"+filename) 
                k = line.split()[5]
                k2 = k.replace(',',' ')
            
                ''' cropping the image ''' 
           
            
                im = im[int(k2.split()[1]):int(k2.split()[3]),int(k2.split()[0]):int(k2.split()[2])]
                im = imresize(im,(32,32))
                im = rgb2gray(im)
           
                if (i < 100):
                    TRAININGSET_P5_2['filename'] = im
                    TRAININGSET2_P5_2[j] = TRAININGSET2_P5_2[j] + [np.reshape(TRAININGSET_P5_2['filename'],(1,1024))]
                    #print(TRAININGSET2[j]) 
                else:
                    pass 
            
                print(filename)

                i += 1
            except:
                 pass

x_temp_P5_2 = []
x_p5_2= np.array((600,1024))

for i in range(0,len(TRAININGSET2_P5_2)):
    for j in range(0,len(TRAININGSET2_P5_2[i])):
        x_temp_P5_2 = x_temp_P5_2 + [TRAININGSET2_P5_2[i][j]]

x_p5_2 = np.vstack(x_temp_P5_2) 
x_ones_P5_2 = np.ones((600,1))
x_p5_2 = np.hstack((x_ones_P5_2,x_p5_2))


def classifier_P5(act,x,y,init_t,alpha,df,grad_descent,x_set):
    '''what we want to do is compute the value of the percentage between 
       how well the theta was tuned in order for us to obtain the proper actor. 
       if the hyptothesis gives a value greater than 0.5, it will be equal to an 
       output of one. Otherwise, it will be zero.'''
       
    a = 0
    b = 0 
    theta = grad_descent(df, x, y, init_t, alpha)
    for i in range(0,len(x_set)):
        if i < (len(x_set)/2) and float(dot(theta,x_set[i].T)) > 0.5:
            a += 1 
        else:
            if i > (len(x_set)/2) and float(dot(theta,x_set[i].T)) < 0.5:
                b += 1
    print("female", float(a)/(len(x_set)/2))
    print("male", float(b)/(len(x_set)/2))
    return

###
# PART 6: ON HOT ENCODING
# First we initialize for our output matrix. This will be of identity. 
# We use six actors for this part
###

y_p6 = np.array((0,0,0,0,0,0))

for i in range(len(x_p5)): #<----- making the y for part 6
    if i < 2:
        y_p6 = np.vstack((y_p6,[1,0,0,0,0,0]))
        
    elif 2 < i < 4:
        y_p6 = np.vstack((y_p6,[0,1,0,0,0,0]))
        
    elif 4 < i < 6:
        y_p6 = np.vstack((y_p6,[0,0,1,0,0,0]))
        
    elif 6 < i < 8:
        y_p6 = np.vstack((y_p6,[0,0,0,1,0,0]))
        
    elif 8 < i < 10:
        y_p6 = np.vstack((y_p6,[0,0,0,0,1,0]))
        
    else: 
        y_p6 = np.vstack((y_p6,[0,0,0,0,0,1]))

y_p6 = y_p6[1:,:]

theta_p6 = np.zeros((1025,6))
h_p6 = 1e-6

#h_p6 = 1e-6, eps = 1e-8 for 0.1315
#h_p6 = 1e-6, eps = 1e-15 for 0.1315
#h_p6 = 1e-7, eps = 1e-8 for 0.23371
#h_p6 = 1e-10, eps = 1e-15 for 0.4159
#h_p6 = 1e-8, eps = 1e-15 for 0.32719

def cost_matrix(x,y,theta):
    return float(sum(sum( (y.T - dot(theta.T,x.T)) ** 2),0))/(2*len(x))

def cost_matrix_2(x,y,theta):
    return float(sum(sum( (y.T - dot(theta.T,x.T)) ** 2),0))
    
def df_matrix(x,y,theta):
    return 2* dot(x.T,(dot(theta.T,x.T) - y.T).T)
    
def finite_difference(df_matrix,cost_matrix2,x,y,theta,h):
    #To evaluate the cost function, we use the fact of computing 
    #finite differences: (f(x+h)-f(x))/h, where "h" is an arbitrary 
    #small constant
    Fdiff = [ ] 
    DFreal = (df_matrix(x,y,theta)).T
    for i in range(0,1025):
        Fdiff.append([]) 
        for j in range(0,6):
            H = np.zeros((1025,6))
            H[i][j] = h
            Fdiff[i] = Fdiff[i] + [(cost_matrix2(x,y,theta + H) - cost_matrix2(x,y,theta)) / (h)]

    
    Fdiffnew = np.asarray(Fdiff)
    Fdiffnew = Fdiffnew.T
    Delta = norm((DFreal - Fdiffnew))
    return Delta
    

def grad_descent_matrix(df_matrix, x, y, init_t, alpha):
    #Evaulating Gradient Descent 
    EPS = 1e-15 #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 100000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df_matrix(x, y, t)
        #print "Iter", iter
        #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
        #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

#EPS = 1e-8, h = 1e-8 -> 0.62925
#EPS = 1e-8, h = 1e-6 -> 0.2176

#Part 7
t_matrix = grad_descent_matrix(df_matrix, x_p5, y_p6, theta_p6, h_p6)
t_matrix = t_matrix.T

for i in range(0,len(t_matrix)):
    t = np.delete(t_matrix[i],0) 
    t = np.reshape(t,(32,32)) 
    imsave("part4/"+str(i)+filename,t)


#Part 8 
def classifier_P8(act,x,y,init_t,alpha,df_matrix,grad_descent_matrix,x_set):
    #what we want to do is compute the value of the percentage between 
    #how well the theta was tuned in order for us to obtain the proper actor. 
    #if the hyptothesis gives a value greater than 0.5, it will be equal to an 
    #output of one. Otherwise, it will be zero. 
    a = 0
    b = 0
    c = 0 
    d = 0 
    e = 0 
    f = 0  
    theta = grad_descent_matrix(df_matrix, x, y, init_t, alpha)
    
    for i in range(0,len(x_set)):
        THETA_X = (dot(theta.T,x_set[i].T)) 
        if (0 < i < 10) and np.amax(THETA_X) == THETA_X[0]:
            a += 1
        elif (10 < i < 20) and np.amax(THETA_X) == THETA_X[1]:
            b += 1 
        elif (20 < i < 30) and np.amax(THETA_X) == THETA_X[2]:
            c += 1
        elif (30 < i < 40) and np.amax(THETA_X) == THETA_X[3]:
            d += 1
        elif (40 < i < 50) and np.amax(THETA_X) == THETA_X[4]:
            e += 1
        elif (50 < i < 60) and np.amax(THETA_X) == THETA_X[5]:
            f += 1
        else:
            pass
        
    print(actP5[0], float(a)/(len(x_set)/len(actP5)))
    print(actP5[1], float(b)/(len(x_set)/len(actP5)))
    print(actP5[2], float(c)/(len(x_set)/len(actP5)))
    print(actP5[3], float(d)/(len(x_set)/len(actP5)))
    print(actP5[4], float(e)/(len(x_set)/len(actP5)))
    print(actP5[5], float(f)/(len(x_set)/len(actP5)))
    
    return

### RESULTS

# cost of 0.072330384
# On Validation 
# ('Fran Drescher', 0.9)
# ('America Ferrera', 0.8)
# ('Kristin Chenoweth', 0.8)
# ('Alec Baldwin', 0.8)
# ('Bill Hader', 0.9)
# ('Steve Carell', 0.7)

# On Test Set
# ('Fran Drescher', 0.7)
# ('America Ferrera', 0.5)
# ('Kristin Chenoweth', 0.8)
# ('Alec Baldwin', 0.7)
# ('Bill Hader', 0.6)
# ('Steve Carell', 0.7)

###






