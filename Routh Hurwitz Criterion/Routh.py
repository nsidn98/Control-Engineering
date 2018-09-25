import numpy as np
import sympy as sp
'''
%ROUTH   Routh array.
%   RA=ROUTH(R,EPSILON,symbolic) returns the symbolic/numeric
%   Routh array RA for polynomial R(s). The following special
%    cases are considered:
%   1) zero first elements and 2) rows of zeros. All zero first
%   elements are replaced with the symbolic variable EPSILON
%   which can be later substituted with positive and negative
%   small numbers using SUBS(RA,EPSILON,...). When a row of
%   zeros is found, the auxiliary polynomial is used.
%   Also a custom determinant function has been used
%   because np.linalg.det has some issues with precision
%
%	Examples:
%
%	1) Routh array for s^3+2*s^2+3*s+1
%
%
%		ra=routh([1,2,3,1],0.01,False)
%		ra =
%
% 		   1.0000    3.0000
% 		   2.0000    1.0000
% 		   2.5000         0
% 		   1.0000         0
%
%	2) Routh array for s^3+a*s^2+b*s+c
%
%		a,b,c=sp.symbols('a b c') # create symbols for the variables
%       # set the value of 'symbolic'=True in the function argument
%		ra=routh([1,a,b,c],0.01,True);
%		ra =
%
%		[          1,          b]
%		[          a,          c]
%		[ (-c+b*a)/a,          0]
%		[          c,          0]
%
%
%   Author:Siddharth Nayak
%   E-mail:siddharthnayak98@gmail.com
%
'''


def routh(polynomial,epsilon,symbolic):
    dim=len(polynomial)
    coeff=dim
    columns=int(np.ceil(coeff/2))
    RA=sp.zeros(coeff,columns) # create the Routh table
    s=sp.Symbol('S')

    #assemble the first and second rows
    top=polynomial[0::2]
    down=polynomial[1::2]
    if len(top)!=len(down):
        down.append(0)

    for i in range(len(top)):
        RA[0,i]+=top[i]
        RA[1,i]+=down[i]

    rows=coeff-2 #number of rows that need determinants
    index=np.zeros(rows)

    for i in range(rows):
        index[rows-i-1]=np.ceil((i+1)/2)
        
    for i in range(2,coeff):  #go from the 3rd row to the last
        if np.sum(np.abs(RA[i-1,:]))==0: # while row is zero
            print('\n Row of zeros detected in row %d. \n Finding Auxillary Polynomial by differentiating row %d'%(i,i-1))
            order=coeff-i+1
            order_arr=np.arange(order,-1,-2)
            poly=0
            #get the polynomial for differentiation
            for k in range(len(order_arr)):
                poly+=(s**(order_arr[k]))*RA[i-2,k]
            diff=sp.diff(poly,s)
            a=sp.Poly(diff,s)
            c=a.coeffs()
            for l in range(columns-len(c)):
                c.append(0)
            for l in range(columns):
                RA[i-1,l]=c[l]
        
        elif RA[i-1:0]==0: # first element is zero
            print('\n First element is zero. Replacing with epsilon')
            RA[i-1:0]=epsilon
            
        for j in range(int(index[i-2])):
            if symbolic:
                RA[i,j]=-sp.det(sp.Matrix([[RA[i-2,0],RA[i-2,j+1]],[RA[i-1,0],RA[i-1,j+1]]]))/RA[i-1,0]
            else:
                RA[i,j]=-my_det(np.array([[RA[i-2,0],RA[i-2,j+1]],[RA[i-1,0],RA[i-1,j+1]]]).astype(float))/RA[i-1,0]
            #RA[i,j]=-sp.det([[RA[i-2,0],RA[i-2,j+1]],[RA[i-1,0],RA[i-1,j+1]]])/RA[i-1,0]
            #RA[i,j]=-(np.linalg.det(np.array([[RA[i-2,0],RA[i-2,j+1]],[RA[i-1,0],RA[i-1,j+1]]]).astype(float))/RA[i-1,0])
            
            
    return RA

def my_det(X):
    '''
    Because numpy.linalg.det sucks in precision of numbers :P
    '''
    X = np.array(X, dtype='float64', copy=True)
    n = len(X)
    s = 0
    if n != len(X[0]):
      return ValueError
    for i in range(0, n):
      maxElement = abs(X[i, i])
      maxRow = i
      for k in range(i + 1, n):
          if abs(X[k, i]) > maxElement:
              maxElement = abs(X[k, i])
              maxRow = k
      if maxRow != i:
          s += 1
      for k in range(i, n):
          X[i, k], X[maxRow, k] = X[maxRow, k], X[i, k]
      for k in range(i + 1, n):
          c = -X[k, i] / X[i, i]
          for j in range(i, n):
              if i == j:
                  X[k, j] = 0
              else:
                  X[k, j] += c * X[i, j]
    det = (-1)**s
    for i in range(n):
      det *= X[i, i]
    return det