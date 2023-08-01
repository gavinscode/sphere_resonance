# Written by Gavin Taylor, 2023
# MIT License

### user input
velLong = 1920; #longitudional sound velocity in m/s (Measured for STMV)

velTrans = 960; #transverse sound velocity in m/s (Poisson's ratio of 1/3)

radius = 50; # radius in nm

mode = 1; # angular momentum index l: 0 to 2 - 0 breathing, 1 dipolar, 2 quadrupolar

harmonic = 1; # harmonic index n: > 1

# note: solutions are only implemented for spherical modes, not torsional



### import packages

import math
from scipy import optimize
import numpy



### local functions

def besselFn(n, x):
    # from wikipedia page on Bessel functions (spherical  section)

    # spherical Bessel function of the first kind, j_n
    match n:
        case 0:
            result = math.sin(x)/x
            
        case 1:
            result = math.sin(x)/math.pow(x,2) - math.cos(x)/x
            
        case 2:
            result = (3/math.pow(x,2)-1)*math.sin(x)/x - 3*math.cos(x)/math.pow(x,2)
            
        case 3:
            result = (15/math.pow(x,3)-6/x)*math.sin(x)/x - (15/math.pow(x,2)-1)*math.cos(x)/x

    return result

def lambEqn(omega, radius, velLong, velTrans, mode):
    # omega is frequency in radians
    
    # can be passed as an array from minimiser
    if hasattr(omega, "__len__"):
        omega = omega[0]    

    radius /= math.pow(10,9) # convert to m

    # convert to eigenvalues
    xi = omega*radius/velLong
    
    eta = omega*radius/velTrans

    # calculate Bessel functions
    j_xi_l = besselFn(mode, xi)
    j_xi_lp1 = besselFn(mode+1, xi)

    j_eta_l = besselFn(mode, eta)
    j_eta_lp1 = besselFn(mode+1, eta)

    # implements equations in Yang, SC. et al. Sci Rep 5, 18030 (2016). https://doi.org/10.1038/srep18030

    match mode:
        case 0:
            # breathing - Eqn 2 supplemental
            result = 4*math.pow(velTrans,2)*j_xi_lp1/(math.pow(velLong,2)*xi)-j_xi_l
            
        case 1:
            # dipolar - Eqn 1 main text
            result = 4*j_xi_lp1/j_xi_l*xi - math.pow(eta,2) + 2*j_eta_lp1/j_eta_l*eta
            
        case 2:
            # quadrupolar - Eqn 1 supplemental with l = 2

            # test to avoid divide by zero error   
            if j_xi_l != 0.0 and j_eta_l != 0.0:
                #simplified
                result = 10*math.pow(eta,2) - math.pow(eta,4) + (eta*j_eta_lp1*(2*math.pow(eta,2) - 32))/j_eta_l + (j_xi_lp1*xi*(4*math.pow(eta,2) \
                    + (16*j_eta_lp1*eta)/j_eta_l - 48))/j_xi_l

            else:
                result = numpy.NAN

        case _:
            raise Exception("Solutions only implemented for modes <= 2")
            
    return result

def absLambEqn(omega, radius, velLong, velTrans, mode):

    # returns absolute value of Lambs eqn for minimiser to find roots

    return abs(lambEqn(omega, radius, velLong, velTrans, mode))



### solver

rootsFound = 0

# 1 MHz step - should not be more than one root within a step
stepSize = math.pow(10,5)*2*math.pi # this should work OK for nm size spheres with GHz roots 

# start in 0.1Ghz to avoid frequncy solutions
lastOmega = 0.1*math.pow(10,9)*2*math.pi # again, should work OK for nm size spheres with GHz roots

# Note: would be good to set step size and start dynamically in case spheres with lower resonances are used...

lastValue = lambEqn(lastOmega, radius, velLong, velTrans, mode)

# loop until required number of roots have been found
while rootsFound < harmonic:

    # increase x to see if function has cross zero
    newOmega = lastOmega+stepSize

    newValue = lambEqn(newOmega, radius, velLong, velTrans, mode)

    # test if values cross zero 
    if ((lastValue <= 0 and newValue >= 0) or (lastValue >= 0 and newValue <= 0)):
        
        olderValue = lambEqn(lastOmega-stepSize, radius, velLong, velTrans, mode)

        # test if function is moving towards zero, if not the zero crossing is a discontinuity
        if abs(lastValue) < abs(olderValue):
            
            # use optimzer to find root from absolute value of function
            newBounds = optimize.Bounds(lastOmega-1, newOmega+1)
            
            optResult = optimize.minimize(absLambEqn, (lastOmega+newOmega)/2, args=(radius, velLong, velTrans, mode), method='Nelder-Mead', bounds=newBounds)

            lastRoot = optResult.x

            rootsFound += 1

    lastOmega = newOmega

    lastValue = newValue

# solved!

lastRoot /= 2*math.pi*math.pow(10,9) # convert to GHz

output = 'Resonance of (SPH, l=' + str(mode) + ', n=' + str(harmonic) + ') at ' + \
         str(lastRoot) + ' GHz'

print(output) 
