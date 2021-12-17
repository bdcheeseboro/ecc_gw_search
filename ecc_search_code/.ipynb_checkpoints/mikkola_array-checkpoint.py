"""
This a a program to solve Keplers's equation l = u - e sin(u) using
modified Mikkola's method.

Arguments:  array of l
            array of e
Returns:    array of u

"""

import numpy as np

def get_u(l,e):
    Pi = np.pi
    TwoPi = 2*Pi
    
    #if e<0 or e>=1:
    #   print("ERROR: The eccentricity of an ellipse must lie within [0,1).")
    #   return np.nan
    
    #if e==0 or l==0:
    #   return l
    
    l_single = False
    if type(l) is int:
        l_single = True
        l = np.array([l])
    elif type(l) is float:
        l_single = True
        l = np.array([l])
    
    e_single = False
    if type(e) is int:
        e_single = True
    elif type(e) is float:
        e_single = True
    elif len(l) != len(e):
        return np.nan
    sgn = np.sign(l)
    l = sgn*l                       ## l>0
    ncycles = (l//TwoPi)
    l = l - (ncycles * TwoPi)       ## 0<=l<2*pi
    flag = l>Pi
    l_flag = l[flag]
    l_flag = TwoPi - l_flag
    l[flag] = l_flag
    #if flag:
    #   l = TwoPi - l               ## 0<=l<=pi
    
    alpha  = (1-e)/(4*e + 0.5)
    alpha3 = alpha*alpha*alpha
    beta   = (l/2.0)/(4*e + 0.5)
    beta2  = beta*beta
    z = np.zeros_like(l)
    
    z_flag_true = beta>0
    if e_single:
        z[z_flag_true] = np.cbrt(beta[z_flag_true] + np.sqrt(alpha3 + beta2[z_flag_true]))
    else:
        z[z_flag_true] = np.cbrt(beta[z_flag_true] + np.sqrt(alpha3[z_flag_true] + beta2[z_flag_true]))
    
    z_flag_false = beta<=0
    if e_single:
        z[z_flag_false] = np.cbrt(beta[z_flag_false] - np.sqrt(alpha3 + beta2[z_flag_false]))
    else:
        z[z_flag_false] = np.cbrt(beta[z_flag_false] - np.sqrt(alpha3[z_flag_false] + beta2[z_flag_false]))
    
    #if beta>0:
    #   z = np.cbrt(beta + np.sqrt(alpha3 + beta2))
    #else:
    #   z = np.cbrt(beta - np.sqrt(alpha3 + beta2))
    
    s = (z - alpha/z)
    s5 = s*s*s*s*s
    w = (s - (0.078*s5)/(1 + e))
    w3= w*w*w
    E0 = (l + e*(3*w - 4*w3))
    u = E0

    esu  = e*np.sin(u)
    ecu  = e*np.cos(u)

    fu  = (u - esu - l)
    f1u = (1 - ecu)
    f2u = (esu)
    f3u = (ecu)
    f4u =-(esu)
    
    u1 = -fu/ f1u
    u2 = -fu/(f1u + f2u*u1/2.0)
    u3 = -fu/(f1u + f2u*u2/2.0 + f3u*(u2*u2)/6.0)
    u4 = -fu/(f1u + f2u*u3/2 + f3u*(u3*u3)/6.0 + f4u*(u3*u3*u3)/24.0)
    xi = (E0 + u4)

    sol = xi
    sol[flag] = TwoPi - xi[flag]

    u = sgn*(sol + ncycles*TwoPi)

    if l_single:
        u = u[0]

    return(u)
    
    
