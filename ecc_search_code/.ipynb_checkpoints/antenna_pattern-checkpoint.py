import numpy as np
from numpy import pi, sin, cos, sqrt, arctan2


def antenna_pattern(alpha_gw, delta_gw, alpha_p, delta_p):

    """
    Arguments:
        alpha_gw    is RA  of GW source
        delta_gw    is DEC of GW source
        alpha_p     is RA  of pulsar
        delta_p     is DEC of pulsar

    Returns:
        coseta      Dot product of direction to GW source and direction to pulsar
        Fp          Antenna pattern F+
        Fx          Antenna pattern Fx 
    """
    
    n1 = cos(alpha_p)*cos(delta_p)
    n2 = sin(alpha_p)*cos(delta_p)
    n3 = sin(delta_p)
    
    cos_theta = cos(delta_gw)*cos(delta_p)*cos(alpha_gw-alpha_p) + sin(delta_gw)*sin(delta_p)
    
    e11p = (sin(alpha_gw))**2 - (cos(alpha_gw))**2 * (sin(delta_gw))**2
    e12p = -sin(alpha_gw)*cos(alpha_gw) * ((sin(delta_gw))**2 + 1)
    e13p = cos(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e21p = -sin(alpha_gw)*cos(alpha_gw) * ((sin(delta_gw))**2 + 1)
    e22p = (cos(alpha_gw))**2 - (sin(alpha_gw))**2 * (sin(delta_gw))**2
    e23p = sin(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e31p = cos(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e32p = sin(alpha_gw)*sin(delta_gw)*cos(delta_gw)
    e33p = -(cos(delta_gw))**2
    
    Fp = (n1*(n1*e11p+n2*e12p+n3*e13p)+
          n2*(n1*e21p+n2*e22p+n3*e23p)+
          n3*(n1*e31p+n2*e32p+n3*e33p))
    Fp = 0.5 * Fp/ (1-cos_theta)
    
    e11c = sin(2*alpha_gw) * sin(delta_gw)
    e12c = -cos(2*alpha_gw) * sin(delta_gw)
    e13c = -sin(alpha_gw) * cos(delta_gw)
    e21c = -cos(2*alpha_gw) * sin(delta_gw)
    e22c = -sin(2*alpha_gw) * sin(delta_gw)
    e23c = cos(alpha_gw) * cos(delta_gw)
    e31c = -sin(alpha_gw) * cos(delta_gw)
    e32c = cos(alpha_gw) * cos(delta_gw)
    e33c  = 0
    
    Fx = (n1*(n1*e11c+n2*e12c+n3*e13c)+
          n2*(n1*e21c+n2*e22c+n3*e23c)+
          n3*(n1*e31c+n2*e32c+n3*e33c))
    Fx = 0.5 * Fx/ (1-cos_theta)
    
    if (cos_theta ==1):
        print("Warning: The pulsar and GW source are collinear. Antenna patterns are undefined.")
        Fp=Fx=np.nan
    
    return cos_theta, Fp, Fx
