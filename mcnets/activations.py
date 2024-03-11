import numpy as np

## Activation Functions ##
# Identity
def lin(calcVec): 
    """Identity activation function"""
    return calcVec

# RELU-Like
def relu(calcVec): 
    """ReLU activation function"""
    return np.maximum(calcVec, 0)

def lrelu(vec):
    """Leaky ReLU (`alpha = -0.3`)"""
    return np.maximum(vec, -0.3)

def silu(calcVec):
    """SiLU / Swish activation function
    
    Alternative to the ReLU function"""
    return calcVec / (1 + np.exp(-np.clip(calcVec, -7e2, None)))

# SIG-Like
e = 2.7182818284590452353602874713527
def sig(cv):
    """Sigmoid activation function"""
    cv = np.maximum(cv, -7e2)
    # cv[cv <= -7e2] = 7e2
    return 1 / (1 + e**(-cv))

def dsilu(calcVec):
    """dSiLU / dSwish activation function
    
    Alternative to the sig function"""
    return sig(calcVec) * (1 + calcVec * (1 - sig(calcVec)))

def tanh(calcVec):
    """Tanh activation function"""
    return np.tanh(calcVec)

# Exponential-Like
def elu(calcVec):
    """ELU activation function"""
    return 0.4 * (np.expm1(np.clip(calcVec, None, 7e2)))

# Logirithmic-like
def root(calcVec):
    """Custom activation function
    
    Uses arctan as a base multiplied by the squareroot of the input"""
    return np.arctan(calcVec) * np.abs(calcVec)**0.5 / 1.5

# dRELU-Like
def sqr(calcVec):
    """Custom activation function
    
    Negative values (and 0) are set to 0, all others become 1"""
    calcVec[calcVec <= 0] = 0
    calcVec[calcVec > 0] = 1
    return calcVec

def rnd(calcVec):
    """Custom activation function
    
    Simply rounds the input vector to the nearest integer"""
    calcVec = np.round(calcVec)
    return calcVec

# Chaotic / Experimental
def resu(calcVec):
    """Custom activation function
    
    Multiplies the input vector by |sin(x)|"""
    return np.abs(np.sin(calcVec))*calcVec

def resu2(calcVec):
    """Custom activation function
    
    Sets all negative values to 0, then multiplies the input vector by |sin(x)|"""
    calcVec[calcVec < 0] = 0
    return abs(np.sin(calcVec))*calcVec

def exp(calcVec):
    """Custom activation function
    
    Returns the input vector squared"""
    return calcVec**2