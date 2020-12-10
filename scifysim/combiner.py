import sympy as sp
import numpy as np
import scifysim as sf
import kernuller
import logging

logit = logging.getLogger(__name__)

class combiner(object):
    def __init__(self,expr, thesubs):
        
        self.Na = expr.shape[1]
        self.M = expr
        
        #X = sp.MatrixSymbol('X', 2, 4)
        X = sp.Matrix(sp.symbols('X:{}'.format(self.Na), real=True))
        Y = sp.Matrix(sp.symbols('Y:{}'.format(self.Na), real=True))
        self.Xm = sp.Matrix([[X,Y]])

        lamb = sp.symbols("lambda", real=True)
        self.k = sp.symbols("k", real=True)
        self.alpha = sp.symbols("alpha", real=True)
        self.beta = sp.symbols("beta", real=True)
        self.A = sp.Matrix([[self.alpha],
                       [self.beta]])
        # The error phasor:
        # Prone to change: vectorization of lambda would require e(lambda) in sympy!
        # Will need to build that instead of the scipy.interp1D
        self.e = sp.Matrix(sp.symbols('e:{}'.format(self.Na), real=False))
        self.E = sf.utilities.vec2diag(self.e)

        # The source amplitude
        s = sp.symbols("s", real=True)
        # Source amplitude is not managed here
        thesubs.append((s, 1))



        geomphaseterm = sp.I*self.k*self.Xm@self.A
        # The pointing matrix:
        self.P = sp.Matrix([sp.exp(term) for term in geomphaseterm])
        self.T_clean = self.M@self.E@(s*self.P)
        self.T_subs = self.T_clean.subs(thesubs)
        
        self.encaps = sf.utilities.ee(self.T_subs)
        # Here, lambdifying for the parameters
        self.encaps.lambdify((self.alpha, self.beta,self.Xm ,self.k, self.e), modules="numexpr")
        pass
    @classmethod
    def angel_woolf(cls, file, ph_shifters=(0,sp.pi/2)):
        
        M = sf.combiners.angel_woolf_ph(ph_shifters=ph_shifters)
        
        #Photometric tap
        logit.warning("Here, forced to assume I have only one symbol: sigma")
        for symbol in M.free_symbols:
            sigma = symbol
        
        thesigma = file.getfloat("configuration", "photometric_tap")
        thesubs = [(sigma, thesigma)]
        obj = cls(M, thesubs)
        return obj
def test_combiner(combiner, nwl=10):
    hr = kernuller.mas2rad(16)
    xx, yy = np.meshgrid(np.linspace(-hr, hr, 1024),
                     np.linspace(-hr, hr, 1024))
    if combiner.Na == 4:
        array = kernuller.VLTI
    elif combiner.Na == 6:
        array = kernuller.CHARA
        
    #Prone to change: wavelength vecorization incomplete
    es = np.ones(combiner.Na)
    # This one is for vectorized wavelength (only yet assumes no injection-wl dependency)
    #amap = combiner.encaps(xx[:,:,None], yy[:,:,None], array.flatten(), 2*np.pi/3.5e-6*np.ones(10)[None,None,:],
    #                      np.ones(combiner.Na))
    lambda_range = np.linspace(3.0e-6, 4.2e-6, 10)
    amap = np.array([combiner.encaps(xx[:,:], yy[:,:], array.flatten(), np.array([2*np.pi/thelambda])[None,None,:],
                          np.ones(combiner.Na)) for thelambda in lambda_range])
    return amap
def test_angel_woolf():
    fpath = "/home/rlaugier/Documents/hi5/SCIFYsim/scifysim/config/default_new_4T.ini"
    logit.warning("Hard path here!")
    theconfig = sf.parsefile.parse_file(fpath)
    acombiner = combiner.angel_woolf(theconfig)
    print("free symbols initial",acombiner.T_clean.free_symbols)
    print("free symbols",acombiner.T_subs.free_symbols)
    amap = test_combiner(acombiner)
    print(amap.shape)
    return amap