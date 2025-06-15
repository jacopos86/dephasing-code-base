from pydephasing.global_params import GPU_ACTIVE

#
#   This module implements the
#   generalized Fermi golden rule
#   calculation
#

if GPU_ACTIVE:
    from pydephasing.global_params import gpu
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda

#
#   Abstract class : instantiate CPU / GPU version
#

class GeneralizedFermiGoldenRule(object):
    def __init__(self):
        pass
    #
    #   instance CPU / GPU
    #
    def generate_instance(self, REAL_TIME, FREQ_DOMAIN):
        if GPU_ACTIVE:
            return GeneralizedFermiGoldenRuleGPU (REAL_TIME, FREQ_DOMAIN)
        else:
            return GeneralizedFermiGoldenRuleCPU (REAL_TIME, FREQ_DOMAIN)
        
# --------------------------------------------------------
#
#    CPU Fermi Golden Rule implementation
#
# --------------------------------------------------------

class GeneralizedFermiGoldenRuleCPU:
    def __init__(self, REAL_TIME, FREQ_DOMAIN):
        self.REAL_TIME = REAL_TIME
        self.FREQ_DOMAIN = FREQ_DOMAIN






















# --------------------------------------------------------
#
#    CPU Fermi Golden Rule implementation
#
# --------------------------------------------------------

class GeneralizedFermiGoldenRuleGPU:
    def __init__(self, REAL_TIME, FREQ_DOMAIN):
        self.REAL_TIME = REAL_TIME
        self.FREQ_DOMAIN = FREQ_DOMAIN