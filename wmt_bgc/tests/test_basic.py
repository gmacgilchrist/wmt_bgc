import pytest
import numpy as np
from wmt_bgc.basic import calc_densityflux

def test_calc_densityflux():
    Qheat=100
    Qfw=1E-4
    
    densityflux=calc_densityflux(Qheat,Qfw)
    
    calculated = np.array([densityflux['densityflux'],
                           densityflux['densityflux_Qheat'],
                           densityflux['densityflux_Qfw'])
    expected = np.array([-9.9498998E-7, 2.50501002E-6, -3.5E-6])
                           
    np.testing.assert_allclose(calculated,expected)
    