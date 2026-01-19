import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define aliases you use often
THz = ureg.terahertz
ps  = ureg.picosecond
ns  = ureg.nanosecond
fs  = ureg.femtosecond
us  = ureg.microsecond
eV  = ureg.electron_volt
T   = ureg.tesla
K   = ureg.kelvin