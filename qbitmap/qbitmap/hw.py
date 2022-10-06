"""hw.py - hardware constants for experimental access."""
import cirq_google as cg

PROJECT_ID = None
PROCESSOR_ID = None
GATESET = None

# Dump info on library import
print("PROJECT ID: {}".format(PROJECT_ID))
print("PROCESSOR:  >>> {} <<<".format(PROCESSOR_ID))
print("GATESET:    {}".format(GATESET))

print("WARNING: QUANTUM ENGINE ACCESS DISABLED")
ENGINE = None
PROESSOR = None
DEVICE = None