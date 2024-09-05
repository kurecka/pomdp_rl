import pickle
import sys
sys.path.append("/opt/paynt")
sys.path.append("/opt/paynt/rl_src")

import paynt.parser.sketch
import paynt.synthesizer.synthesizer_pomdp

from environment.environment_wrapper import *
from environment.pomdp_builder import *



sketch_path = "/opt/paynt/rl_src/models/refuel-10/sketch.templ"
properties_path = "/opt/paynt/rl_src/models/refuel-10/sketch.props"

quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
k = 3  # May be unknown?
quotient.set_imperfect_memory_size(k)
synthesizer = paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(quotient, method="ar", storm_control=None)


assignment = synthesizer.synthesize()
# before the quotient is modified we can use this assignment to compute Q-values
assert assignment is not None
qvalues = quotient.compute_qvalues(assignment)
# note Q(s,n) may be None if (s,n) exists in the unfolded POMDP but is not reachable in the induced DTMC
memory_size = len(qvalues[0])
assert k == memory_size

print(qvalues)
# qvalues = PAYNT_Playground.fill_nones_in_qvalues(qvalues)
# return qvalues
