import pickle
import sys

import paynt.parser.sketch
import paynt.synthesizer.synthesizer_pomdp



sketch_path = "/opt/learning/rl_src/models/evade-n6-r2/sketch.templ"
properties_path = "/opt/learning/rl_src/models/evade-n6-r2/sketch.props"

quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
pomdp = quotient.pomdp
for m in dir(pomdp):
    print(m)

# states = list(pomdp.states)
# print(states[:10])
# print(pomdp.transition_matrix.nr_entries/6)
print(len(pomdp.states))
print(len(pomdp.observations))
print(pomdp.observation_valuations.get_json(0))
print(dir(pomdp.observation_valuations))
# print(pomdp.labeling)
# print(pomdp.labeling.get_states('goal'))




# k = 3  # May be unknown?
# quotient.set_imperfect_memory_size(k)


# synthesizer = paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(quotient, method="ar", storm_control=None)


# paynt.utils.profiler.GlobalTimeoutTimer.start(15)
# assignment = synthesizer.synthesize()
# # before the quotient is modified we can use this assignment to compute Q-values
# assert assignment is not None
# qvalues = quotient.compute_qvalues(assignment)
# # note Q(s,n) may be None if (s,n) exists in the unfolded POMDP but is not reachable in the induced DTMC
# memory_size = len(qvalues[0])
# print(memory_size)

# # # qvalues = PAYNT_Playground.fill_nones_in_qvalues(qvalues)
# # # return qvalues
