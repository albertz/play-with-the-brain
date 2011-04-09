#!/usr/bin/python

lastNum = ""
lastLetter = ""
def nextWantedOutput(nextInput):
        global lastNum, lastLetter
        if nextInput in ["1","2"]:
                lastNum = nextInput
                lastLetter = ""
                return "L"
        elif nextInput in ["A","B"]:
                lastLetter = nextInput
                return "L"
        elif nextInput in ["X","Y"]:
                seq = lastNum + lastLetter + nextInput
                lastLetter = nextInput
                if seq in ["1AX","2BY"]: return "R"
                return "L"
        return None

def nextWantedOutputs(nextInputs):
        return [ nextWantedOutput(c) for c in nextInputs ]


import pybrain
import pybrain.tools.shortcuts as pys
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer

nn = pys.buildNetwork(9, 9, 2, hiddenclass=LSTMLayer)





