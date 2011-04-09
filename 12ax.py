#!/usr/bin/python

class SeqGenerator:
    def __init__(self):
        self.lastNum = self.lastLetter = ""

    def peek(self, nextInput):
        if nextInput in ["1","2"]:
            return "L"
        elif nextInput in ["A","B"]:
            return "L"
        elif nextInput in ["X","Y"]:
            seq = self.lastNum + self.lastLetter + nextInput
            if seq in ["1AX","2BY"]: return "R"
            return "L"
        return None

    def next(self, nextInput):
        out = self.peek(nextInput)
        if nextInput in ["1","2"]:
            self.lastNum = nextInput
            self.lastLetter = ""
        elif nextInput in ["A","B","X","Y"]:
            self.lastLetter = nextInput
        return out

    def nextSeq(self, nextInputs):
        return [ self.next(c) for c in nextInputs ]

    def nextStr(self, nextInputs):
        return "".join([ self.next(c) or "" for c in nextInputs ])

def seqStr(s): return SeqGenerator().nextStr(s)

import pybrain
import pybrain.tools.shortcuts as bs
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer, SoftmaxLayer
import pybrain.structure.networks as bn
import pybrain.structure.connections as bc
import pybrain.rl.learners.valuebased as bl
import pybrain.supervised as bt
import pybrain.datasets.sequential as bd

#nn = bs.buildNetwork(9, 9, 2, hiddenclass=LSTMLayer)

print "preparing network ...",
nn = bn.RecurrentNetwork()
nn.addInputModule(LinearLayer(9, name="in"))
nn.addModule(LSTMLayer(9, name="hidden"))
nn.addOutputModule(LinearLayer(2, name="out"))
nn.addConnection(bc.FullConnection(nn["in"], nn["hidden"], name="c1"))
nn.addConnection(bc.FullConnection(nn["hidden"], nn["out"], name="c2"))
nn.addRecurrentConnection(bc.FullConnection(nn["hidden"], nn["hidden"], name="c3"))
nn.sortModules()
print "done"

import random

def getRandomSeq(seqlen):
    s = ""
    count = 0
    gen = SeqGenerator()
    for i in xrange(seqlen):
        if(float(count) / (i+1) < random.uniform(0.0,0.2)):
            if gen.lastNum + gen.lastLetter == "1A": c = "X"
            elif gen.lastNum + gen.lastLetter == "2B": c = "Y"
            elif gen.lastNum == "1": c = "A"
            elif gen.lastNum == "2": c = "B"
            else: c = random.choice("12")
        else:
            c = random.choice("123ABCXYZ")
        s += c
        if gen.next(c) == "R": count += 1
    return s

import pybrain.utilities
def inputAsVec(c): return pybrain.utilities.one_to_n("123ABCXYZ".index(c), 9)
def outputAsVec(c):
    if c is None: return (0.0,0.0)
    else: return pybrain.utilities.one_to_n("LR".index(c), 2)

def addSequence(dataset, seqlen):
    dataset.newSequence()
    s = getRandomSeq(seqlen)
    for i,o in zip(s, SeqGenerator().nextSeq(s)):
        dataset.addSample(inputAsVec(i), outputAsVec(o))

def generateData(seqlen = 100, nseq = 20):
    dataset = bd.SequentialDataSet(9, 2)
    for i in xrange(nseq): addSequence(dataset, seqlen)
    return dataset

#l = bl.QLambda()
trainer = bt.RPropMinusTrainer(module=nn)
#trainer = bt.BackpropTrainer( nn, momentum=0.9, learningrate=0.00001 )

from pybrain.tools.validation import ModuleValidator

import thread
def userthread():
    from IPython.Shell import IPShellEmbed
    ipshell = IPShellEmbed()
    ipshell()
#thread.start_new_thread(userthread, ())

def getSeqOutputFromNN(module, seq):
    outputs = ""
    module.reset()
    for i in xrange(len(seq)):
        output = module.activate(seq[i][0])
        l,r = output
        l,r = l > 0.5, r > 0.5
        if l and not r: c = "L"
        elif not l and r: c = "R"
        elif not l and not r: c = ""
        else: c = "?"
        outputs += c
    return outputs


# carry out the training
for i in xrange(100):
    trndata = generateData()
    tstdata = generateData()
    trainer.setData(trndata)
    trainer.train()
    trnresult = 100. * (ModuleValidator.MSE(nn, trndata))
    tstresult = 100. * (ModuleValidator.MSE(nn, tstdata))
    print "train error: %5.2f%%" % trnresult, ",  test error: %5.2f%%" % tstresult

    s = getRandomSeq(100)
    print " real:", seqStr(s)
    print "   nn:", getSeqOutputFromNN(nn, s)
    