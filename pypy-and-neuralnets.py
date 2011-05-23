#!/usr/bin/python
## RPython Neural Network - 0.1b
## by Brett Hartshorn 2010
## goatman.py@gmail.com
## tested with Ubuntu Lucid, copy this file to your pypy root directory and run "python rAI.py"
## you can test just pypy compilation with "python rAI.py --pypy --subprocess"
## you need to install SDL headers, apt-get install libsdl-dev

'''
PyPY Tips

math.radians	not available
random.uniform	not available

incorrect:
	string.strip()	will not work without an arg
	list.sort()		not available

correct:
	string.strip(' \n')
	list = sortd( list )
	list.pop( index )		# no list.pop() with end as default

'''

COLUMNS = 24
LAYERS = 5
STEM = 8

import os, sys, time
#from random import *		# not valid in rpython?
import math				# math.radians is missing in pypy?

degToRad = math.pi / 180.0
def radians(x): 
	"""radians(x) -> converts angle x from degrees to radians
	"""
	return x * degToRad

from pypy.rlib import streamio
from pypy.rlib import rpoll

# apt-get install libsdl-dev
from pypy.rlib.rsdl import RSDL, RSDL_helper
from pypy.rlib.rarithmetic import r_uint
from pypy.rpython.lltypesystem import lltype, rffi
from pypy.rlib.listsort import TimSort
from pypy.rlib.jit import hint, we_are_jitted, JitDriver, purefunction_promote

#random.randrange(0, up)		# not a replacement for random.uniform
#Choose a random item from range(start, stop[, step]).   
#This fixes the problem with randint() which includes the
#endpoint; in Python this is usually not what you want.


if '--pypy' in sys.argv:		# random.random works in pypy, but random.uniform is missing?
	from pypy.rlib import rrandom
	RAND = rrandom.Random()
	RAND.init_by_array([1, 2, 3, 4])
	def random(): return RAND.random()
	def uniform(start,end): return ( RAND.random() * (end-start) ) - start
else:
	from random import *

def distance( v1, v2 ):
	dx = v1[0] - v2[0] 
	dy = v1[1] - v2[1] 
	dz = v1[2] - v2[2] 
	t = dx*dx + dy*dy + dz*dz
	return math.sqrt( float(t) )


Njitdriver = JitDriver(
	greens = 'cur train times branes last_spike spikers temporal thresh triggers self abs_refactory abs_refactory_value'.split(),
	reds = 'i t a b c bias neuron'.split()
)
class RecurrentSpikingModel(object):
	'''
	Model runs in realtime and lossy - stores recent
	spikes in a list for realtime learning.
	Uses spike train, with time delay based on distance, 		Spikes will trigger absolute refactory period.
	Membrane has simple linear falloff.

		notes:
			log(1.0) = 0.0
			log(1.5) = 0.40546510810816438
			log(2.0) = 0.69314718055994529
			log(2.7) = 0.99325177301028345
			log(3.0) = 1.0986122886681098

	'''
	def iterate( self ):
		now = float( time.time() )
		self._state_dirty = True; train = self._train
		brane = self._brane; rest = self._rest
		branes = self._branes; temporal = self._temporal
		abs_refactory = self._abs_refactory
		abs_refactory_value = self._abs_refactory_value
		fps = self._fps
		elapsed = now - self._lasttime
		clip = self._clip
		cur = self._lasttime
		last_spike = self._last_spike
		spikers = self._spikers
		thresh = self._thresh
		triggers = self._triggers

		## times of incomming spikes ##
		times = train.keys()
		## To do, if seziure, lower all connection strengths
		##TimSort(times).sort()		# pypy note - list have no .sort(), and JIT dislikes builtin sorted(list)

		if not times:
			a = now - self._lasttime
			if a > 0.1: brane = rest
			else:
				b = a * 10	# 0.0-1.0
				brane += (rest - brane) * b
			branes.append( (brane,now) )

		i = 0
		t = a = b = c = bias = .0
		neuron = self
		ntrain = len(train)
		while train:		#the JIT does only properly support a while loop as the main dispatch loop
			Njitdriver.can_enter_jit( 
				cur=cur, train=train, times=times, branes=branes, last_spike=last_spike, spikers=spikers,
				temporal=temporal, thresh=thresh, triggers=triggers, self=self,
				abs_refactory=abs_refactory, abs_refactory_value=abs_refactory_value,
				i=i, t=t, a=a, b=b, c=c,
				neuron=neuron, bias=bias
			)
			Njitdriver.jit_merge_point(
				cur=cur, train=train, times=times, branes=branes, last_spike=last_spike, spikers=spikers,
				temporal=temporal, thresh=thresh, triggers=triggers, self=self,
				abs_refactory=abs_refactory, abs_refactory_value=abs_refactory_value,
				i=i, t=t, a=a, b=b, c=c,
				neuron=neuron, bias=bias
			)
			##bias, neuron = train.pop( t )	## not pypy
			t = times[i]; i += 1
			bias,neuron = train[t]
			del train[t]


			if t <= cur:
				#print 'time error'
				pass
			else:
				a = t - cur
				cur += a
				#print 'delay', a
				if cur - last_spike < abs_refactory:
					brane = abs_refactory_value
					branes.append( (brane,cur) )
				else:
					spikers.append( neuron )
					if a > 0.1:
						brane = rest + bias
						branes.append( (brane,cur) )
					else:
						b = a * 10	# 0.0-1.0
						brane += (rest - brane) * b
						c = b * temporal
						bias *= math.log( (temporal-c)+2.8 )
						brane += bias
						branes.append( (brane,cur) )

						if brane > thresh:
							triggers.append( neuron )
							self.spike( cur )
							brane = abs_refactory_value
							#fps *= 0.25		# was this to play catch up?
							break # this is efficient!

		#self._train = {}
		self._brane = brane
		#for lst in [branes, spikers, triggers]:	# not allowed in pypy
		while len(branes) > clip: branes.pop(0)
		while len(spikers) > clip: spikers.pop(0)
		while len(triggers) > clip: triggers.pop(0)

		self._lasttime = end = float(time.time())
		#print ntrain, end-now


	def detach( self ):
		self._inputs = {}		# time:neuron
		self._outputs = []	# children (neurons)
		self._train = {}
		self._branes = []
		self._spikers = []	# last spikers
		self._triggers = []	# last spikers to cause a spike

	def __init__( self, name='neuron', x=.0,y=.0,z=.0, column=0, layer=0, fps=12, thresh=100.0, dendrite_bias=35.0, dendrite_noise=1.0, temporal=1.0, distance_factor=0.1, red=.0, green=.0, blue=.0 ):
		self._name = name
		self._fps = fps		# active fps 10?
		self._thresh = thresh
		self._column = column
		self._layer = layer
		self._brane = self._rest = -65.0
		self._abs_refactory = 0.05
		self._abs_refactory_value = -200.0
		self._spike_value = 200.0
		self._temporal = temporal	# ranges 1.0 - inf
		self._distance_factor = distance_factor
		self._dendrite_bias = dendrite_bias
		self._dendrite_noise = dendrite_noise
		self._clip = 128
		self._learning = False
		self._learning_rate = 0.1

		## the spike train of incomming spikes
		self._train = {}

		## list of tuples, (brane value, abs time)
		## use this list to render wave timeline ##
		self._branes = []

		self._spikers = []	# last spikers
		self._triggers = []	# last spikers to cause a spike
		self._lasttime = time.time()
		self._last_spike = 0
		self._inputs = {}		# time:neuron
		self._outputs = []	# children (neurons)

		self._color = [ red, green, blue ]
		#x = uniform( 0.2, 0.8 )
		#y = random()	#uniform( 0.2, 0.8 )
		#z = random()	#uniform( 0.2, 0.8 )
		self._pos = [ x,y,z ]
		self._spike_callback = None
		self._state_dirty = False
		self._active = False
		self._cached_distances = {}
		self._clipped_spikes = 0	# for debugging

		self._draw_spike = False
		#self._braneRect = self._spikeRect = None

	def randomize( self ):
		for n in self._inputs:
			bias = self._dendrite_bias + uniform( -self._dendrite_noise, self._dendrite_noise )
			self._inputs[ n ] = bias


	def mutate( self, v ):
		for n in self._spikers:
			self._inputs[ n ] += uniform(-v*2,v*2)
		for n in self._triggers:
			self._inputs[ n ] += uniform(-v,v)

		if not self._spikes or not self._triggers:
			for n in self._inputs:
				self._inputs[ n ] += uniform(-v,v)


	def reward( self, v ):
		if not self._spikers: print 'no spikers to reward'
		for n in self._spikers:
			bias = self._inputs[ n ]
			if abs(bias) < 100:
				if bias > 15:
					self._inputs[ n ] += v
				else:
					self._inputs[ n ] -= v
			

	def punish( self, v ):
		for n in self._inputs:
			self._inputs[n] *= 0.9
			
		for n in self._spikers:
			self._inputs[ n ] += uniform( -v, v )



	def attach_dendrite( self, neuron ):
		bias = self._dendrite_bias + uniform( -self._dendrite_noise, self._dendrite_noise )
		# add neuron to input list
		if neuron not in self._inputs:
			self._inputs[ neuron ] = bias
			#print 'attached neuron', neuron, bias
		if self not in neuron._outputs:
			neuron._outputs.append( self )
			neuron.update_distances()
		return bias

	def update_distances( self ):
		for child in self._outputs:
			dist = distance( self._pos, child._pos )
			self._cached_distances[ child ] = dist

	def spike( self, t ):
		#print 'spike', t
		self._draw_spike = True
		self._last_spike = t
		self._branes.append( (self._spike_value,t) )
		self._brane = self._abs_refactory_value
		self._branes.append( (self._brane,t) )
		if self._learning and self._triggers:
			#print 'learning'
			n = self._triggers[-1]
			bias = self._inputs[ n ]
			if bias > 0: self._inputs[n] += self._learning_rate
			else: self._inputs[n] -= self._learning_rate

		if self._spike_callback:
			self._spike_callback( t )

		for child in self._outputs:
			#print 'spike to', child
			bias = child._inputs[ self ]
			#dist = distance( self._pos, child._pos )
			dist = self._cached_distances[ child ]
			dist *= self._distance_factor
			child._train[ t+dist ] = (bias,self)

	def stop( self ):
		self._active = False
		print 'clipped spikes', self._clipped_spikes
	def start( self ):
		self._active = True
		self.iterate()

	def setup_draw( self, format, braneRect, groupRect, spikeRect, colors ):
		self._braneRect = braneRect
		self._groupRect = groupRect
		self._spikeRect = spikeRect
		self._sdl_colors = colors
		r,g,b = self._color
		self._group_color = RSDL.MapRGB(format, int(r*255), int(g*255), int(b*255))

	def draw( self, surf ):
		#if self._braneRect:
		fmt = surf.c_format
		b = int(self._brane * 6)
		if b > 255: b = 255
		elif b < 0: b = 0
		color = RSDL.MapRGB(fmt, 0, 0, b)
		RSDL.FillRect(surf, self._braneRect, color)
		RSDL.FillRect(surf, self._groupRect, self._group_color)
		if self._draw_spike:
			RSDL.FillRect(surf, self._spikeRect, self._sdl_colors['white'])
		else:
			RSDL.FillRect(surf, self._spikeRect, color )#self._sdl_colors['black'])
		self._draw_spike = False




Bjitdriver = JitDriver(
	reds=['loops'], 
	greens='layers neurons pulse_layers'.split()
)

class Brain( object ):
	def loop( self ):
		self._active = True
		fmt = self.screen.c_format
		RSDL.FillRect(self.screen, lltype.nullptr(RSDL.Rect), self.ColorGrey)
		layers = self._layers
		neurons = self._neurons
		pulse_layers = self._pulse_layers
		screen = self.screen
		loops = 0	
		now = start = float( time.time() )
		while self._active:
			#Bjitdriver.can_enter_jit( layers=layers, neurons=neurons, pulse_layers=pulse_layers, loops=loops )
			#Bjitdriver.jit_merge_point( layers=layers, neurons=neurons, pulse_layers=pulse_layers, loops=loops)
			now = float( time.time() )
			self._fps = loops / float(now-start)
			for i,lay in enumerate(self._layers):
				if self._pulse_layers[i] and False:
					#print 'pulse layer: %s neurons: %s ' %(i, len(lay))
					for n in lay:
						if random()*random() > 0.8:
							n.spike( now )
			for i,col in enumerate(self._columns):
				if self._pulse_columns[i]:
					for n in col: n.spike(now)
			for n in self._neurons:
				n.iterate()
				n.draw(self.screen)
			#r,w,x = rpoll.select( [self._stdin], [], [], 1 )	# wait
			rl,wl,xl = rpoll.select( [0], [], [], 0.000001 )	# wait
			if rl:
				cmd = self._stdin.readline().strip('\n').strip(' ')
				self.do_command( cmd )
			loops += 1
			self._iterations = loops
			#print loops		# can not always print in mainloop, then select can never read from stdin
			RSDL.Flip(self.screen)
			#self._fps = float(time.time()) - now
		#return loops
		return 0

	def __init__(self):
		start = float(time.time())
		self._neurons = []
		self._columns = []
		self._layers = [ [] ] * LAYERS
		self._pulse_layers = [0] * LAYERS
		self._pulse_layers[ 0 ] = 1

		self._pulse_columns = [0] * COLUMNS
		self._pulse_columns[ 0 ] = 1
		self._pulse_columns[ 1 ] = 1
		self._pulse_columns[ 2 ] = 1
		self._pulse_columns[ 3 ] = 1


		inc = 360.0 / COLUMNS
		scale = float( LAYERS )
		expansion = 1.333
		linc = scale / LAYERS
		for column in range(COLUMNS):
			colNeurons = []
			self._columns.append( colNeurons )
			X = math.sin( radians(column*inc) )
			Y = math.cos( radians(column*inc) )
			expanding = STEM
			width = 1.0 / scale
			for layer in range(LAYERS):
				Z = layer * linc
				r = random() * random()
				g = 0.2
				b = 0.2
				for i in range(int(expanding)):
					x = uniform( -width, width )
					rr = random()*random()		# DJ's trick
					y = uniform( -width*rr, width*rr ) + X
					z = Z + Y
					# create 50/50 exitatory/inhibitory
					n = RecurrentSpikingModel(x=x, y=y, z=z, column=column, layer=layer, red=r, green=g, blue=b )
					self._neurons.append( n )
					colNeurons.append( n )
					self._layers[ layer ].append( n )

				expanding *= expansion
				width *= expansion

		dendrites = 0
		interlayer = 0
		for lay in self._layers:
			for a in lay:
				for b in lay:
					if a is not b and a._column == b._column:
						a.attach_dendrite( b )
						dendrites += 1
						interlayer += 1

		intercol = 0
		for col in self._columns:
			for a in col:
				for b in col:
					if a is not b and random()*random() > 0.75:
						a.attach_dendrite( b )
						intercol += 1
						dendrites += 1

		intercore = 0
		core = self._layers[-1]
		for a in core:
			for b in core:
				if a is not b and random()*random() > 0.85:
					a.attach_dendrite( b )
					intercore += 1
					dendrites += 1

		print 'brain creation time (seconds)', float(time.time())-start
		print 'neurons per column', len(self._columns[0])
		print 'inter-layer dendrites', interlayer
		print 'inter-column dendrites', intercol
		print 'inter-neocoretex dendrites', intercore
		print 'total dendrites', dendrites
		print 'total neurons', len(self._neurons)
		for i,lay in enumerate(self._layers):
			print 'layer: %s	neurons: %s' %(i,len(lay))
		for i,col in enumerate(self._columns):
			print 'column: %s	neurons: %s' %(i,len(col))



		self._stdin = streamio.fdopen_as_stream(0, 'r', 1)
		#self._stdout = streamio.fdopen_as_stream(1, 'w', 1)
		#self._stderr = streamio.fdopen_as_stream(2, 'w', 1)

		self._width = 640; self._height = 480
		assert RSDL.Init(RSDL.INIT_VIDEO) >= 0
		self.screen = RSDL.SetVideoMode(self._width, self._height, 32, 0)
		assert self.screen
		fmt = self.screen.c_format
		self.ColorWhite = white = RSDL.MapRGB(fmt, 255, 255, 255)
		self.ColorGrey = grey = RSDL.MapRGB(fmt, 128, 128, 128)
		self.ColorBlack = black = RSDL.MapRGB(fmt, 0, 0, 0)
		self.ColorBlue = blue = RSDL.MapRGB(fmt, 0, 0, 200)

		colors = {'white':white, 'grey':grey, 'black':black, 'blue':blue}

		x = 1; y = 1
		for i,n in enumerate(self._neurons):
			braneRect = RSDL_helper.mallocrect(x, y, 12, 12)
			groupRect = RSDL_helper.mallocrect(x, y, 12, 2)
			spikeRect = RSDL_helper.mallocrect(x+4, y+4, 4, 4)
			n.setup_draw( self.screen.c_format, braneRect, groupRect, spikeRect, colors )
			x += 13
			if x >= self._width-14:
				x = 1
				y += 13

	def do_command( self, cmd ):
		if cmd == 'spike-all':
			t = float(time.time())
			for n in self._neurons: n.spike(t)
		elif cmd == 'spike-one':
			t = float(time.time())
			self._neurons[0].spike(t)
		elif cmd == 'spike-column':
			t = float(time.time())
			for n in self._columns[0]:
				n.spike(t)
		elif cmd == 'info':
			info = self.info()
			#sys.stderr.write( info )
			#sys.stderr.flush()
			print info

	def info(self):
		r = ' "num-layers": %s,' %len(self._layers)
		r += ' "num-neurons": %s,' %len(self._neurons)
		r += ' "fps" : %s, ' %self._fps
		r += ' "iterations" : %s, ' %self._iterations
		return '<load_info> { %s }' %r






import subprocess, select, time
import gtk, glib

class App:
	def load_info( self, arg ): print arg

	def __init__(self):
		self._commands = cmds = []
		self.win = gtk.Window()
		self.win.connect('destroy', lambda w: gtk.main_quit())
		self.root = gtk.VBox(False,10); self.win.add( self.root )
		self.root.set_border_width(20)
		self.header = header = gtk.HBox()
		self.root.pack_start( header, expand=False )
		b = gtk.Button('spike all neurons')
		b.connect('clicked', lambda b,s: s._commands.append('spike-all'), self )
		self.header.pack_start( b, expand=False )

		b = gtk.Button('spike one neuron')
		b.connect('clicked', lambda b,s: s._commands.append('spike-one'), self )
		self.header.pack_start( b, expand=False )

		b = gtk.Button('spike column 1')
		b.connect('clicked', lambda b,s: s._commands.append('spike-column'), self )
		self.header.pack_start( b, expand=False )

		self.header.pack_start( gtk.SeparatorMenuItem() )

		b = gtk.Button('debug')
		b.connect('clicked', lambda b,s: s._commands.append('info'), self )
		self.header.pack_start( b, expand=False )

		da = gtk.DrawingArea()
		da.set_size_request( 640,480 )
		da.connect('realize', self.realize)
		self.root.pack_start( da )

		self._read = None
		glib.timeout_add( 33, self.loop )
		self.win.show_all()


	def realize(self, da ):
		print 'realize'
		xid = da.window.xid
		self._process = process = subprocess.Popen( 'python rAI.py --pypy --subprocess %s' %xid, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=32, shell=True )
		self._write = write = process.stdin
		self._read = read = process.stdout
		print 'read', read
		print 'write', write

	def loop( self ):
		if self._read:
			rlist,wlist,xlist = select.select( [self._read], [], [], 0.001 )
			while self._commands:
				cmd = self._commands.pop()
				print 'sending cmd ->', cmd
				self._write.write( '%s\n'%cmd )
				self._write.flush()
			if rlist:
				a = self._read.readline().strip()
				if a:
					print a
					if a.startswith('<'):
						func = a[ 1 : a.index('>') ]
						arg = a[ a.index('>')+1 : ].strip()
						func = getattr(self, func)
						func( eval(arg) )
		return True


if '--subprocess' in sys.argv:
	os.putenv('SDL_WINDOWID', sys.argv[-1])
	def pypy_entry_point():
		def jitpolicy(*args):
			from pypy.jit.metainterp.policy import JitPolicy
			return JitPolicy()

		brain = Brain()
		brain.loop()
	if '--pypy' in sys.argv:
		from pypy.translator.interactive import Translation
		t = Translation( pypy_entry_point )
		## NotImplementedError: --gcrootfinder=asmgcc requires standalone ##
		#t.config.translation.suggest(jit=True, jit_debug='steps', jit_backend='x86', gc='boehm')
		t.annotate()
		t.rtype()
		f = t.compile_c()
		f()
	else:
		pypy_entry_point()

else:
	a = App()
	gtk.main()
	print '-------------------exit toplevel-----------------'
