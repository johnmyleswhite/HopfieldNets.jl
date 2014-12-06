HopfieldNets.jl
===============

# NOTICE

**This package is unmaintained. Its reliability is not guaranteed.**

# Introduction

Discrete and continuous Hopfield nets in Julia.

# Usage Example

We'll restore a corrupted representation fo the letter X specified as a vector of -1's and +1's:

	using HopfieldNets

	include(Pkg.dir("HopfieldNets", "demo", "letters.jl"))

	patterns = hcat(X, O)

	n = size(patterns, 1)

	net = DiscreteHopfieldNet(n)

	train!(net, patterns)

	settle!(net, 10, true)

	Xcorrupt = copy(X)
	for i = 2:7
	     Xcorrupt[i] = 1
	end

	Xrestored = associate!(net, Xcorrupt)
	all(Xcorrupt .== X)
	all(Xrestored .== X)

# Sources

The idea for the letters demo is taken from an implementation of [Hopfield nets in Ruby](https://github.com/bartolsthoorn/hopfield-ruby). The code is based on the presentation of the theory in Mackay's book and in Storkey's learning rule paper.
