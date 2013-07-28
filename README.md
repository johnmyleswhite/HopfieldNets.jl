HopfieldNets.jl
===============

A simple implementation of Hopfield nets in Julia using Hebbian learning.

The idea for the letters demo is taken from an implementation of [Hopfield nets in Ruby](https://github.com/bartolsthoorn/hopfield-ruby). The code is based on the presentation of the theory in Mackay's book.

# Usage Example

We'll restore a corrupted representation fo the letter X specified as a vector of -1's and +1's:

	using HopfieldNets

	include(Pkg.dir("HopfieldNets", "demo", "letters.jl"))

	patterns = hcat(X, O)

	n = size(patterns, 1)

	h = DiscreteHopfieldNet(n)

	train!(h, patterns)

	settle!(h, 10, true)

	Xcorrupt = copy(X)
	for i = 2:7
	     Xcorrupt[i] = 1
	end

	Xrestored = associate!(h, Xcorrupt)
	all(Xcorrupt .== X)
	all(Xrestored .== X)
