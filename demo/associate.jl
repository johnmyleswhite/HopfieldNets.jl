using HopfieldNets

include(Pkg.dir("HopfieldNets", "demo", "letters.jl"))
include(Pkg.dir("HopfieldNets", "demo", "display.jl"))

patterns = hcat(X, O)

n = size(patterns, 1)

h = DiscreteHopfieldNet(n)

train!(h, patterns)

settle!(h, 10, true)

# Restore corrupt images

Xcorrupt = copy(X)
for i = 2:7
     Xcorrupt[i] = 1
end

Ocorrupt = copy(O)
for i = 2:7
     Ocorrupt[i] = -1
end

display(reshape(X, 7, 6))
display(reshape(Xcorrupt, 7, 6))
display(reshape(associate!(h, Xcorrupt), 7, 6))

display(reshape(O, 7, 6))
display(reshape(Ocorrupt, 7, 6))
display(reshape(associate!(h, Ocorrupt), 7, 6))

# Associate new patterns with familiar patterns

display(reshape(F1, 7, 6))
display(reshape(associate!(h, F1), 7, 6))

display(reshape(F2, 7, 6))
display(reshape(associate!(h, F2), 7, 6))
