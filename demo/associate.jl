using HopfieldNets

include(Pkg.dir("HopfieldNets", "demo", "letters.jl"))
include(Pkg.dir("HopfieldNets", "demo", "display.jl"))

patterns = hcat(X, O)

n = size(patterns, 1)

net = DiscreteHopfieldNet(n)

train!(net, patterns)

settle!(net, 10, true)

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
display(reshape(associate!(net, Xcorrupt), 7, 6))

display(reshape(O, 7, 6))
display(reshape(Ocorrupt, 7, 6))
display(reshape(associate!(net, Ocorrupt), 7, 6))

# Associate new patterns with familiar patterns

display(reshape(F1, 7, 6))
display(reshape(associate!(net, F1), 7, 6))

display(reshape(F2, 7, 6))
display(reshape(associate!(net, F2), 7, 6))
