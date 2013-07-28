using HopfieldNets
using Base.Test

include(Pkg.dir("HopfieldNets", "demo", "letters.jl"))

patterns = hcat(X, O)

n = size(patterns, 1)

dnet = DiscreteHopfieldNet(n)
cnet = ContinuousHopfieldNet(n)

for net in [dnet] # Why does the training break on cnet?
    storkeytrain!(net, patterns)

    e0 = energy(net)
    settle!(net, 1_000, false)
    eFinal = energy(net)
    @assert e0 != eFinal

    Xcorrupt = copy(X)
    for i = 2:7
         Xcorrupt[i] = 1
    end
    Xrestored = associate!(net, Xcorrupt)
    @test norm(Xcorrupt - Xrestored) > 1e-4
    @test norm(X - Xrestored) < 1e-4

    Ocorrupt = copy(O)
    for i = 2:7
         Ocorrupt[i] = -1
    end
    Orestored = associate!(net, Ocorrupt)
    @test norm(Ocorrupt - Orestored) > 1e-4
    @test norm(O - Orestored) < 1e-4
end
