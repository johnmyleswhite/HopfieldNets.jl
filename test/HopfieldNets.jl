using HopfieldNets
using Base.Test

include(Pkg.dir("HopfieldNets", "demo", "letters.jl"))

patterns = hcat(X, O)

n = size(patterns, 1)

dh = DiscreteHopfieldNet(n)
ch = ContinuousHopfieldNet(n)

for h in [dh, ch]
    train!(h, patterns)

    e0 = energy(h)
    settle!(h, 1_000, false)
    eFinal = energy(h)
    @assert e0 != eFinal

    Xcorrupt = copy(X)
    for i = 2:7
         Xcorrupt[i] = 1
    end
    Xrestored = associate!(h, Xcorrupt)
    @test !all(Xcorrupt .== Xrestored)
    @test all(X .== Xrestored)

    Ocorrupt = copy(O)
    for i = 2:7
         Ocorrupt[i] = -1
    end
    Orestored = associate!(h, Ocorrupt)
    @test !all(Ocorrupt .== Orestored)
    @test all(O .== Orestored)
end
