module HopfieldNets
    export HopfieldNet, DiscreteHopfieldNet, ContinuousHopfieldNet
    export update!, energy, settle!, train!, associate!

    include("generic.jl")
    include("discrete.jl")
    include("continuous.jl")
end
