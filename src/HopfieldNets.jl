module HopfieldNets

    import Base: show
    export HopfieldNet, DiscreteHopfieldNet, ContinuousHopfieldNet,
           LearningAlgorithm, Hebbian, Storkey, update!, update, energy,
           settle!, syncsettle!, train!, associate!, show

    include("generic.jl")
    include("discrete.jl")
    include("continuous.jl")
end
