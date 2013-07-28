type ContinuousHopfieldNet <: HopfieldNet
    s::Vector{Float64} # State
    W::Matrix{Float64} # Weights
end

function ContinuousHopfieldNet(n::Integer)
    s = ones(n)
    W = zeros(n, n)
    ContinuousHopfieldNet(s, W)
end

# Perform one asynchronous update on randomly selected neuron
function update!(net::ContinuousHopfieldNet)
    i = rand(1:length(net.s))
    net.s[i] = tanh(dot(net.W[:, i], net.s))
    return
end

function Base.show(io::IO, net::ContinuousHopfieldNet)
    @printf io "A continuous Hopfield net with %d neurons\n" length(net.s)
end
