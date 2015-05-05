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
    return nothing
end

# Perform one asynchronous update on a specific neuron
function update!(net::ContinuousHopfieldNet, i::Int)
    @assert i in 1:length(net.s) "Neuron index $i out of range"
    net.s[i] = tanh(dot(net.W[:, i], net.s))
    return nothing
end

# Perform one asynchronous update on a specific neuron without modifying the network
function update(net::ContinuousHopfieldNet, i::Int)
    @assert i in 1:length(net.s) "Neuron index $i out of range"
    return tanh(dot(net.W[:, i], net.s))
end

function show(io::IO, net::ContinuousHopfieldNet)
    @printf io "A continuous Hopfield net with %d neurons" length(net.s)
end
