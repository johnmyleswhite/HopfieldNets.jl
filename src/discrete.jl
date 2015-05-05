type DiscreteHopfieldNet <: HopfieldNet
    s::Vector{Float64} # State -- could have used Int's
    W::Matrix{Float64} # Weights
end

function DiscreteHopfieldNet(n::Integer)
    s = ones(n)
    W = zeros(n, n)
    DiscreteHopfieldNet(s, W)
end

# Perform one asynchronous update on randomly selected neuron
function update!(net::DiscreteHopfieldNet)
    i = rand(1:length(net.s))
    net.s[i] = dot(net.W[:, i], net.s) > 0 ? +1 : -1
    return nothing
end

# Perform one asynchronous update on a specific neuron
function update!(net::DiscreteHopfieldNet, i::Int)
    @assert i in 1:length(net.s) "Neuron index $i out of range"
    net.s[i] = dot(net.W[:, i], net.s) > 0 ? +1 : -1
    return nothing
end

# Perform one asynchronous update on a specific neuron without modifying the network
function update(net::DiscreteHopfieldNet, i::Int)
    @assert i in 1:length(net.s) "Neuron index $i out of range"
    return dot(net.W[:, i], net.s) > 0 ? +1 : -1
end

function show(io::IO, net::DiscreteHopfieldNet)
    @printf io "A discrete Hopfield net with %d neurons" length(net.s)
end
