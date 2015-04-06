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

function show(io::IO, net::DiscreteHopfieldNet)
    @printf io "A discrete Hopfield net with %d neurons" length(net.s)
end
