type DiscreteHopfieldNet <: HopfieldNet
    state::Vector{Float64} # Use float's to speed up computation
    weights::Matrix{Float64}
end

function DiscreteHopfieldNet(n::Integer)
    s = ones(n)
    W = zeros(n, n)
    DiscreteHopfieldNet(s, W)
end

# Perform one asynchronous update on randomly selected neuron
function update!(h::DiscreteHopfieldNet)
    i = rand(1:length(h.state))
    h.state[i] = dot(h.weights[:, i], h.state) > 0 ? +1 : -1
    return
end

# Hebbian updates
# Columns are patterns
# TODO: Can this be shared with ContinuousHopfielfNet?
function train!(h::DiscreteHopfieldNet, patterns::Matrix{Float64})
    p = size(patterns, 2)
    n, n = size(h.weights)
    # NB: Could use outer products here
    # h.weights += (1 / p) * (patterns[:, mu] * patterns[:, mu]')
    for i in 1:n
        for j in 1:n
            s = 0.0
            for mu in 1:p
                s += patterns[i, mu] * patterns[j, mu]
            end
            s = s / p
            h.weights[i, j] += s
            h.weights[j, i] += s
        end
    end
    for i in 1:n
        h.weights[i, i] = 0.0
    end
    return
end
