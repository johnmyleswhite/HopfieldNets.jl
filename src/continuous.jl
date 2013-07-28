type ContinuousHopfieldNet <: HopfieldNet
    state::Vector{Float64}
    weights::Matrix{Float64}
end

function ContinuousHopfieldNet(n::Integer)
    s = ones(n)
    W = zeros(n, n)
    ContinuousHopfieldNet(s, W)
end

# Asynchronous, random updates
function update!(h::ContinuousHopfieldNet)
    i = rand(1:length(h.state))
    h.state[i] = tanh(dot(h.weights[:, i], h.state))
    return
end

# Hebbian updates
# Columns are patterns
# TODO: Can this be shared with ContinuousHopfielfNet?
function train!(h::ContinuousHopfieldNet, patterns::Matrix{Float64})
    p = size(patterns, 2)
    n, n = size(h.weights)
    # Could use outer products here
    # (1 / p) * (patterns[:, mu] * patterns[:, mu]')
    for i in 1:n
        for j in 1:n
            s = 0.0
            for mu in 1:p
                s += patterns[i, mu] * patterns[j, mu]
            end
            s = s / p # May need to be careful here
            h.weights[i, j] += s
            h.weights[j, i] += s
        end
    end
    for i in 1:n
        h.weights[i, i] = 0.0
    end
    return
end
