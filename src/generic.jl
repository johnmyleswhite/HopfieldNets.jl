abstract HopfieldNet

function energy(net::HopfieldNet)
    e = 0.0
    n = length(net.s)
    for i in 1:n
        for j in 1:n
            e += net.W[i, j] * net.s[i] * net.s[j]
        end
    end
    e *= -0.5
    for i in 1:n
        e += net.W[i] * net.s[i]
    end
    return e
end

function settle!(net::HopfieldNet,
                 iterations::Integer = 1_000,
                 trace::Bool = false)
    for i in 1:iterations
        update!(net)
        if trace
            @printf "%5.0d: %.4f\n" i energy(net)
        end
    end
    return
end

function associate!{T <: Real}(net::HopfieldNet,
                               pattern::Vector{T};
                               iterations::Integer = 1_000,
                               trace::Bool = false)
    copy!(net.s, pattern)
    settle!(net, iterations, trace)
    # TODO: Decide if this should really be a copy
    return copy(net.s)
end

# Hebbian learning steps w/ columns as patterns
function train!{T <: Real}(net::HopfieldNet, patterns::Matrix{T})
    n = length(net.s)
    p = size(patterns, 2)
    # Could use outer products here
    # (1 / p) * (patterns[:, mu] * patterns[:, mu]')
    for i in 1:n
        for j in (i + 1):n
            s = 0.0
            for mu in 1:p
                s += patterns[i, mu] * patterns[j, mu]
            end
            s = s / p # May need to be careful here
            net.W[i, j] += s
            net.W[j, i] += s
        end
    end
    return
end

function h{T <: Real}(i::Integer, j::Integer, mu::Integer, n::Integer,
                      W::Matrix{Float64}, patterns::Matrix{T})
    res = 0.0
    for k in 1:n
        if k != i && k != j
            res += W[i, k] * patterns[k, mu]
        end
    end
    return res
end

# Storkey learning steps w/ columns as patterns
function storkeytrain!{T <: Real}(net::HopfieldNet, patterns::Matrix{T})
    n = length(net.s)
    p = size(patterns, 2)
    for i in 1:n
        for j in (i + 1):n
            for mu in 1:p
                s = patterns[i, mu] * patterns[j, mu]
                s -= patterns[i, mu] * h(j, i, mu, n, net.W, patterns)
                s -= h(i, j, mu, n, net.W, patterns) * patterns[j, mu]
                s *= 1 / n
                net.W[i, j] += s
                net.W[j, i] += s
            end
        end
    end
    return
end
