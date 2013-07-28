abstract HopfieldNet

function energy(h::HopfieldNet)
    e = 0.0
    n = length(h.state)
    for i in 1:n
        for j in 1:n
            e += h.weights[i, j] * h.state[i] * h.state[j]
        end
    end
    e *= -0.5
    for i in 1:n
        e += h.weights[i] * h.state[i]
    end
    return e
end

function settle!(h::HopfieldNet,
                 iterations::Integer = 1_000,
                 trace::Bool = false)
    for i in 1:iterations
        update!(h)
        if trace
            @printf "%5.0d: %.4f\n" i energy(h)
        end
    end
    return
end

function associate!(h::HopfieldNet,
                    pattern::Vector{Float64};
                    iterations::Integer = 1_000,
                    trace::Bool = false)
    copy!(h.state, pattern)
    settle!(h, iterations, trace)
    # TODO: Decide if this should really be a copy
    return copy(h.state)
end

function Base.show(io::IO, h::HopfieldNet)
    @printf io "A Hopfield net with %d neurons\n" length(h.state)
end
