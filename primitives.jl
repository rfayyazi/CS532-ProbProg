module Primitives

export get_primitives, NormalMix, nm_cdf

using Distributions


struct NormalMix
    normals::Array
    wts::Array
end


function NormalMix(params...)
    normals = []
    wts = []
    for i in 1:convert(Int64, length(params) / 3)
        push!(normals, Normal(params[3*i-1], params[3*i]))
        push!(wts, params[3*i-2])
    end
    return NormalMix(normals, wts)
end


function nm_cdf(nm::NormalMix, x::Array)
    cdf_vals = []
    for (wt, normal) in zip(nm.wts, nm.normals)
        push!(cdf_vals, wt * cdf.(normal, x))
    end
    return sum(cdf_vals)
end


function Categorical_custom(x)
    x = Array{Real}(x)
    return Categorical(x)
end


function get_primitives()
    primitive_procedures = Dict()

    # basic math operators
    primitive_procedures["+"] = +
    primitive_procedures["-"] = -
    primitive_procedures["*"] = *
    primitive_procedures["/"] = /
    primitive_procedures["sqrt"] = sqrt

    # logical opertators
    primitive_procedures["<"] = <
    primitive_procedures[">"] = >

    # linear algebra opertators
    primitive_procedures["mat-transpose"] = permutedims
    primitive_procedures["mat-add"] = +
    primitive_procedures["mat-mul"] = *
    primitive_procedures["mat-repmat"] = repeat
    primitive_procedures["mat-tanh"] = x -> map(tanh, x)

    # data structure constructors
    primitive_procedures["vector"] = vector
    primitive_procedures["hash-map"] = (x...) -> Dict([([x...][i*2-1], [x...][i*2]) for i in 1:convert(Int64, length([x...])/2)])

    # data structure helpers
    primitive_procedures["get"] = get
    primitive_procedures["put"] = put
    primitive_procedures["first"] = first
    primitive_procedures["second"] = x -> x[2]
    # primitive_procedures["nth"] = (x, n) -> x[n+1]
    primitive_procedures["rest"] = x -> x[2:end]
    primitive_procedures["last"] = last
    primitive_procedures["append"] = push!
    # primitive_procedures["conj"] = push!
    # primitive_procedures["cons"] = pushfirst!

    # distributions
    primitive_procedures["uniform"] = Uniform
    primitive_procedures["discrete"] = (x...) -> Categorical(Array{Real}(x...))
    primitive_procedures["bernoulli"] = Bernoulli
    primitive_procedures["normal"] = Normal
    primitive_procedures["beta"] = Beta
    primitive_procedures["exponential"] = Exponential
    primitive_procedures["normalmix"] = NormalMix

    return primitive_procedures
end


function vector(x...)
    v = Array{Any}([x...])
    if isa(v[1], Array)  # deal with matrix construction in 4.daphne
        if length(v[1]) == 1  # ie W_0 or x or columns of W_1
            return reshape(vcat(v...), (size(v)..., 1))
        else
            if length(v) > 1  # ie W_1
                return hcat(v...)
            else  # ie W_2
                return reshape(vcat(v...), (1, size(v[1])...))
            end
        end
    end
    return v
end


function get(x, i)
    if isa(x, Dict)
        return x[i]
    else
        return x[i+1]  # daphne uses 0 based indexing
    end
end


function put(x, i, e)
    if isa(x, Dict)
        x[i] = e
        return x
    else
        x[i+1] = e  # daphne uses 0 based indexing
        return x
    end
end


end
