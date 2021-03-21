module Primitives

export get_primitives, NormalMix, nm_cdf

using Distributions
using FunctionalCollections


struct NormalMix
    normals::Array
    wts::Array
end


function get_primitives()
    primitive_procedures = Dict()

    primitive_procedures["alpha"] = ""
    primitive_procedures["push-address"] = (x, y) -> x * y

    # basic math operators
    primitive_procedures["+"] = +
    primitive_procedures["-"] = -
    primitive_procedures["*"] = *
    primitive_procedures["/"] = /
    primitive_procedures["sqrt"] = x -> sqrt.(x)
    primitive_procedures["log"] = x -> log.(x)

    # control flow operators
    primitive_procedures["if"] = (pred, cons, alt) -> pred ? cons : alt

    # logical opertators
    primitive_procedures["="] = isequal
    primitive_procedures["<"] = <
    primitive_procedures[">"] = >
    primitive_procedures["and"] = (x, y) -> x && y
    primitive_procedures["or"] = (x, y) -> x || y

    # linear algebra opertators
    primitive_procedures["mat-add"] = mat_add
    primitive_procedures["mat-mul"] = mat_mul
    primitive_procedures["mat-tanh"] = mat_tanh
    primitive_procedures["mat-transpose"] = mat_transpose
    primitive_procedures["mat-repmat"] = mat_repmat

    # data structure constructors
    # primitive_procedures["vector"] = (x...) -> @Persistent{Any} [x...]
    primitive_procedures["vector"] = vector
    primitive_procedures["hash-map"] = hash_map

    # data structure helpers
    primitive_procedures["empty?"] = isempty
    primitive_procedures["get"] = get
    primitive_procedures["put"] = put
    primitive_procedures["first"] = first
    primitive_procedures["second"] = x -> x[2]
    primitive_procedures["rest"] = x -> x[2:end]
    primitive_procedures["last"] = last
    primitive_procedures["peek"] = last
    primitive_procedures["append"] = append
    primitive_procedures["conj"] = conjunct

    # distribution constructors
    primitive_procedures["uniform"] = Uniform
    primitive_procedures["uniform-continuous"] = Uniform
    primitive_procedures["discrete"] = x -> Categorical(Array{Real}([(x ./ sum(x))...]))
    primitive_procedures["flip"] = Bernoulli
    primitive_procedures["normal"] = Normal
    primitive_procedures["beta"] = Beta
    primitive_procedures["gamma"] = Gamma
    primitive_procedures["exponential"] = Exponential
    primitive_procedures["dirichlet"] = (x...) -> Dirichlet(Array{Real}(x...))
    primitive_procedures["dirac"] = Dirac
    primitive_procedures["normalmix"] = NormalMix

    return primitive_procedures
end


function vector(x...)
    v = @Persistent []
    for e in x
        v = append(v, e)
    end
    return v
end


function hash_map(x...)
    hm = @Persistent Dict()
    for i in 1:2:length(x)
        hm = assoc(hm, x[i], x[i+1])
    end
    return hm
end


function get(x, i)
    if isa(x, PersistentHashMap)
        return x[i]
    else
        return x[i+1]  # daphne uses 0 based indexing
    end
end


function conjunct(x, y)
    v = @Persistent [y]
    append(v, x)
end


function put(x, i, e)
    if isa(x, PersistentHashMap)
        x = assoc(x, i, e)
        return x
    else
        x = assoc(x, i+1, e)
        return x
    end
end


to_mat(x) = [r[i] for r in x, i in 1:length(x[1])]

to_v_of_v(x) = [x[i, :] for i in 1:size(x, 1)]


mat_add(x, y) = to_v_of_v(to_mat(x) + to_mat(y))

mat_mul(x, y) = to_v_of_v(to_mat(x) * to_mat(y))

mat_tanh(x) = to_v_of_v(tanh.to_mat(x))

mat_transpose(x) = to_v_of_v(transpose(to_mat(x)))

mat_repmat(x, d...) = to_v_of_v(repeat(to_mat(x), d...))


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

end
