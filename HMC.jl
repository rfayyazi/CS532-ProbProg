include("./daphne.jl")
include("./primitives.jl")
include("./utils.jl")

using .Daphne
using .Primitives
using .Utils

using Distributions
using DataStructures
using Flux
using LinearAlgebra
using ProgressBars


primitive_procedures = get_primitives()
rho = Dict()  # user defined procedure environment
Y = Dict()
P = Dict()
# V = []
dists = Dict()


function build_dist_dict(X)
    ds = Dict()
    for (name, expr) in P
        d, sig, l = evaluate_expression(expr[2], Dict("log_W"=>0.0), merge(X, Y))
        ds[name] = d
    end
    return ds
end


function define_user_procedures(user_procedures)
    for fn in keys(user_procedures)
        global rho[fn] = Dict("arg_names"=>user_procedures[fn][2], "body"=>user_procedures[fn][3])
    end
end


function initialize(graph)
    define_user_procedures(graph[1])
    global P = deepcopy(graph[2]["P"])
    # global V = deepcopy(graph[2]["V"])
    _, l = sample_from_joint(graph)
    X = OrderedDict()
    for (k, v) in l
        if k[1] == 's'
            X[k] = v
        elseif k[1] == 'o'
            global Y[k] = v
        end
    end
    global dists = build_dist_dict(X)
    return X
end


function sample_from_joint(graph)
    _g = deepcopy(graph)
    # define_user_procedures(_g[1])
    rv_queue = topological_sort(_g[2]["V"], _g[2]["A"])
    sigma = Dict("log_W"=>0.0)
    l = Dict()
    for rv in rv_queue
        sample, sigma, l =  evaluate_expression(_g[2]["P"][rv], sigma, l)
        l[rv] = sample
    end
    sample, sigma, l = evaluate_expression(_g[3], sigma, l)
    return sample, l
end


function evaluate_expression(e, sigma, l)
    case = match_case(e)
    if case == "sample"
        d, sigma = evaluate_expression(e[2], sigma, l)
        sample = rand(d)
        if isa(d, DiscreteNonParametric)
            sample -= 1  # because get must use 0 based indexing to be consistent with daphne
        end
        return sample, sigma, l
    elseif case == "observe"
        d, sigma = evaluate_expression(e[2], sigma, l)
        c, sigma = evaluate_expression(e[3], sigma, l)
        sigma["log_W"] += logpdf(d, c)
        return c, sigma, l
    elseif case == "constant"
        return e, sigma, l
    elseif case == "variable"
        return l[e], sigma, l
    elseif case == "let"
        c1, sigma = evaluate_expression(e[2][2], sigma, l)
        l[e[2][1]] = c1
        return evaluate_expression(e[3], sigma, l)
    elseif case == "if"
        predicate, sigma = evaluate_expression(e[2], sigma, l)
        if predicate
            return evaluate_expression(e[3], sigma, l)
        else
            return evaluate_expression(e[4], sigma, l)
        end
    else
        for (i, arg) in enumerate(e[2:end])
            arg_evaluated, sigma = evaluate_expression(arg, sigma, l)
            e[i+1] = arg_evaluated
        end
        if case == "prim_procedure"
            return primitive_procedures[e[1]](e[2:end]...), sigma, l
        else
            arg_names = rho[e[1]]["arg_names"]
            body = deepcopy(rho[e[1]]["body"])
            for (i, arg) in enumerate(arg_names)
                l[arg] = e[i+1]
            end
            return evaluate_expression(body, sigma, l)
        end
    end
end


function match_case(e)
    if isa(e, Number) || isa(e, Distribution)
        return "constant"
    elseif isa(e, String)
        return "variable"
    else
        if isa(e[1], Number) || isa(e[1], Distribution) # ie constant vector
            return "constant"
        elseif e[1] == "sample*"
            return "sample"
        elseif e[1] == "observe*"
            return "observe"
        elseif e[1] == "let"
            return "let"
        elseif e[1] == "if"
            return "if"
        else
            if e[1] in keys(primitive_procedures)
                return "prim_procedure"
            else
                return "user_procedure"
            end
        end
    end
end


function E_U(X)
    E_X = 0.0
    E_Y = 0.0
    for x in keys(X)
        E_X += logpdf(dists[x], X[x])[1]
    end
    for y in keys(Y)
        E_Y += logpdf(dists[y], Y[y])[1]
    end
    -1.0 * (E_X + E_Y)
end


function leapfrog(X, R, T::Int, ϵ::Real)
    R -= 0.5 * ϵ * [v for v in values(gradient(E_U, X)[1])]
    for _ in 1:T
        for (i, k) in enumerate(keys(X))
            X[k] += ϵ * R[i]
        end
        R -= ϵ * [v for v in values(gradient(E_U, X)[1])]
    end
    for (i, k) in enumerate(keys(X))
        X[k] += ϵ * R[i]
    end
    R -= 0.5 * ϵ * [v for v in values(gradient(E_U, X)[1])]
    return X, R
end


pi_(X, R, σ) = exp(-E_U(X) + 0.5 * R'*(σ*Matrix(I, length(X), length(X)))*R)


function HMC(X::OrderedDict, S::Int, T::Int, ϵ::Real, σ::Real)
    #=
        S: number of samples
        T: number of leapfrog steps
        ϵ: discretization interval
        σ: variance for isotropic covariance matrix corresponding abs2(sig) * I
    =#
    n_X = length(X)
    X_trace = [X]
    n = 0
    for _ in ProgressBar(1:S)
        R = rand(MvNormal(n_X, σ))
        X_, R_ = leapfrog(deepcopy(X), deepcopy(R), T, ϵ)
        u = rand(Uniform(0, 1))
        α = pi_(X_, R_, σ) / pi_(X, R, σ)
        if u < α
            n += 1
            X = X_  # deepcopy(X_)
        end
        push!(X_trace, X)
    end
    return X_trace, n
end


function single_main(program_idx::Int, n_samples::Int)
    graph = daphne(["graph", "-i", "../CS532-ProbProg-Assignment/programs_HW3/$(program_idx).daphne"])
    X = initialize(graph)
    trace = HMC(X, n_samples, 15, 1.0, 1)
    return trace
end


trace, n = single_main(1, 10000)


# function main(n_samples::Number)
#     n_samples = convert(Int64, n_samples)
#     samples_dict = OrderedDict()
#     for i in 1:4  # 5
#         graph = daphne(["graph", "-i", "../CS532-ProbProg-Assignment/programs_HW3/$(i).daphne"])
#         samples_dict[i] = gibbs(graph, n_samples)
#         if isa(samples_dict[i][1], Bool)
#             samples_dict[i] = [convert(Int64, s) for s in samples_dict[i]]
#         end
#     end
#     return samples_dict, expectations_dict, variance_dict
# end


# samples_dict = main(1e5)
# expectations_dict, variance_dict = get_stats(samples_dict)
