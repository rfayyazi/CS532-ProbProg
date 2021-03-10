include("./daphne.jl")
include("./primitives.jl")
include("./analysis.jl")
include("./utils.jl")

using .Daphne
using .Primitives
using .Analysis
using .Utils

using Distributions
using DataStructures
using ProgressBars


rho = Dict()  # user defined procedure environment
primitive_procedures = get_primitives()


function sample_from_joint(graph)
    _g = deepcopy(graph)
    define_user_procedures(_g[1])
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


function define_user_procedures(user_procedures)
    for fn in keys(user_procedures)
        rho[fn] = Dict("arg_names"=>user_procedures[fn][2], "body"=>user_procedures[fn][3])
    end
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


function init_vars(graph)
    samples, l = sample_from_joint(deepcopy(graph))
    X = Dict()
    Y = Dict()
    for (k, v) in l
        if k[1] == 's'
            X[k] = v
        elseif k[1] == 'o'
            Y[k] = v
        end
    end
    return X, Y, samples
end


function accept(graph, k, X, X_, Y)
    q  = evaluate_expression(deepcopy(graph[2]["P"][k][2]), Dict("log_W"=>0.0), X )[1]
    q_ = evaluate_expression(deepcopy(graph[2]["P"][k][2]), Dict("log_W"=>0.0), X_)[1]
    if isa(q, DiscreteNonParametric)
        log_α = logpdf(q_, X[k] + 1) - logpdf(q, X_[k] + 1)
    else
        log_α = logpdf(q_, X[k]) - logpdf(q, X_[k])
    end
    Vx = [graph[2]["A"][k]; k]
    V  = merge(X, Y)
    V_ = merge(X_, Y)
    for v in Vx
        p  = evaluate_expression(deepcopy(graph[2]["P"][v][2]), Dict("log_W"=>0.0), V )[1]
        p_ = evaluate_expression(deepcopy(graph[2]["P"][v][2]), Dict("log_W"=>0.0), V_)[1]
        if isa(p, DiscreteNonParametric)
            log_α += logpdf(p_, V_[v] + 1)
            log_α -= logpdf(p, V[v] + 1)
        else
            log_α += logpdf(p_, V_[v])
            log_α -= logpdf(p, V[v])
        end
    end
    return exp(log_α)
end


function gibbs_step(graph, X, Y)
    for k in keys(X)
        if k == "sample6" || k == "sample7" || k == "sample9"
            stop = true
        end
        X_ = deepcopy(X)
        q = evaluate_expression(deepcopy(graph[2]["P"][k][2]), Dict("log_W"=>0.0), X)[1]
        sample = rand(q)
        if isa(q, DiscreteNonParametric)
            sample -= 1  # because get must use 0 based indexing to be consistent with daphne
        end
        X_[k] = sample
        α = accept(graph, k, X, X_, Y)
        u = rand(Uniform(0, 1))
        if u < α
            X = X_
        end
    end
    return X
end


function gibbs(graph, n_samples)
    samples = []
    X, Y, sample = init_vars(deepcopy(graph))
    push!(samples, sample)
    for i in ProgressBar(1:n_samples)
        X = gibbs_step(deepcopy(graph), X, Y)
        sample = evaluate_expression(deepcopy(graph[3]), Dict("log_W"=>0.0), X)[1]
        push!(samples, sample)
    end
    return samples
end


function get_stats(sample_streams::OrderedDict)
    expectations_dict = OrderedDict()
    variance_dict = OrderedDict()
    for (k, v) in sample_streams
        expectations_dict[k] = mean(samples_dict[k])
        variance_dict[k] = mean([s .^ 2 for s in samples_dict[k]]) - (expectations_dict[k] .^ 2)
    end
    return expectations_dict, variance_dict
end


function main(n_samples::Number)
    n_samples = convert(Int64, n_samples)
    samples_dict = OrderedDict()
    for i in 1:4  # 5
        graph = daphne(["graph", "-i", "../CS532-ProbProg-Assignment/programs_HW3/$(i).daphne"])
        samples_dict[i] = gibbs(graph, n_samples)
        if isa(samples_dict[i][1], Bool)
            samples_dict[i] = [convert(Int64, s) for s in samples_dict[i]]
        end
    end
    return samples_dict
end


samples_dict = main(1e5)
expectations_dict, variance_dict = get_stats(samples_dict)


# expectations_dict with 100 samples
# 1 => 7.30177
# 2 => [2.15636, -0.53384]
# 3 => 0.690553
# 4 => 0.321377

# variance_dict with 100 samples
# 1 => 0.892789
# 2 => [0.0624608, 0.925648]
# 3 => 0.21369
# 4 => 0.218094
