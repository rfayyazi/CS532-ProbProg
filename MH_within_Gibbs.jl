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


function get_x_dependents(graph, k)
    dependents = [k]
    for n in graph[2]["A"][k]
        if n[1] == 's'
            push!(dependents, n)
        end
    end
    return dependents
end


function accept(graph, k, X, X_, Y)
    q  = evaluate_expression(deepcopy(graph[2]["P"][k][2]), Dict("log_W"=>0.0), X )[1]
    q_ = evaluate_expression(deepcopy(graph[2]["P"][k][2]), Dict("log_W"=>0.0), X_)[1]
    log_α = logpdf(q_, X[k]) - logpdf(q, X_[k])
    Vx = [graph[2]["A"][k]; k]
    # Vx = get_x_dependents(graph, k)
    V  = merge(X, Y)
    V_ = merge(X_, Y)
    for v in Vx
        p  = evaluate_expression(deepcopy(graph[2]["P"][v][2]), Dict("log_W"=>0.0), V )[1]
        p_ = evaluate_expression(deepcopy(graph[2]["P"][v][2]), Dict("log_W"=>0.0), V_)[1]
        log_α += logpdf(p_, V_[v])
        log_α -= logpdf(p, V[v])
    end
    return exp(log_α)
end


function gibbs_step(graph, X, Y)
    for k in keys(X)
        q = evaluate_expression(graph[2]["P"][k][2], Dict("log_W"=>0.0), X)[1]
        X_ = deepcopy(X)
        X_[k] = rand(q)
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
    X, Y = init_vars(graph)
    push!(samples, X)
    for i in ProgressBar(1:n_samples)
        X = gibbs_step(graph, X, Y)
        push!(samples, X)
    end
    return samples
end


function init_vars(graph)
    samples, l = sample_from_joint(graph)
    X = Dict()
    Y = Dict()
    for (k, v) in l
        if k[1] == 's'
            X[k] = v
        elseif k[1] == 'o'
            Y[k] = v
        end
    end
    return X, Y
end


function get_Q(graph, X)
    Q = Dict()
    for (label, expr) in graph[2]["P"]
        if label[1] == 's'
            Q[label] = evaluate_expression(expr[2], Dict("log_W"=>0.0), X)[1]
        end
    end
    return Q
end


function get_samples(program_idx::Int, n_samples::Int)
    graph = daphne(["graph", "-i", "../CS532-ProbProg-Assignment/programs_HW3/$(program_idx).daphne"])
    samples = gibbs(graph, n_samples)
    return samples
end


function main(n_samples::Number)
    n_samples = convert(Int64, n_samples)
    samples_dict = OrderedDict()
    expectations_dict = OrderedDict()
    variance_dict = OrderedDict()
    for i in 1:4  # 5
        print("Testing on program $(i) ... ")
        # samples_dict[i] = get_samples(i, n_samples)
        graph = daphne(["graph", "-i", "../CS532-ProbProg-Assignment/programs_HW3/$(program_idx).daphne"])
        samples_dict[i] = gibbs(graph, n_samples)
        if isa(samples_dict[i][1][1], Bool)
            samples_dict[i] = [(convert(Int64, s), w) for (s, w) in samples_dict[i]]
        end
        expectations_dict[i] = mean([d["sample2"] for d in samples])
        # variance_dict[i] = compute_variance(samples_dict[i])
        print("Done. \n")
    end
    return samples_dict, expectations_dict  # , variance_dict
end

# samples_dict, expectations_dict, variance_dict = main(10)
# samples = get_samples(1, 1e7)
# mean([d["sample2"] for d in samples])
