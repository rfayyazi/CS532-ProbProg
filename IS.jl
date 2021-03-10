include("./daphne.jl")
include("./primitives.jl")
include("./analysis.jl")

using .Daphne
using .Primitives
using .Analysis

using Distributions
using DataStructures
using Plots
using JSON
using ProgressBars

# initialize global environments
rho = Dict()  # user defined procedure environment
primitive_procedures = get_primitives()


function define_user_procedures(ast)
    while length(ast) > 1
        user_procedure = popfirst!(ast)
        rho[user_procedure[2]] = Dict("arg_names"=>user_procedure[3], "body"=>user_procedure[4])
    end
    return ast
end


function evaluate_expression(e, sigma, l)
    case = match_case(e)
    if case == "sample"
        d, sigma = evaluate_expression(e[2], sigma, l)
        sample = rand(d)
        if isa(d, DiscreteNonParametric)
            sample -= 1  # because get must use 0 based indexing to be consistent with daphne
        end
        return sample, sigma
    elseif case == "observe"
        d, sigma = evaluate_expression(e[2], sigma, l)
        c, sigma = evaluate_expression(e[3], sigma, l)
        sigma["log_W"] += logpdf(d, c)
        return c, sigma
    elseif case == "constant"
        return e, sigma
    elseif case == "variable"
        return l[e], sigma
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
            return primitive_procedures[e[1]](e[2:end]...), sigma
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
        elseif e[1] == "sample"
            return "sample"
        elseif e[1] == "observe"
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

function likelihood_weighting(ast, n_samples::Int)
    _ast = deepcopy(ast)
    _ast = define_user_procedures(_ast)
    _ast = _ast[1]
    # sigma = Dict("log_W"=>0.0)
    samples = []
    for i in ProgressBar(1:n_samples)
        l = Dict()
        sigma = Dict("log_W"=>0.0)
        sample, sigma = evaluate_expression(deepcopy(_ast), sigma, l)
        push!(samples, (sample, sigma["log_W"]))
    end
    return samples
end


function get_samples(program_idx::Int, n_samples::Int)
    ast = daphne(["desugar", "-i", "../CS532-ProbProg-Assignment/programs_HW3/$(program_idx).daphne"])
    samples = likelihood_weighting(ast, n_samples)
    return samples
end


function main(n_samples::Number)
    n_samples = convert(Int64, n_samples)
    samples_dict = OrderedDict()
    expectations_dict = OrderedDict()
    variance_dict = OrderedDict()
    for i in 1:4  # 5
        # print("Testing on program $(i) ... ")
        samples_dict[i] = get_samples(i, n_samples)
        if isa(samples_dict[i][1][1], Bool)
            samples_dict[i] = [(convert(Int64, s), w) for (s, w) in samples_dict[i]]
        end
        expectations_dict[i] = compute_expectation(samples_dict[i])
        variance_dict[i] = compute_variance(samples_dict[i])
        # print("Done. \n")
    end
    return samples_dict, expectations_dict, variance_dict
end


# function sample_histograms(save_path::String, samples_dict::OrderedDict)
#     for (k, v) in samples_dict
#         for
#     end
# end


samples_dict, expectations_dict, variance_dict = main(1e4)

# expectations_dict
# 1 => 7.10179
# 2 => [2.11181, -0.346656]
# 3 => 0.424771
# 4 => 0.312953

# variance_dict
# 1 => 0.661564
# 2 => [0.0568294, 0.910442]
# 3 => 0.244341
# 4 => 0.215014
