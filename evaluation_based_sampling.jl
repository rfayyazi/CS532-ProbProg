include("./daphne.jl")
include("./primitives.jl")
include("./tests.jl")
include("./analysis.jl")

using .Daphne
using .Primitives
using .Tests
using .Analysis

using ProgressBars
using Distributions
using ResumableFunctions


# initialize global environments
rho = Dict()  # user defined procedure environment
primitive_procedures = get_primitives()


function evaluate_program(ast)
    _ast = deepcopy(ast)
    _ast = define_user_procedures(_ast)
    sigma = Dict("log_W"=>0.0)
    l = Dict()
    return evaluate_expression(_ast[1], sigma, l)
end


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
            body = rho[e[1]]["body"]
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


@resumable function get_stream(ast, n)
    for i in 1:convert(Int64, n)
        sample, sigma = evaluate_program(ast)
        @yield sample
    end
end


function run_deterministic_tests()
    ast_list = []
    truth_list = []
    ret_list = []
    n_tests = 13
    pbar = ProgressBar(1:n_tests)
    set_description(pbar, "Deterministic tests")
    for i in pbar
        ast = daphne(["desugar", "-i", "../HW2/programs/tests/deterministic/test_$(i).daphne"])
        truth = load_truth("programs/tests/deterministic/test_$(i).truth")
        ret, sigma = evaluate_program(ast)
        try
            @assert istol(ret, truth)
        catch
            AssertionError("Returned value differs from truth for test $(i)")
        end
        push!(ast_list, ast)
        push!(truth_list, truth)
        push!(ret_list, ret)
    end
    return ast_list, truth_list, ret_list
end


function run_probabilistic_tests()
    pval_list = []
    num_samples = 1e4
    max_p_value = 1e-4
    n_tests = 6
    pbar = ProgressBar(1:n_tests)
    set_description(pbar, "Probabilistic tests")
    for i in pbar
        ast = daphne(["desugar", "-i", "../HW2/programs/tests/probabilistic/test_$(i).daphne"])
        truth = load_truth("programs/tests/probabilistic/test_$(i).truth")
        stream = get_stream(ast, num_samples)
        p_val = run_prob_test(stream, truth, num_samples)
        push!(pval_list, p_val)
        @assert p_val > max_p_value
    end
    return pval_list
end


function run_main_tests()
    sample_list = []
    n_tests = 4
    pbar = ProgressBar(1:n_tests)
    set_description(pbar, "Program")
    for i in pbar
        ast = daphne(["desugar", "-i", "../HW2/programs/$(i).daphne"])
        ret, sigma = evaluate_program(ast)
        push!(sample_list, ret)
    end
    return sample_list
end


function run_1000_samples()
    sample_dict = Dict(1=>[], 2=>[], 3=>[], 4=>[])
    for i in 1:4
        ast = daphne(["desugar", "-i", "../HW2/programs/$(i).daphne"])
        pbar = ProgressBar(1:1000)
        set_description(pbar, "Sampling from program $(i)/4")
        for _ in pbar
            ret, sigma = evaluate_program(ast)
            push!(sample_dict[i], ret)
        end
    end
    marginal_expectation_dict = compute_marginal_expectations(sample_dict)
    # plot_histograms(sample_dict, "./results/eval_based/histograms")
    plot_heatmaps(sample_dict, "./results/eval_based/heatmaps")
    return sample_dict
end


# ast_list, truth_list, ret_list = run_deterministic_tests()
# pval_list = run_probabilistic_tests()
# sample_list = run_main_tests()  # 4.daphne output: col 1 = W_0 ; col 2 = b_0; col 3:12 = W_1 ; col 13 = b_1

sample_dict = run_1000_samples()



# ast_list:
    # Any[Any["+", 5, 2]]
    # Any[Any["sqrt", 2]]
    # Any[Any["*", 3.0, 8.0]]
    # Any[Any["/", 2, 8]]
    # Any[Any["/", 2, Any["+", 3, Any["*", 3, 2.7]]]]
    # Any[Any["vector", 2, 3, 4, 5]]
    # Any[Any["get", Any["vector", 2, 3, 4, 5], 2]]
    # Any[Any["put", Any["vector", 2, 3, 4, 5], 2, 3]]
    # Any[Any["first", Any["vector", 2, 3, 4, 5]]]
    # Any[Any["last", Any["vector", 2, 3, 4, 5]]]
    # Any[Any["append", Any["vector", 2, 3, 4, 5], 3.14]]
    # Any[Any["get", Any["hash-map", 6, 5.3, 1, 3.2], 6]]
    # Any[Any["put", Any["hash-map", 6, 5.3, 1, 3.2], 6, 2]]

# truth_list:
    # 7.0
    # 1.4142136
    # 24.0
    # 0.25
    # 0.18018
    # Any[2, 3, 4, 5]
    # 4
    # Any[2, 3, 3, 5]
    # 2
    # 5
    # Any[2, 3, 4, 5, 3.14]
    # 5.3
    # Dict(6.0 => 2.0,1.0 => 3.2)

# ret_list:
    # 7
    # 1.4142135623730951
    # 24.0
    # 0.25
    # 0.18018018018018014
    # Any[2, 3, 4, 5]
    # 4
    # Any[2, 3, 3, 5]
    # 2
    # 5
    # Any[2, 3, 4, 5, 3.14]
    # 5.3
    # Dict(6.0 => 2.0,1.0 => 3.2)
