include("./daphne.jl")
include("./utils.jl")
include("./primitives.jl")
include("./tests.jl")
include("./analysis.jl")

using .Daphne
using .Utils
using .Primitives
using .Tests
using .Analysis

using DataStructures
using Distributions
using ResumableFunctions
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
    return sample
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


@resumable function get_stream(graph, n)
    for i in 1:convert(Int64, n)
        # all_samples, target_samples = sample_from_joint(graph)
        # @yield target_sample[1]
        sample = sample_from_joint(graph)
        @yield sample
    end
end


function run_deterministic_tests()
    graph_list = []
    truth_list = []
    ret_list = []
    pbar = ProgressBar(1:13)
    set_description(pbar, "Deterministic tests")
    for i in pbar
        if i == 13
            continue  # skip due to compiler bug
        end
        graph = daphne(["graph","-i", "../HW2/programs/tests/deterministic/test_$(i).daphne"])
        truth = load_truth("programs/tests/deterministic/test_$(i).truth")
        ret = sample_from_joint(graph)
        try
            @assert istol(ret, truth)
        catch
            AssertionError("Returned value differs from truth for test $(i)")
        end
        push!(graph_list, graph)
        push!(truth_list, truth)
        push!(ret_list, ret)
    end
    return graph_list, truth_list, ret_list
end


function run_probabilistic_tests()
    pval_list = []
    num_samples = 1e4
    max_p_value = 1e-4
    pbar = ProgressBar(1:6)
    set_description(pbar, "Probabilistic tests")
    for i in pbar
        if i == 4
            continue  # skip due to compiler bug,  "sample0" => ["sample*", Any["normal", nothing, nothing]]
        end
        graph = daphne(["graph","-i", "../HW2/programs/tests/probabilistic/test_$(i).daphne"])
        truth = load_truth("programs/tests/probabilistic/test_$(i).truth")
        # sample = sample_from_joint(graph)
        stream = get_stream(graph, num_samples)
        p_val = run_prob_test(stream, truth, num_samples)
        push!(pval_list, p_val)
        @assert p_val > max_p_value
    end
    return pval_list
end


function run_main_tests()
    sample_list = []
    pbar = ProgressBar(1:4)
    set_description(pbar, "Program")
    for i in pbar
        graph = daphne(["graph", "-i", "../HW2/programs/$(i).daphne"])
        sample = sample_from_joint(graph)
        push!(sample_list, sample)
    end
    return sample_list
end


function run_1000_samples()
    sample_dict = Dict(1=>[], 2=>[], 3=>[], 4=>[])
    for i in 1:4
        graph = daphne(["graph", "-i", "../HW2/programs/$(i).daphne"])
        pbar = ProgressBar(1:1000)
        set_description(pbar, "Sampling from program $(i)/4")
        for _ in pbar
            sample = sample_from_joint(graph)
            push!(sample_dict[i], sample)
        end
    end
    marginal_expectation_dict = compute_marginal_expectations(sample_dict)
    # plot_histograms(sample_dict, "./results/graph_based/histograms")
    plot_heatmaps(sample_dict, "./results/graph_based/heatmaps")
    return sample_dict
end


# graph_list, truth_list, ret_list = run_deterministic_tests()
# pval_list = run_probabilistic_tests()
# sample_list = run_main_tests()

sample_dict = run_1000_samples()
