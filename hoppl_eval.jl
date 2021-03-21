include("./daphne.jl")
include("./hoppl_prims.jl")
include("./tests.jl")

using .Daphne
using .Primitives
using .Tests

using Distributions
using FunctionalCollections
using ProgressBars
using ResumableFunctions
using Plots


struct Env
    dict
    outer
end


struct Procedure
    args
    body
    env
end


function Env(args, vals, outer)
    if length(args) == 0
        return Env(Dict(), outer)
    else
        return Env(Dict(args[i] => vals[i] for i in 1:length(args)), outer)
    end

end


standard_env() = Env(get_primitives(), nothing)


deepcopy_env(env::Env) = Env(env.dict, env.outer)


search_stack(env::Env, var::String) =  var in keys(env.dict) ? env : search_stack(env.outer, var)


function call_procedure(p, vals)
    if isa(p, Procedure)
        return eval_expr(p.body, Env(p.args, vals, p.env))
    else
        return p(vals...)
    end
end


function eval_expr(expr, env)
    case = match_case(expr)

    if case == "constant"
        return expr

    elseif case == "variable"
        try
            return search_stack(env, expr).dict[expr]
        catch
            return expr
        end

    elseif case == "if"
        (predic, conseq, altern) = expr[2:end]
        eval_expr(predic, env) ? eval_expr(conseq, env) : eval_expr(altern, env)

    elseif case == "fn"
        (args, body) = expr[2:end]
        length(args) < 2 ? args = [] : args = args[2:end]
        return Procedure(args, body, Env(Dict(), deepcopy_env(env)))

    elseif case == "sample"
        d = eval_expr(expr[3], env)
        s = rand(d)
        if isa(d, DiscreteNonParametric)
            s -= 1
        end
        return s

    elseif case == "observe"
        return eval_expr(expr[end], env)

    elseif case == "procedure"
        proc = eval_expr(expr[1], env)
        vals = [eval_expr(e, env) for e in expr[3:end]]
        return call_procedure(proc, vals)
    end
end


function match_case(expr)
    if isa(expr, String) && expr != "fn"
        return "variable"
    elseif isa(expr, Number) || isa(expr[1], Number)
        return "constant"
    elseif expr[1] == "if"
        return "if"
    elseif expr[1] == "fn"
        return "fn"
    elseif expr[1] == "sample"
        return "sample"
    elseif expr[1] == "observe"
        return "observe"
    else
        return "procedure"
    end
end


function run_det_tests()
    expr_list = []
    truth_list = []
    ret_list = []
    pbar = ProgressBar(1:13)
    set_description(pbar, "Deterministic tests")
    for i in pbar
        file_path = "../CS532-ProbProg-Assignment/hopple_programs/tests/deterministic"
        expr = daphne(["desugar-hoppl", "-i", joinpath(file_path, "test_$(i).daphne")])
        truth = load_truth(joinpath(file_path, "test_$(i).truth"))
        ret = call_procedure(eval_expr(expr, standard_env()), [""])
        try
            @assert istol(ret, truth)
        catch
            AssertionError("Returned value differs from truth for test $(i)")
        end
        push!(expr_list, expr)
        push!(truth_list, truth)
        push!(ret_list, ret)
    end
    return expr_list, truth_list, ret_list
end


function run_hoppl_det_tests()
    expr_list = []
    truth_list = []
    ret_list = []
    pbar = ProgressBar(1:11)
    set_description(pbar, "HOPPL-Deterministic tests")
    for i in pbar
        file_path = "../CS532-ProbProg-Assignment/hopple_programs/tests/hoppl-deterministic"
        expr = daphne(["desugar-hoppl", "-i", joinpath(file_path, "test_$(i).daphne")])
        truth = load_truth(joinpath(file_path, "test_$(i).truth"))
        ret = call_procedure(eval_expr(expr, standard_env()), [""])
        try
            @assert istol(ret, truth)
        catch
            AssertionError("Returned value differs from truth for test $(i)")
        end
        push!(expr_list, expr)
        push!(truth_list, truth)
        push!(ret_list, ret)
    end
    return expr_list, truth_list, ret_list
end


@resumable function get_stream(expr, n)
    for i in 1:n
        sample = call_procedure(eval_expr(expr, standard_env()), [""])
        @yield sample
    end
end


function run_prob_tests()
    pval_list = []
    num_samples = 1e4
    max_p_value = 1e-4
    pbar = ProgressBar(1:6)
    set_description(pbar, "Probabilistic tests")
    for i in pbar
        file_path = "../CS532-ProbProg-Assignment/hopple_programs/tests/probabilistic"
        expr = daphne(["desugar-hoppl", "-i", joinpath(file_path, "test_$(i).daphne")])
        truth = load_truth(joinpath(file_path, "test_$(i).truth"))
        stream = get_stream(expr, convert(Int64, num_samples))
        p_val = run_prob_test(stream, truth, num_samples)
        push!(pval_list, p_val)
        @assert p_val > max_p_value
    end
    return pval_list
end


function analyse(sample_dict, n_samples)
    expectations = Dict()
    variances = Dict()
    for k in keys(sample_dict)
        expectations[k] = foldl(+, sample_dict[k]) ./ n_samples
        if isa(sample_dict[k][1], Number)
            exp_x_sqr = foldl(+, [s ^ 2 for s in sample_dict[k]]) ./ n_samples
            variances[k] = exp_x_sqr - (expectations[k] ^ 2)
        else
            exp_x_sqr = foldl(+, [[s...] .^ 2 for s in sample_dict[k]]) ./ n_samples
            variances[k] = exp_x_sqr - ([expectations[k]...] .^ 2)
        end
    end
    return expectations, variances
end


function run_main_programs(n_samples::Int, to_plot::Bool)
    sample_dict = Dict()
    for i in 1:3
        sample_dict[i] = []
        expr = daphne(["desugar-hoppl", "-i", "../CS532-ProbProg-Assignment/hopple_programs/$(i).daphne"])
        pbar = ProgressBar(1:n_samples)
        set_description(pbar, "Sampling from program $(i)/3")
        for _ in pbar
            ret = call_procedure(eval_expr(expr, standard_env()), [""])
            push!(sample_dict[i], ret)
        end
    end

    E, V = analyse(sample_dict, n_samples)

    if to_plot
        for k in keys(sample_dict)
            if isa(sample_dict[k][1], PersistentVector)
                for i in eachindex(sample_dict[k][1])
                    samples = []
                    for sample in sample_dict[k]
                        push!(samples, sample[i])
                    end
                    histogram(samples)
                    title!("Sample histogram for program $(k) random variable $(i)")
                    savefig("./results/program_$(k)_rv_$(i).png")
                end
            else
                samples = [sample for sample in sample_dict[k]]
                histogram(samples)
                title!("Sample histogram for program $(k) random variable")
                savefig("./results/program_$(k)_rv.png")
            end
        end
    end

    return sample_dict, E, V
end



d_expr_list, d_truth_list, d_ret_list = run_det_tests()
hd_expr_list, hd_truth_list, hd_ret_list = run_hoppl_det_tests()
pvals = run_prob_tests()
main_samples, E, V = run_main_programs(1000, false)
