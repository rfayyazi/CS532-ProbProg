module Tests

export load_truth, is_tol, run_prob_test

using ..Primitives

using JSON
using Distributions
using HypothesisTests


function load_truth(path)
    truth = read(open(path), String)
    truth = JSON.parse(truth)
    if isa(truth, Dict)
        truth = Dict(parse(Float64, key) => val for (key, val) in truth)  #TODO: this will NOT work for nested dicts
    end
    return truth
end


function istol(a, b)
    if isa(a, Dict)
        if keys(a) == keys(b)
            for (key, val) in a
                if !(istol(val, b[key]))
                    return false
                end
            end
            return true
        else
            return false
        end
    else
        return all(abs.(a - b) .< 1e-5)
    end
end


function run_prob_test(stream, truth, num_samples)
    distrs = Dict("uniform"=>Uniform, "normal"=>Normal, "beta"=>Beta, "exponential"=>(x,y)->Exponential(1/y), "normalmix"=>NormalMix)
    samples = Array{Float64}(undef, convert(Int64, num_samples))
    i = 1
    for next in stream
        samples[i] = next
        i += 1
    end
    truth_dist = distrs[truth[1]](truth[2:end]...)
    if isa(truth_dist, NormalMix)
        test_results = NM_ExactOneSampleKSTest(samples, truth_dist)  # overloading ExactOneSampleKSTest caused error
    else
        test_results = ExactOneSampleKSTest(samples, truth_dist)
    end
    return pvalue(test_results)
end


# implementation from scipy.stats.ks_1samp and HypothesisTests.jl/kolmogorov_smirnov.jl/ksstats,
# but with a bit more precise notation (ie. collect and element-wise ops)
function NM_ExactOneSampleKSTest(x::Array, nm::NormalMix)
    n = length(x)
    cdfs = nm_cdf(nm, sort(x))
    δp = maximum((collect(1.0:1:n) ./ n) .- cdfs)
    δn = -minimum((collect(0.0:1:n-1) ./ n) .- cdfs)
    δ = max(δp, δn)
    return ExactOneSampleKSTest(n, δ, δp, δn)
end

end
