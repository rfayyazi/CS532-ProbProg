module Analysis

export plot_histograms, plot_heatmaps
export compute_marginal_expectations, compute_expectation, compute_variance

using Plots
using StatsBase


function compute_marginal_expectations(sample_dict)
    marginal_expectation_dict = Dict(k => foldl(+, sample_dict[k]) ./ 1000 for k in keys(sample_dict))
    return marginal_expectation_dict
end


function plot_histograms(sample_dict, save_path)
    for k in keys(sample_dict)
        if isa(sample_dict[k][1], Array)
            for i in eachindex(sample_dict[k][1])
                samples = []
                for sample in sample_dict[k]
                    push!(samples, sample[i])
                end
                histogram(samples)
                title!("Sample histogram for program $(k) random variable $(i)")
                savefig(joinpath(save_path, "program_$(k)_rv_$(i).png"))
            end
        else
            samples = [sample for sample in sample_dict[k]]
            histogram(samples)
            title!("Sample histogram for program $(k) random variable")
            savefig(joinpath(save_path, "program_$(k)_rv.png"))
        end
    end
end


function plot_heatmaps(sample_dict, save_path)
    BNN_samples = Array{Float64, 3}(undef, 1000, 10, 13)
    for i in 1:1000
        BNN_samples[i, :, :] = sample_dict[4][i]
    end
    for i in 1:13
        counts = Array{Int64, 2}(undef, 10, length((-4.0:0.5:4))-1)  # there are length((-4.0:0.5:4))-1 bins
        for j in 1:10
            counts[j, :] = fit(Histogram, BNN_samples[:, j, i], -4.0:0.5:4).weights
        end
        heatmap((-4.0:0.5:3.5), (1:1:10), counts)
        xlabel!("sample value")
        ylabel!("random variable")
        title!("Histograms of random variable samples, column $(i)/13")
        savefig(joinpath(save_path, "heatmap_program_4_column_$(i).png"))
    end
    HMM_samples = Array{Float64, 2}(undef, 1000, 17)
    for i in 1:1000
        HMM_samples[i, :] = sample_dict[3][i]
    end
    counts = Array{Int64, 2}(undef, 17, 3)  # there are length((-4.0:0.5:4))-1 bins
    for i in 1:17
        counts[i, :] = fit(Histogram, HMM_samples[:, i], 0:1:3).weights
    end
    heatmap((0:1:2), (1:1:17), counts)
    xlabel!("sample value")
    ylabel!("random variable")
    title!("Histograms of random variable samples")
    savefig(joinpath(save_path, "heatmap_program_3.png"))
end


function compute_expectation(stream::Array)
    samples = first.(stream)
    logWs = last.(stream)
    W = exp.(logWs)
    normalized_Ws = W ./ sum(W)
    expectation = foldl(+, [samples[i] * normalized_Ws[i] for i in 1:length(samples)])
    return expectation
end


function compute_variance(stream::Array)
    # Var[X] = E[X^2] - E[X]^2
    samples = first.(stream)
    logWs = last.(stream)
    W = exp.(logWs)
    normalized_Ws = W ./ sum(W)
    T1 = foldl(+, [(samples[i] .^ 2 * normalized_Ws[i]) for i in 1:length(samples)])
    T2 = foldl(+, [samples[i] * normalized_Ws[i] for i in 1:length(samples)]) .^ 2
    return T1 - T2
end


end
