#=
Trivial baselines for the PB pretraining objectives on the pretrain test split (same units as the losses).
erecon: train mean per gene (etf) / per rank (rtf); mlm: most frequent rank per gene (etf) / gene per rank (rtf).
Usage: julia --project=x86_64 scripts/analyses/pretrain_trivial.jl lincs|tahoe
=#

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../..", arch_dir))
using JLD2, DataFrames, Statistics, StatsBase, CSV, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"))
using Preprocess: select_hvg, rank_genes, inverse_ranks, gene_medians_for
using ProcessLabels: load_pt_split, split_path_for

const DATASET = isempty(ARGS) ? "lincs" : ARGS[1]
@assert DATASET in ("lincs", "tahoe")
const REPO = abspath(joinpath(@__DIR__, "../.."))
const N_HVG = 1024
const TOP_K = 1024

function main()
    config = Dict{String,Any}("data_format" => DATASET)
    X = if DATASET == "lincs"
        Float32.(load(joinpath(REPO, "data/lincs/lincs_trt_data.jld2"))["filtered_data"].expr)
    else
        Float32.(reduce(hcat, load(joinpath(REPO, "data/tahoe/filtered_pseudobulks_alpha_10000.jld2"))["df"].expr))
    end
    s = load_pt_split(split_path_for(config))
    tr, te = s.train, s.test
    println("$DATASET: $(size(X)) | pretrain train $(length(tr)) test $(length(te))")
    rows = DataFrame()

    # ETF (HVGs)
    Xe, hvg_idx = N_HVG < size(X, 1) ? select_hvg(X, N_HVG) : (X, nothing)
    med_e = gene_medians_for(config, Xe; hvg_idx=hvg_idx)
    R_e = rank_genes(Xe, med_e)                # rank -> gene
    inv_e = inverse_ranks(R_e)                 # gene -> rank
    n_e = size(Xe, 1)
    # erecon etf: gene train mean
    μg = vec(mean(Xe[:, tr], dims=2))
    mse = mean((Xe[:, te] .- μg) .^ 2)
    push!(rows, (; dataset=DATASET, objective="erecon", arch="etf", predictor="train gene mean",
                   metric="mse", value=mse, accuracy=NaN))
    # mlm etf: most frequent rank per gene
    acc, ce = _mode_baseline(inv_e, tr, te, n_e)
    push!(rows, (; dataset=DATASET, objective="mlm", arch="etf", predictor="train modal rank per gene",
                   metric="ce", value=ce, accuracy=acc))

    # RTF (top-1024 ranks)
    med_r = gene_medians_for(config, X)
    R = rank_genes(X, med_r)
    k = min(TOP_K, size(R, 1))
    Rk = R[1:k, :]
    # erecon rtf: train mean at each rank
    Er = similar(X, Float32, k, size(X, 2))
    for j in axes(Rk, 2), r in 1:k
        Er[r, j] = X[Rk[r, j], j]
    end
    μr = vec(mean(Er[:, tr], dims=2))
    mse_r = mean((Er[:, te] .- μr) .^ 2)
    push!(rows, (; dataset=DATASET, objective="erecon", arch="rtf", predictor="train mean at rank position",
                   metric="mse", value=mse_r, accuracy=NaN))
    # mlm rtf: most frequent gene per rank
    acc_r, ce_r = _mode_baseline(Rk, tr, te, size(X, 1))
    push!(rows, (; dataset=DATASET, objective="mlm", arch="rtf", predictor="train modal gene per rank position",
                   metric="ce", value=ce_r, accuracy=acc_r))

    foreach(r -> @printf("%-6s %-7s %-4s %-36s %-3s %.5f  acc=%.4f\n", r.dataset, r.objective, r.arch, r.predictor,
                         r.metric, r.value, r.accuracy), eachrow(rows))
    out = joinpath(REPO, "results", DATASET == "lincs" ? "lincs" : "tahoe/pb", "analyses")
    mkpath(out)
    CSV.write(joinpath(out, "pretrain_trivial.csv"), rows)
    println("wrote $(joinpath(out, "pretrain_trivial.csv"))")
end

# modal train class per row -> acc, CE
function _mode_baseline(L::AbstractMatrix, tr, te, n_classes::Int)
    correct = 0; nll = 0.0; n = 0
    for p in axes(L, 1)
        c = countmap(view(L, p, tr))
        mode_c = first(sort(collect(c), by=last, rev=true))[1]
        denom = length(tr) + n_classes
        for j in te
            y = L[p, j]
            correct += (y == mode_c)
            nll -= log((get(c, y, 0) + 1) / denom)
            n += 1
        end
    end
    return correct / n, nll / n
end

main()
