module Preprocess

# using Statistics, Random
using Statistics, Random, JLD2

# export select_hvg, rank_genes, inverse_ranks, reindex_to_rank_order, ttsplit, tvsplit
# export select_hvg, nonzero_medians, rank_genes, inverse_ranks, reindex_to_rank_order, ttsplit, tvsplit
export select_hvg, nonzero_medians, rank_genes, inverse_ranks, reindex_to_rank_order, ttsplit, tvsplit
export default_medians_path, load_gene_medians, gene_medians_for, rank_feature_k, rank_features!, rank_features
export pad_token_id, gene_set_tag

function select_hvg(data_expr::Matrix, n_hvg::Int)
    gene_vars = vec(var(data_expr, dims=2))
    hvg_idx = sortperm(gene_vars, rev=true)[1:n_hvg]
    sort!(hvg_idx)
    return data_expr[hvg_idx, :], hvg_idx
end

# inv[gene_id, sample] = rank of that gene (1 = highest expression)
function inverse_ranks(X_ranks::Matrix{Int32})
    inv = similar(X_ranks)
    for j in axes(X_ranks, 2)
        for i in axes(X_ranks, 1)
            inv[X_ranks[i, j], j] = Int32(i)
        end
    end
    return inv
end

# reorder expression into rank order so position i = expression of gene at rank i
# matches what SC pipeline does via sortperm in build_batch_etf
function reindex_to_rank_order(X_expr::Matrix{Float32}, X_ranks::Matrix{Int32})
    X_ranked = similar(X_expr)
    for j in axes(X_expr, 2)
        for i in axes(X_ranks, 1)
            X_ranked[i, j] = X_expr[X_ranks[i, j], j]
        end
    end
    return X_ranked
end

# per-gene median over samples where the gene is nonzero
# replaces median(X) .+ 1f-10, which let zero-median genes (x / 1f-10) flood the top ranks
# genes that are zero in every sample get 1f0 (they rank at the bottom anyway)
function nonzero_medians(X::AbstractMatrix)
    meds = Vector{Float32}(undef, size(X, 1))
    buf = Vector{Float32}(undef, size(X, 2))
    for i in axes(X, 1)
        n = 0
        for j in axes(X, 2)
            v = X[i, j]
            if v > 0
                n += 1
                buf[n] = v
            end
        end
        meds[i] = n == 0 ? 1f0 : Float32(median!(view(buf, 1:n)))
    end
    return meds
end

# --- geneformer-style ranking: shared rules for every rank-based model (RTF, rlog, rmlp; PB + SC) ---
# 1. rank on the stored log scale (PB log1p CP10k, SC log1p CP10k, LINCS log2), divided by per-gene nonzero medians
# 2. medians are computed once on the pretraining train split and loaded from a file (never recomputed on the
#    samples being ranked, so test samples don't shape them and pretrain/finetune tokens match)
# 3. detected genes (x > 0) first by x / median; undetected genes after them in gene-index order
#    (rank_genes gets this for free: undetected ratio = 0 and sortperm is stable)
# 4. RTF + top-k baselines truncate to top_k; only no_pretrain `_full` baselines keep every gene

# Geneformer-style padding for rank tokens (SC cells with < top_k detected genes): token ids 1..n_genes are genes,
# n_genes + 1 = MASK (last embedding row), n_genes + 2 = PAD. PAD is never looked up by the embedding: Models.encode_rank
# swaps it to MASK for the lookup, excludes it as an attention key and from pooling; masking never selects it
pad_token_id(n_genes::Integer) = Int32(n_genes + 2)

default_medians_path(fmt::AbstractString) =
    fmt == "lincs" ? "data/lincs/gene_medians.jld2" :
    fmt == "sc"    ? "data/tahoe/sc_gene_medians.jld2" : "data/tahoe/pb_gene_medians.jld2"

# medians vector from a compute_medians / compute_hvg file; nothing if the file is missing or doesn't match n_genes
# n_genes = nothing skips the length check (caller indexes with hvg_idx)
function load_gene_medians(path::AbstractString, n_genes::Union{Integer,Nothing}=nothing)
    isfile(path) || return nothing
    meds = Float32.(load(path, "medians"))
    if !isnothing(n_genes) && length(meds) != n_genes
        @warn "gene medians in $path have $(length(meds)) genes, expected $n_genes"
        return nothing
    end
    return meds
end

# medians for ranking X (genes × samples): the saved train-split file (config "medians_path", default per data_format),
# indexed by hvg_idx when X is an HVG subset of the full gene set. falls back to nonzero_medians(X) with a warning
# if no usable file exists
function gene_medians_for(config::Dict, X::AbstractMatrix; hvg_idx=nothing)
    fmt = get(config, "data_format", "tahoe")
    path = get(config, "medians_path", "")
    path = path == "" ? default_medians_path(fmt) : path
    meds = load_gene_medians(path, isnothing(hvg_idx) ? size(X, 1) : nothing)
    if !isnothing(meds) && !isnothing(hvg_idx)
        meds = maximum(hvg_idx) <= length(meds) ? meds[hvg_idx] : nothing
    end
    if isnothing(meds)
        @warn "no usable gene medians file at $path (run scripts/pretrain/pb/compute_medians.jl); computing nonzero medians on the $(size(X, 2)) samples being ranked"
        return nonzero_medians(X)
    end
    println("gene medians: loaded $path")
    return meds
end

# results-folder tag for baselines: default gene-set size -> modeltype; all genes -> <modeltype>_full (only when the
# dataset has more genes than the default, so LINCS' 978 genes never get _full); any other size -> _<kind><n>
# kind = "topk" (rlog/rmlp top-k ranks) or "hvg" (elog/emlp HVGs)
function gene_set_tag(modeltype::AbstractString, n_used::Integer, n_genes::Integer; kind::AbstractString, default::Integer = 1024)
    n_used >= n_genes && return n_genes <= default ? modeltype : "$(modeltype)_full"
    return n_used == default ? modeltype : "$(modeltype)_$(kind)$(n_used)"
end

# k for the rank-baseline encoding: --rank_top_k if given (0 = all genes), else config top_k; capped at n_genes
function rank_feature_k(config::Dict, n_genes::Integer)
    k = something(get(config, "rank_top_k", nothing), get(config, "top_k", 1024))
    return (k == 0 || k > n_genes) ? n_genes : k
end

# rank baseline (rlog/rmlp) features: 0-1 scale, absent genes tied at position k+1 (Fagin 2003), flipped so absent = 0
#   top-k (k < n): gene at rank r <= min(k, n_det) -> (k+1-r)/k; every other gene -> 0
#   full  (k = n): detected gene at rank r -> (n_det+1-r)/n; undetected genes -> 0
# ids: gene ids in rank order (detected first); n_det: number of detected genes in that sample
# encoding = :rev (default) -> (k+1-r)/k, absent 0   (top gene 1, sparse: absent genes are 0)
# encoding = :rk            -> r/k, absent (k+1)/k   (log.txt rank / n convention; ~18k constant ~1 inputs)
# :rk barely trains: 1000-step Tahoe lvl2 rlog acc 0.007 vs 0.207 for :rev (2026-09-25), same information
# candidates under discussion (2026-09-26, not default): :logrank, :binary, :recip (all absent 0)
function rank_features!(out::AbstractVector{Float32}, ids::AbstractVector{<:Integer}, n_det::Integer, top_k::Integer;
                        encoding::Symbol = :rev)
    n = length(out)
    k = min(top_k, n)
    m = min(k, n_det, length(ids))
    denom = Float32(k)
    top_r = k == n ? n_det + 1 : k + 1   # position "absent" genes are tied at
    if encoding == :rev
        fill!(out, 0f0)
        @inbounds for r in 1:m
            out[ids[r]] = Float32(top_r - r) / denom
        end
    elseif encoding == :logrank   # 1 - log(r)/log(top_r): top gene 1, decays like log rank (Cell2Sentence), absent 0
        fill!(out, 0f0)
        lt = log(Float32(top_r))
        @inbounds for r in 1:m
            out[ids[r]] = 1f0 - log(Float32(r)) / lt
        end
    elseif encoding == :binary    # 1 for the top-k (or detected) genes, absent 0: which genes, no order
        fill!(out, 0f0)
        @inbounds for r in 1:m
            out[ids[r]] = 1f0
        end
    elseif encoding == :recip     # 1/r (reciprocal rank), absent 0: weight concentrated on the first ranks
        fill!(out, 0f0)
        @inbounds for r in 1:m
            out[ids[r]] = 1f0 / Float32(r)
        end
    else
        fill!(out, Float32(top_r) / denom)
        @inbounds for r in 1:m
            out[ids[r]] = Float32(r) / denom
        end
    end
    return out
end

function rank_features(X_ranked::AbstractMatrix{<:Integer}, n_det::AbstractVector{<:Integer}, n_genes::Integer, top_k::Integer;
                       encoding::Symbol = :rev)
    F = Matrix{Float32}(undef, n_genes, size(X_ranked, 2))
    for j in axes(X_ranked, 2)
        rank_features!(view(F, :, j), view(X_ranked, :, j), n_det[j], top_k; encoding=encoding)
    end
    return F
end

function rank_genes(expr::Matrix, medians::Vector)
    n, m = size(expr)
    data_ranked = Matrix{Int32}(undef, size(expr))
    normalized_col = Vector{Float32}(undef, n)
    sorted_ind_col = Vector{Int32}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        @. normalized_col = unsorted_expr_col / medians
        sortperm!(sorted_ind_col, normalized_col, rev=true)
        data_ranked[:, j] .= sorted_ind_col
    end
    return data_ranked
end

function ttsplit(X::Matrix, ratio::AbstractFloat; y=nothing)
    idx = shuffle(1:size(X, 2))
    n_test = floor(Int, length(idx) * ratio)
    s = length(idx) - n_test
    train_idx = idx[1:s]
    test_idx = idx[s+1:end]
    if isnothing(y)
        return X[:, train_idx], X[:, test_idx], train_idx, test_idx
    end
    return X[:, train_idx], y[:, train_idx], X[:, test_idx], y[:, test_idx], train_idx, test_idx
end


function tvsplit(X::Matrix, val_ratio::AbstractFloat, test_ratio::AbstractFloat; y=nothing)
    idx = shuffle(1:size(X, 2))
    n_test = floor(Int, length(idx) * test_ratio)
    n_val  = floor(Int, length(idx) * val_ratio)
    s_test = length(idx) - n_test
    s_val  = s_test - n_val
    train_idx = idx[1:s_val]
    val_idx   = idx[s_val+1:s_test]
    test_idx  = idx[s_test+1:end]
    if isnothing(y)
        return X[:, train_idx], X[:, val_idx], X[:, test_idx], train_idx, val_idx, test_idx
    end
    return X[:, train_idx], y[:, train_idx], X[:, val_idx], y[:, val_idx], X[:, test_idx], y[:, test_idx], train_idx, val_idx, test_idx
end


end  # module Preprocess
