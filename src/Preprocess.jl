module Preprocess

using Statistics, Random, JLD2

export select_hvg, nonzero_medians, rank_genes, inverse_ranks, reindex_to_rank_order, ttsplit, tvsplit
export default_medians_path, load_gene_medians, gene_medians_for, rank_feature_k, rank_features!, rank_features
export pad_token_id, gene_set_tag
export dmso_means, plate_delta

function select_hvg(data_expr::Matrix, n_hvg::Int)
    gene_vars = vec(var(data_expr, dims=2))
    hvg_idx = sortperm(gene_vars, rev=true)[1:n_hvg]
    sort!(hvg_idx)
    return data_expr[hvg_idx, :], hvg_idx
end

# inv[gene, sample] = rank
function inverse_ranks(X_ranks::Matrix{Int32})
    inv = similar(X_ranks)
    for j in axes(X_ranks, 2)
        for i in axes(X_ranks, 1)
            inv[X_ranks[i, j], j] = Int32(i)
        end
    end
    return inv
end

# reorder expression to rank order
function reindex_to_rank_order(X_expr::Matrix{Float32}, X_ranks::Matrix{Int32})
    X_ranked = similar(X_expr)
    for j in axes(X_expr, 2)
        for i in axes(X_ranks, 1)
            X_ranked[i, j] = X_expr[X_ranks[i, j], j]
        end
    end
    return X_ranked
end

# per-gene nonzero median
# all-zero genes get 1
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

# ranking rules
# 1. log scale / train-split nonzero medians (from file)
# 2. detected first, undetected after in gene order
# 3. RTF + top-k baselines truncate to top_k, _full keeps all

# token ids: 1..n genes, n+1 MASK, n+2 PAD
pad_token_id(n_genes::Integer) = Int32(n_genes + 2)

default_medians_path(fmt::AbstractString) =
    fmt == "lincs" ? "data/lincs/gene_medians.jld2" :
    fmt == "sc"    ? "data/tahoe/sc_gene_medians.jld2" : "data/tahoe/pb_gene_medians.jld2"

# load medians file, nothing if missing
function load_gene_medians(path::AbstractString, n_genes::Union{Integer,Nothing}=nothing)
    isfile(path) || return nothing
    meds = Float32.(load(path, "medians"))
    if !isnothing(n_genes) && length(meds) != n_genes
        @warn "gene medians in $path have $(length(meds)) genes, expected $n_genes"
        return nothing
    end
    return meds
end

# medians for ranking X, falls back to nonzero_medians
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

# baseline folder tag
function gene_set_tag(modeltype::AbstractString, n_used::Integer, n_genes::Integer; kind::AbstractString, default::Integer = 1024)
    n_used >= n_genes && return n_genes <= default ? modeltype : "$(modeltype)_full"
    return n_used == default ? modeltype : "$(modeltype)_$(kind)$(n_used)"
end

# rank-baseline k, 0 = all genes
function rank_feature_k(config::Dict, n_genes::Integer)
    k = something(get(config, "rank_top_k", nothing), get(config, "top_k", 1024))
    return (k == 0 || k > n_genes) ? n_genes : k
end

# rank baseline features: (k+1-r)/k for the top-k detected genes (top gene = 1), absent/undetected = 0
# (k = all genes -> (d+1-r)/k, d = n detected)
function rank_features!(out::AbstractVector{Float32}, ids::AbstractVector{<:Integer}, n_det::Integer, top_k::Integer)
    n = length(out)
    k = min(top_k, n)
    m = min(k, n_det, length(ids))
    denom = Float32(k)
    top_r = k == n ? n_det + 1 : k + 1
    fill!(out, 0f0)
    @inbounds for r in 1:m
        out[ids[r]] = Float32(top_r - r) / denom
    end
    return out
end

function rank_features(X_ranked::AbstractMatrix{<:Integer}, n_det::AbstractVector{<:Integer}, n_genes::Integer, top_k::Integer)
    F = Matrix{Float32}(undef, n_genes, size(X_ranked, 2))
    for j in axes(X_ranked, 2)
        rank_features!(view(F, :, j), view(X_ranked, :, j), n_det[j], top_k)
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


# plate-matched DMSO deltas: delta = x - mean of the DMSO columns of the same (cell line, plate),
# in the stored log space (no z-scoring). match controls exactly ("DMSO" LINCS / tahoe PB, "DMSO_TF" SC); a substring
# match also catches "Trametinib (DMSO_TF solvate)"

# (cell line, plate) => mean of the DMSO columns of X (Float32 vectors)
function dmso_means(X::AbstractMatrix, cl::AbstractVector, plate::AbstractVector, is_dmso::AbstractVector{Bool})
    sums = Dict{Tuple{String,String},Vector{Float64}}(); counts = Dict{Tuple{String,String},Int}()
    for j in findall(is_dmso)
        k = (string(cl[j]), string(plate[j]))
        s = get!(() -> zeros(size(X, 1)), sums, k)
        s .+= view(X, :, j); counts[k] = get(counts, k, 0) + 1
    end
    return Dict(k => Float32.(s ./ counts[k]) for (k, s) in sums), counts
end

# deltas for columns idx of X; matched[i] = false when (cl, plate) of idx[i] has no DMSO (that column stays 0)
function plate_delta(X::AbstractMatrix, cl::AbstractVector, plate::AbstractVector, is_dmso::AbstractVector{Bool},
                     idx::AbstractVector{<:Integer})
    means, _ = dmso_means(X, cl, plate, is_dmso)
    D = zeros(Float32, size(X, 1), length(idx)); matched = falses(length(idx))
    for (i, j) in enumerate(idx)
        m = get(means, (string(cl[j]), string(plate[j])), nothing)
        isnothing(m) && continue
        D[:, i] .= view(X, :, j) .- m; matched[i] = true
    end
    return D, matched
end


end  # module Preprocess
