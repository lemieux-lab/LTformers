module Preprocess

using Statistics, Random

export select_hvg, rank_genes, inverse_ranks, reindex_to_rank_order, ttsplit, tvsplit

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
