module Preprocess

using Statistics, Random

export select_hvg, rank_genes, ttsplit, log1p_normalize


function log1p_normalize(data_expr::Matrix)
    return Float32.(log1p.(data_expr))
end

function select_hvg(data_expr::Matrix, n_hvg::Int)
    gene_vars = vec(var(data_expr, dims=2))
    hvg_idx = sortperm(gene_vars, rev=true)[1:n_hvg]
    sort!(hvg_idx)
    return data_expr[hvg_idx, :], hvg_idx
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


end  # module Preprocess
