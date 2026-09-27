module EvalSC

# SC pretrain test-set error accumulation

using LinearAlgebra: dot, norm

export ErrorAcc, accumulate_rank_errors!, accumulate_sq_errors!
export LreconDiag, accumulate_lrecon_diag!


# per-gene / per-rank error sums
struct ErrorAcc
    gene_sums::Vector{Float32}
    gene_counts::Vector{Int}
    rank_sums::Vector{Float32}
    rank_counts::Vector{Int}
end

ErrorAcc(n_genes::Int, n_ranks::Int) =
    ErrorAcc(zeros(Float32, n_genes), zeros(Int, n_genes), zeros(Float32, n_ranks), zeros(Int, n_ranks))

function _add!(acc::ErrorAcc, gene, rank, err)
    acc.gene_sums[gene] += err
    acc.gene_counts[gene] += 1
    acc.rank_sums[rank] += err
    acc.rank_counts[rank] += 1
end

# gene / rank of token at (pos, j)
_gene_rank(pos, v, hvg_idx) = isnothing(hvg_idx) ? (Int(v), pos) : (hvg_idx[pos], Int(v))

# mlm rank errors
function accumulate_rank_errors!(acc::ErrorAcc, errs, y_labels, n_classes, hvg_idx)
    masked_idx = 0
    @inbounds for j in axes(y_labels, 2), pos in axes(y_labels, 1)
        r = y_labels[pos, j]
        (r == -100 || r <= 0 || r > n_classes) && continue
        masked_idx += 1
        gene, rank = isnothing(hvg_idx) ? (Int(r), pos) : (hvg_idx[pos], Int(r))
        _add!(acc, gene, rank, errs[masked_idx])
    end
end

# erecon squared errors
function accumulate_sq_errors!(acc::ErrorAcc, preds, targets, mask, ids_or_ranks, hvg_idx)
    masked_idx = 0
    @inbounds for j in axes(mask, 2), pos in axes(mask, 1)
        mask[pos, j] || continue
        masked_idx += 1
        gene, rank = _gene_rank(pos, ids_or_ranks[pos, j], hvg_idx)
        _add!(acc, gene, rank, (preds[masked_idx] - targets[masked_idx])^2)
    end
end


# lrecon per-token diagnostics
struct LreconDiag
    mse::Vector{Float32}
    cossim::Vector{Float32}
    positions::Vector{Int32}
    sample_preds::Vector{Vector{Float32}}
    sample_targets::Vector{Vector{Float32}}
    sample_positions::Vector{Int32}
end

LreconDiag() = LreconDiag(Float32[], Float32[], Int32[], Vector{Float32}[], Vector{Float32}[], Int32[])

# dec / tgt: (embed_dim, n_masked)
function accumulate_lrecon_diag!(acc::ErrorAcc, diag::LreconDiag, dec, tgt, mask, ids_or_ranks, hvg_idx;
                                 save_embed::Bool)
    embed_dim = size(dec, 1)
    masked_idx = 0
    @inbounds for j in axes(mask, 2), pos in axes(mask, 1)
        mask[pos, j] || continue
        masked_idx += 1
        d = dec[:, masked_idx]
        t = tgt[:, masked_idx]
        emb_mse = sum((d .- t) .^ 2) / embed_dim
        gene, rank = _gene_rank(pos, ids_or_ranks[pos, j], hvg_idx)
        _add!(acc, gene, rank, emb_mse)
        push!(diag.mse, emb_mse)
        push!(diag.cossim, Float32(dot(d, t) / (norm(d) * norm(t) + 1f-8)))
        push!(diag.positions, Int32(rank))
        if save_embed
            push!(diag.sample_preds, d)
            push!(diag.sample_targets, t)
            push!(diag.sample_positions, Int32(rank))
        end
    end
end


end  # module EvalSC
