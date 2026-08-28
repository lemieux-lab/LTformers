module ProcessSC

using Flux, CUDA, Statistics, Random, StatsBase, SparseArrays, Base.Threads

export log_normalize_col!, process_cell_topk_flat
export build_batch_rtf, build_batch_etf, build_batch_etf_hvg, batches_from_shard
export batches_from_shard_data
export sc_mask_input!, sc_mask_input_exp!, sc_mask_input_exp_rank!, sc_mask_input_erecon!, sc_masked_loss
export cell_to_dense_flat!, process_cell_topk_flat
export gpu_rank_errors


# cell processing

function log_normalize_col!(vec::AbstractVector{Float32}; α::Float32 = 10000f0)
    total = sum(vec)
    if total > 0f0
        sf = α / total
        @inbounds for i in eachindex(vec)
            if vec[i] > 0f0
                vec[i] = log1p(sf * vec[i])
            end
        end
    end
end

function cell_to_dense_flat!(dense::Vector{Float32},
                             genes_flat::Vector{Int64}, offsets::Vector{Int64},
                             expr_flat::Vector{Float32}, cell_idx::Int,
                             token_to_idx::Dict{Int,Int})
    fill!(dense, 0f0)
    s = offsets[cell_idx] + 1
    e = offsets[cell_idx + 1]
    # skip sentinel (negative expression = metadata flag)
    if s <= e && expr_flat[s] < 0
        s += 1
    end
    @inbounds for i in s:e
        ci = get(token_to_idx, genes_flat[i], 0)
        ci == 0 && continue
        dense[ci] = expr_flat[i]
    end
    log_normalize_col!(dense)
end

function process_cell_topk_flat(dense::Vector{Float32},
                                genes_flat::Vector{Int64}, offsets::Vector{Int64},
                                expr_flat::Vector{Float32}, cell_idx::Int,
                                token_to_idx::Dict{Int,Int}, n_coding::Int, top_k::Int)
    cell_to_dense_flat!(dense, genes_flat, offsets, expr_flat, cell_idx, token_to_idx)

    @inbounds for i in 1:n_coding
        dense[i] += randn(Float32) * 1f-10
    end
    perm = partialsortperm(dense, 1:top_k, rev=true)  # O(n + k log k) vs O(n log n)
    top_gene_ids = Int32.(perm)
    top_expr_vals = dense[perm]
    return top_gene_ids, top_expr_vals
end


# batch building

function build_batch_rtf(genes_flat::Vector{Int64}, offsets::Vector{Int64},
                         expr_flat::Vector{Float32}, cell_indices::AbstractVector{Int},
                         token_to_idx::Dict{Int,Int}, n_coding::Int, top_k::Int)
    bs = length(cell_indices)
    batch = Matrix{Int32}(undef, top_k, bs)
    nt = nthreads()
    if nt > 1 && bs >= 4
        # dense_bufs = [Vector{Float32}(undef, n_coding) for _ in 1:nt]
        @threads for j in 1:bs
            # per-iteration dense buffer (76KB each, avoids threadid() issues in Channel tasks)
            dense = Vector{Float32}(undef, n_coding)
            gene_ids, _ = process_cell_topk_flat(dense, genes_flat, offsets, expr_flat,
                                                 cell_indices[j], token_to_idx, n_coding, top_k)
            batch[:, j] = gene_ids
        end
    else
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            gene_ids, _ = process_cell_topk_flat(dense, genes_flat, offsets, expr_flat,
                                                 ci, token_to_idx, n_coding, top_k)
            batch[:, j] = gene_ids
        end
    end
    return batch
end

function build_batch_etf(genes_flat::Vector{Int64}, offsets::Vector{Int64},
                         expr_flat::Vector{Float32}, cell_indices::AbstractVector{Int},
                         token_to_idx::Dict{Int,Int}, n_coding::Int, top_k::Int)
    bs = length(cell_indices)
    batch_ids = Matrix{Int32}(undef, top_k, bs)
    batch_expr = Matrix{Float32}(undef, top_k, bs)
    nt = nthreads()
    if nt > 1 && bs >= 4
        # dense_bufs = [Vector{Float32}(undef, n_coding) for _ in 1:nt]
        @threads for j in 1:bs
            dense = Vector{Float32}(undef, n_coding)
            gene_ids, expr_vals = process_cell_topk_flat(dense, genes_flat, offsets, expr_flat,
                                                         cell_indices[j], token_to_idx, n_coding, top_k)
            batch_ids[:, j] = gene_ids
            batch_expr[:, j] = expr_vals
        end
    else
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            gene_ids, expr_vals = process_cell_topk_flat(dense, genes_flat, offsets, expr_flat,
                                                         ci, token_to_idx, n_coding, top_k)
            batch_ids[:, j] = gene_ids
            batch_expr[:, j] = expr_vals
        end
    end
    return batch_ids, batch_expr
end

# ETF-HVG batch building: extract expression at fixed HVG gene positions (gene-position paradigm)
# Returns (batch_expr, batch_ranks) where position i = hvg gene i across all cells
function build_batch_etf_hvg(genes_flat::Vector{Int64}, offsets::Vector{Int64},
                              expr_flat::Vector{Float32}, cell_indices::AbstractVector{Int},
                              token_to_idx::Dict{Int,Int}, n_coding::Int,
                              hvg_idx::Vector{Int})
    n_hvg = length(hvg_idx)
    bs = length(cell_indices)
    batch_expr = Matrix{Float32}(undef, n_hvg, bs)
    batch_ranks = Matrix{Int32}(undef, n_hvg, bs)
    nt = nthreads()
    if nt > 1 && bs >= 4
        @threads for j in 1:bs
            dense = Vector{Float32}(undef, n_coding)
            cell_to_dense_flat!(dense, genes_flat, offsets, expr_flat,
                                cell_indices[j], token_to_idx)
            hvg_expr = dense[hvg_idx]
            batch_expr[:, j] = hvg_expr
            # compute ranks within HVG set (1 = highest expression)
            perm = sortperm(hvg_expr, rev=true)
            for (rank, idx) in enumerate(perm)
                batch_ranks[idx, j] = Int32(rank)
            end
        end
    else
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            cell_to_dense_flat!(dense, genes_flat, offsets, expr_flat,
                                ci, token_to_idx)
            hvg_expr = dense[hvg_idx]
            batch_expr[:, j] = hvg_expr
            # compute ranks within HVG set (1 = highest expression)
            perm = sortperm(hvg_expr, rev=true)
            for (rank, idx) in enumerate(perm)
                batch_ranks[idx, j] = Int32(rank)
            end
        end
    end
    return batch_expr, batch_ranks
end

# yields one batch at a time via Channel so first step starts immediately
function batches_from_shard(path::String, coding_tokens::Vector{Int}, n_coding::Int,
                            top_k::Int, batch_size::Int;
                            modeltype::String = "rtf",
                            token_to_idx::Dict{Int,Int} = error("token_to_idx required"),
                            load_shard_fn = error("load_shard_fn required"),
                            hvg_idx::Union{Vector{Int}, Nothing} = nothing)
    println("  loading shard: $(basename(path))")
    shard = load_shard_fn(path)
    println("  loaded $(shard.n_cells) cells, building batches...")
    cell_order = shuffle(1:shard.n_cells)

    return Channel{Any}(1) do ch
        for start_idx in 1:batch_size:shard.n_cells
            end_idx = min(start_idx + batch_size - 1, shard.n_cells)
            ci = cell_order[start_idx:end_idx]
            if modeltype == "etf" && !isnothing(hvg_idx)
                # ETF-HVG: gene-position paradigm
                expr, ranks = build_batch_etf_hvg(shard.genes_flat, shard.offsets, shard.expr_flat,
                                                   ci, token_to_idx, n_coding, hvg_idx)
                put!(ch, (expr, ranks))
            elseif modeltype == "etf"
                # ETF legacy: rank-ordered top_k (kept for backward compat)
                ids, vals = build_batch_etf(shard.genes_flat, shard.offsets, shard.expr_flat,
                                            ci, token_to_idx, n_coding, top_k)
                put!(ch, (ids, vals))
            else
                batch = build_batch_rtf(shard.genes_flat, shard.offsets, shard.expr_flat,
                                        ci, token_to_idx, n_coding, top_k)
                put!(ch, batch)
            end
        end
    end
end


# SC masking

function sc_mask_input!(X_masked::AbstractMatrix{Int32}, mask_labels::AbstractMatrix{Int32},
                        X::AbstractMatrix{Int32}, mask_ratio::Float64,
                        mask_val::Int, mask_id::Int32)
    copyto!(X_masked, X)
    fill!(mask_labels, Int32(mask_val))
    n_rows, n_samples = size(X)
    num_masked = ceil(Int, n_rows * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(1:n_rows, num_masked, replace=false)
        for pos in mask_pos
            mask_labels[pos, j] = X[pos, j]
            X_masked[pos, j] = mask_id
        end
    end
    return X_masked, mask_labels
end

function sc_mask_input_exp!(X_masked::AbstractMatrix{Float32}, mask_labels::AbstractMatrix{Int32},
                            X_expr::AbstractMatrix{Float32}, X_ids::AbstractMatrix{Int32},
                            mask_ratio::Float64, mask_val::Int)
    copyto!(X_masked, X_expr)
    fill!(mask_labels, Int32(mask_val))
    n_rows, n_samples = size(X_expr)
    num_masked = ceil(Int, n_rows * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(1:n_rows, num_masked, replace=false)
        for pos in mask_pos
            mask_labels[pos, j] = X_ids[pos, j]
            X_masked[pos, j] = -1f0
        end
    end
    return X_masked, mask_labels
end


# ETF-HVG masking: mask expression values, labels = ranks at masked positions
function sc_mask_input_exp_rank!(X_masked::AbstractMatrix{Float32}, mask_labels::AbstractMatrix{Int32},
                                 X_expr::AbstractMatrix{Float32}, X_ranks::AbstractMatrix{Int32},
                                 mask_ratio::Float64, mask_val::Int)
    copyto!(X_masked, X_expr)
    fill!(mask_labels, Int32(mask_val))
    n_rows, n_samples = size(X_expr)
    num_masked = ceil(Int, n_rows * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(1:n_rows, num_masked, replace=false)
        for pos in mask_pos
            mask_labels[pos, j] = X_ranks[pos, j]  # rank as label (not gene ID)
            X_masked[pos, j] = -1f0
        end
    end
    return X_masked, mask_labels
end

function sc_mask_input_erecon!(X_masked::AbstractMatrix{Int32}, expr_labels::AbstractMatrix{Float32},
                              X_ids::AbstractMatrix{Int32}, X_expr::AbstractMatrix{Float32},
                              mask_ratio::Float64, mask_val::Float32, mask_id::Int32)
    copyto!(X_masked, X_ids)
    fill!(expr_labels, mask_val)
    n_rows, n_samples = size(X_ids)
    num_masked = ceil(Int, n_rows * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(1:n_rows, num_masked, replace=false)
        for pos in mask_pos
            expr_labels[pos, j] = X_expr[pos, j]
            X_masked[pos, j] = mask_id
        end
    end
    return X_masked, expr_labels
end


# SC loss

function _sc_classify_masked(classifier, transformed, y_gpu, n_classes)
    ed = size(transformed, 1)
    transformed_2d = reshape(transformed, ed, :)
    y_flat = vec(y_gpu)
    mask = (y_flat .!= -100) .& (y_flat .<= n_classes) .& (y_flat .> 0)
    if !any(mask)
        return 0f0, nothing, nothing
    end
    masked_emb = transformed_2d[:, mask]
    logits_masked = classifier(masked_emb)
    y_masked = y_flat[mask]
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes)
    return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
end

function sc_masked_loss(model, x_gpu, y_gpu, n_classes)
    transformed = encode(model, x_gpu)
    _sc_classify_masked(model.classifier, transformed, y_gpu, n_classes)
end

function sc_masked_loss(model, x_gpu::CuArray{Float32}, y_gpu, n_classes)
    x3d = reshape(x_gpu, 1, size(x_gpu)...)
    projected = model.proj(x3d)
    mask_3d = reshape(x_gpu .== -1f0, 1, size(x_gpu)...)
    projected = projected .* (1f0 .- mask_3d) .+ model.mask_emb .* mask_3d
    gene_ids = cu(Int32.(1:size(x_gpu, 1)))
    combined = projected .+ model.pos_emb(gene_ids)
    dropped = model.emb_dropout(combined)
    transformed = model.transformer(dropped)
    _sc_classify_masked(model.classifier, transformed, y_gpu, n_classes)
end


## async shard pre-loading: build batches from already-loaded shard data
function batches_from_shard_data(shard, coding_tokens::Vector{Int}, n_coding::Int,
                                 top_k::Int, batch_size::Int;
                                 modeltype::String = "rtf",
                                 token_to_idx::Dict{Int,Int} = error("token_to_idx required"),
                                 hvg_idx::Union{Vector{Int}, Nothing} = nothing)
    cell_order = shuffle(1:shard.n_cells)
    return Channel{Any}(1) do ch
        for start_idx in 1:batch_size:shard.n_cells
            end_idx = min(start_idx + batch_size - 1, shard.n_cells)
            ci = cell_order[start_idx:end_idx]
            if modeltype == "etf" && !isnothing(hvg_idx)
                expr, ranks = build_batch_etf_hvg(shard.genes_flat, shard.offsets, shard.expr_flat,
                                                   ci, token_to_idx, n_coding, hvg_idx)
                put!(ch, (expr, ranks))
            elseif modeltype == "etf"
                ids, vals = build_batch_etf(shard.genes_flat, shard.offsets, shard.expr_flat,
                                            ci, token_to_idx, n_coding, top_k)
                put!(ch, (ids, vals))
            else
                batch = build_batch_rtf(shard.genes_flat, shard.offsets, shard.expr_flat,
                                        ci, token_to_idx, n_coding, top_k)
                put!(ch, batch)
            end
        end
    end
end


## GPU rank errors — replaces O(n_classes) CPU loop per masked token
## processes in chunks to limit GPU memory (19020 × chunk_size × 4 bytes per temp)
function gpu_rank_errors(logits_masked::CuArray{Float32, 2}, y_targets::CuArray;
                         chunk_size::Int = 2000)
    n_classes, n_tokens = size(logits_masked)
    errors = Vector{Int}(undef, n_tokens)
    nc = Int32(n_classes)
    for start in 1:chunk_size:n_tokens
        stop = min(start + chunk_size - 1, n_tokens)
        n_chunk = stop - start + 1
        chunk_logits = logits_masked[:, start:stop]           # (n_classes, n_chunk) on GPU
        chunk_targets = y_targets[start:stop]                  # (n_chunk,) on GPU
        # gather logit at the true class for each masked token via linear indexing
        offsets_gpu = cu(collect(Int32(0):Int32(n_chunk - 1))) .* nc
        lin_idx = chunk_targets .+ offsets_gpu
        logits_flat = reshape(chunk_logits, :)                 # flat view, no copy
        true_vals = logits_flat[lin_idx]                       # (n_chunk,) on GPU
        # count how many logits exceed the true-class logit per column
        exceeds = chunk_logits .> reshape(true_vals, 1, :)     # (n_classes, n_chunk) Bool
        chunk_errs = vec(sum(exceeds, dims=1))                 # (n_chunk,)
        errors[start:stop] = Int.(cpu(chunk_errs))
    end
    return errors
end


end  # module ProcessSC
