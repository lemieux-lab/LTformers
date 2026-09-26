module ProcessSC

# using Flux, CUDA, Statistics, Random, StatsBase, SparseArrays, Base.Threads
using Flux, CUDA, Statistics, Random, StatsBase, SparseArrays, Base.Threads, JLD2

let d = joinpath(@__DIR__, ".."); d in LOAD_PATH || push!(LOAD_PATH, d); end
using Preprocess: pad_token_id

# encode (sc_masked_loss) and the shared expression corruption (sc_mask!)
let s = joinpath(@__DIR__, ".."); s in LOAD_PATH || push!(LOAD_PATH, s); end
using Models: encode
using Train: corrupt_expr!

export log_normalize_col!, process_cell_topk_flat
export build_batch_rtf, build_batch_etf, build_batch_etf_hvg, batches_from_shard
export batches_from_shard_data
export sc_mask_input!, sc_mask_input_exp!, sc_mask_input_exp_rank!, sc_mask_input_erecon!, sc_masked_loss
export cell_to_dense_flat!, process_cell_topk_flat
export gpu_rank_errors
export sc_inverse_ranks_batch
export sc_gene_medians, set_sc_gene_medians!
export sc_shard_batches, sc_mask_buffers, sc_mask!, sc_masked_cache


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

# per-gene nonzero medians (log1p CP10k) on SC train shards, from scripts/pretrain/sc/compute_hvg.jl.
# loaded once per process (lazily, thread-safe); override the path with ENV["SC_MEDIANS_PATH"] or set_sc_gene_medians!
const _SC_MEDIANS = Ref{Union{Nothing,Vector{Float32}}}(nothing)
const _SC_MEDIANS_LOADED = Ref(false)
const _SC_MEDIANS_LOCK = ReentrantLock()

function set_sc_gene_medians!(meds::Union{Nothing,AbstractVector})
    lock(_SC_MEDIANS_LOCK) do
        _SC_MEDIANS[] = isnothing(meds) ? nothing : Float32.(meds)
        _SC_MEDIANS_LOADED[] = true
    end
end

function sc_gene_medians(n_coding::Int)
    _SC_MEDIANS_LOADED[] && return _SC_MEDIANS[]
    lock(_SC_MEDIANS_LOCK) do
        if !_SC_MEDIANS_LOADED[]
            path = get(ENV, "SC_MEDIANS_PATH", joinpath(@__DIR__, "..", "..", "data", "tahoe", "sc_gene_medians.jld2"))
            meds = isfile(path) ? Float32.(load(path, "medians")) : nothing
            if isnothing(meds)
                @warn "no SC gene medians at $path (run scripts/pretrain/sc/compute_hvg.jl): ranking SC cells WITHOUT median normalization"
            elseif length(meds) != n_coding
                @warn "SC gene medians at $path have $(length(meds)) genes, expected $n_coding: ranking WITHOUT median normalization"
                meds = nothing
            else
                println("SC gene medians: loaded $path")
            end
            _SC_MEDIANS[] = meds
            _SC_MEDIANS_LOADED[] = true
        end
    end
    return _SC_MEDIANS[]
end

# function process_cell_topk_flat(dense::Vector{Float32},
#                                 genes_flat::Vector{Int64}, offsets::Vector{Int64},
#                                 expr_flat::Vector{Float32}, cell_idx::Int,
#                                 token_to_idx::Dict{Int,Int}, n_coding::Int, top_k::Int)
#     cell_to_dense_flat!(dense, genes_flat, offsets, expr_flat, cell_idx, token_to_idx)
#
#     @inbounds for i in 1:n_coding
#         dense[i] += randn(Float32) * 1f-10
#     end
#     perm = partialsortperm(dense, 1:top_k, rev=true)  # O(n + k log k) vs O(n log n)
#     top_gene_ids = Int32.(perm)
#     top_expr_vals = dense[perm]
#     return top_gene_ids, top_expr_vals
# end
# ^ no median normalization, and undetected genes were ordered by random noise (cells with < top_k detected genes)

# geneformer-style (same rules as PB, see Preprocess.jl): detected genes by log1p CP10k / nonzero median, stable ties
# (gene index); if fewer than top_k are detected, the tail is PAD (pad_token_id(n_coding), expression 0), which the
# model ignores (Models.encode_rank attention mask + masked pooling). was: undetected genes in gene-index order.
# a cell with 0 detected genes keeps position 1 as the undetected gene 1 so attention never sees an all-PAD row.
# returns (gene ids in rank order, their expression values, number of detected genes)
function process_cell_topk_flat(dense::Vector{Float32},
                                genes_flat::Vector{Int64}, offsets::Vector{Int64},
                                expr_flat::Vector{Float32}, cell_idx::Int,
                                token_to_idx::Dict{Int,Int}, n_coding::Int, top_k::Int)
    cell_to_dense_flat!(dense, genes_flat, offsets, expr_flat, cell_idx, token_to_idx)
    meds = sc_gene_medians(n_coding)
    det = findall(>(0f0), dense)
    ratio = isnothing(meds) ? dense[det] : dense[det] ./ view(meds, det)
    order = det[sortperm(ratio; rev=true)]   # sortperm is stable -> ties by gene index
    n_det = length(order)
    top_gene_ids = Vector{Int32}(undef, top_k)
    m = min(top_k, n_det)
    @inbounds for r in 1:m
        top_gene_ids[r] = order[r]
    end
    # if m < top_k
    #     r = m
    #     @inbounds for g in 1:n_coding
    #         dense[g] > 0f0 && continue
    #         r += 1
    #         top_gene_ids[r] = g
    #         r == top_k && break
    #     end
    # end
    # top_expr_vals = dense[top_gene_ids]
    pad = pad_token_id(n_coding)
    top_expr_vals = zeros(Float32, top_k)
    @inbounds for r in 1:m
        top_expr_vals[r] = dense[top_gene_ids[r]]
    end
    @inbounds for r in (m + 1):top_k
        top_gene_ids[r] = pad
    end
    if m == 0 && top_k > 0
        top_gene_ids[1] = Int32(1)   # guard: never an all-PAD sequence (observed min is 298 detected genes)
    end
    return top_gene_ids, top_expr_vals, n_det
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
            local dense = Vector{Float32}(undef, n_coding)  # local: otherwise shared Core.Box across threads (race)
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
        @threads for j in 1:bs
            local dense = Vector{Float32}(undef, n_coding)  # local: otherwise shared Core.Box across threads (race)
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
            # dense = Vector{Float32}(undef, n_coding)
            local dense = Vector{Float32}(undef, n_coding)  # local: otherwise shared Core.Box across threads (race)
            cell_to_dense_flat!(dense, genes_flat, offsets, expr_flat,
                                cell_indices[j], token_to_idx)
            hvg_expr = dense[hvg_idx]
            batch_expr[:, j] = hvg_expr
            # compute ranks within HVG set (1 = highest expression)
            # perm = sortperm(hvg_expr, rev=true)
            hvg_meds = sc_gene_medians(n_coding)   # same median normalization as the RTF ranking
            perm = sortperm(isnothing(hvg_meds) ? hvg_expr : hvg_expr ./ view(hvg_meds, hvg_idx), rev=true)
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
            # perm = sortperm(hvg_expr, rev=true)
            hvg_meds = sc_gene_medians(n_coding)   # same median normalization as the RTF ranking
            perm = sortperm(isnothing(hvg_meds) ? hvg_expr : hvg_expr ./ view(hvg_meds, hvg_idx), rev=true)
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
    GC.gc()  # free PyCall temporaries on this thread; otherwise finalizers can run in @threads workers without the GIL
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
    pad = mask_id + Int32(1)   # PAD tail (pad_token_id) is never masked; mask count scales with the cell's real tokens
    for j in 1:n_samples
        # mask_pos = sample(1:n_rows, num_masked, replace=false)
        n_valid = count(!=(pad), view(X, :, j))
        mask_pos = n_valid == n_rows ? sample(1:n_rows, num_masked, replace=false) :
            sample(1:n_valid, min(n_valid, ceil(Int, n_valid * mask_ratio)), replace=false)
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
    pad = mask_id + Int32(1)   # PAD tail is never masked
    for j in 1:n_samples
        # mask_pos = sample(1:n_rows, num_masked, replace=false)
        n_valid = count(!=(pad), view(X_ids, :, j))
        mask_pos = n_valid == n_rows ? sample(1:n_rows, num_masked, replace=false) :
            sample(1:n_valid, min(n_valid, ceil(Int, n_valid * mask_ratio)), replace=false)
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


"""
    sc_inverse_ranks_batch(X_rtf_batch, n_coding) -> Matrix{Float32}

Per-batch version of sc_inverse_ranks for streaming rmlp finetuning.
Converts RTF gene-id tokens to inverse ranks normalized by n_coding.
X_rtf_batch: (top_k, bs) Int32 matrix where X[rank, sample] = gene_id
Returns: (n_coding, bs) Float32 matrix where out[gene_id, sample] = rank / n_coding
"""
function sc_inverse_ranks_batch(X_rtf_batch::Matrix{Int32}, n_coding::Int)
    inv = zeros(Float32, n_coding, size(X_rtf_batch, 2))
    @inbounds for j in axes(X_rtf_batch, 2)
        for r in axes(X_rtf_batch, 1)
            g = X_rtf_batch[r, j]
            if g > 0 && g <= n_coding
                inv[g, j] = Float32(r) / Float32(n_coding)
            end
        end
    end
    return inv
end



## SC pretrain masking. every function takes one run-config NamedTuple `mc` with fields
##   obj (:mlm | :erecon | :lrecon), use_exp, seq_len, mask_ratio, mask_id, batch_size,
##   coding_tokens, n_coding, top_k, token_to_idx, hvg_idx, load_shard_fn

# batches of one shard: ETF -> (expr, ranks) at fixed HVG positions; RTF erecon -> (ids, expr); RTF mlm/lrecon -> gene ids.
# a shard's last batch has n_cells mod batch_size cells; single-cell batches are skipped (lrecon standardizes teacher
# targets with std over the batch, which is NaN for one cell -> NaN gradient). drops at most 1 cell per shard
_batch_ncells(b) = b isa Tuple ? size(b[1], 2) : size(b, 2)
function sc_shard_batches(path::String, mc)
    modeltype = (mc.use_exp || mc.obj == :erecon) ? "etf" : "rtf"
    batches = batches_from_shard(path, mc.coding_tokens, mc.n_coding, mc.top_k, mc.batch_size;
                                 modeltype=modeltype, token_to_idx=mc.token_to_idx,
                                 load_shard_fn=mc.load_shard_fn, hvg_idx=mc.hvg_idx)
    return Iterators.filter(b -> _batch_ncells(b) >= 2, batches)
end

# preallocated (seq_len, batch_size) masking buffers for the run's objective/input
function sc_mask_buffers(mc)
    n, bs = mc.seq_len, mc.batch_size
    if mc.use_exp
        mc.obj == :mlm && return (x=Matrix{Float32}(undef, n, bs), y=Matrix{Int32}(undef, n, bs))
        return (x=Matrix{Float32}(undef, n, bs), m=falses(n, bs))   # erecon/lrecon ETF: donor-swap corruption
    end
    return (x=Matrix{Int32}(undef, n, bs), y=Matrix{mc.obj == :erecon ? Float32 : Int32}(undef, n, bs))
end

# mask one batch into `bufs` (returns views; copy before keeping):
#   mlm:    (x, y)                    y = labels, -100 = unmasked (ETF: rank within HVG set, RTF: gene id)
#   erecon: (x, y, m, ids_or_ranks)   ETF: y = clean expr, m = corrupted positions; RTF: y = expr at masked (-100 else)
#   lrecon: (x, clean, m, ids_or_ranks)
function sc_mask!(bufs, batch, mc)
    if mc.use_exp
        expr, ranks = batch
        bs = size(expr, 2)
        x = view(bufs.x, :, 1:bs)
        if mc.obj == :mlm
            y = view(bufs.y, :, 1:bs)
            sc_mask_input_exp_rank!(x, y, expr, ranks, mc.mask_ratio, -100)
            return (x=x, y=y)
        end
        m = view(bufs.m, :, 1:bs)
        corrupt_expr!(x, m, expr, mc.mask_ratio)
        mc.obj == :erecon && return (x=x, y=expr, m=m, ids_or_ranks=ranks)
        return (x=x, clean=expr, m=m, ids_or_ranks=ranks)
    elseif mc.obj == :erecon
        ids, expr = batch
        bs = size(expr, 2)
        x, y = view(bufs.x, :, 1:bs), view(bufs.y, :, 1:bs)
        sc_mask_input_erecon!(x, y, ids, expr, mc.mask_ratio, -100f0, mc.mask_id)
        return (x=x, y=y, m=(y .!= -100f0), ids_or_ranks=ids)
    end
    bs = size(batch, 2)
    x, y = view(bufs.x, :, 1:bs), view(bufs.y, :, 1:bs)
    sc_mask_input!(x, y, batch, mc.mask_ratio, -100, mc.mask_id)
    mc.obj == :mlm && return (x=x, y=y)
    return (x=x, clean=batch, m=(y .!= -100), ids_or_ranks=batch)
end

# copy every array field once (fields aliasing the same array stay aliased, e.g. RTF lrecon clean === ids_or_ranks)
function _copy_masked(nt)
    seen = IdDict{Any,Any}()
    return map(v -> v isa AbstractArray ? get!(() -> copy(v), seen, v) : v, nt)
end

# statically masked batches for val/test (same masks every eval)
function sc_masked_cache(shard_paths, mc)
    cache = NamedTuple[]
    for sp in shard_paths, batch in sc_shard_batches(sp, mc)
        push!(cache, _copy_masked(sc_mask!(sc_mask_buffers(mc), batch, mc)))
    end
    println("  cached $(length(cache)) batches from $(length(shard_paths)) shards")
    return cache
end


end  # module ProcessSC
