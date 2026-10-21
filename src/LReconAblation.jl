module LReconAblation

# lrecon ablation switches (all default to the original lrecon):
#   --final_ln 1        parameter-free per-token LayerNorm on the encoder output (student + teacher)
#   --target_k K        target = mean of per-token-normalized outputs of the top K teacher blocks (data2vec)
#   --target_norm s     batch (original: z-score over the batch) | token (per-token LN) | none
#   --ema_sched cosine  teacher decay goes ema_decay -> ema_end on a cosine schedule (BYOL / data2vec / GeneJepa)
#   --lr_loss cos       1 - cosine(pred, target) instead of MSE (GeneJepa, Cell-JEPA)
#   --vic_var / --vic_cov  VICReg variance / covariance terms on the masked predictions (GeneJepa)
#   (--weight_decay is passed to AdamW in the scripts)

using Flux, CUDA, Statistics
using Models: encode, encode_rank, rank_keep_mask

export lr_opts, lr_default_targets, tok_ln, lr_targets, lr_loss, ema_decay_at

_get(c, k, d) = something(get(c, k, nothing), d)

lr_opts(c::Dict) = (final_ln = _get(c, "final_ln", 0) == 1,
                    target_k = Int(_get(c, "target_k", 1)),
                    target_norm = string(_get(c, "target_norm", "batch")),
                    ema_sched = string(_get(c, "ema_sched", "fixed")),
                    ema_end = Float32(_get(c, "ema_end", 0.9999)),
                    loss = string(_get(c, "lr_loss", "mse")),
                    vic_var = Float32(_get(c, "vic_var", 0.0)),
                    vic_cov = Float32(_get(c, "vic_cov", 0.0)))

# original target path (last block, z-scored over the batch)
lr_default_targets(o) = !o.final_ln && o.target_k == 1 && o.target_norm == "batch"

# per-token LayerNorm over features, no affine
tok_ln(h) = (h .- mean(h, dims=1)) ./ (std(h, dims=1; corrected=false) .+ 1f-5)

# outputs of every encoder block for clean inputs (+ PAD keep mask for rank models)
function block_outputs(m, x, use_exp::Bool)
    if use_exp
        enc = m.proj(reshape(x, 1, size(x)...)) .+ m.pos_emb(cu(Int32.(1:size(x, 1))))
        h = m.emb_dropout(enc); keep = nothing; amask = nothing
    else
        keep, has_pad = rank_keep_mask(m.embedding, x)
        xl = has_pad ? ifelse.(keep, x, eltype(x)(size(m.embedding.weight, 2))) : x
        emb = m.embedding(xl)
        h = m.emb_dropout(emb .+ m.pos_emb(cu(Int32.(1:size(emb, 2)))))
        amask = has_pad ? reshape(keep, size(keep, 1), 1, 1, size(keep, 2)) : nothing
        has_pad || (keep = nothing)
    end
    outs = Any[]
    for layer in m.transformer.layers
        h = layer(h, amask)
        push!(outs, h)
    end
    return outs, keep
end

# z-score over the batch per (feature, position), PAD-aware
function _batch_std(t, keep)
    isnothing(keep) && return (t .- mean(t, dims=3)) ./ (std(t, dims=3) .+ 1f-6)
    w = reshape(Float32.(keep), 1, size(keep)...)
    n = sum(w, dims=3)
    mu = sum(t .* w, dims=3) ./ max.(n, 1f0)
    sd = sqrt.(sum(((t .- mu) .* w) .^ 2, dims=3) ./ max.(n .- 1f0, 1f0))
    return ((t .- mu) ./ (sd .+ 1f-6)) .* w .* Float32.(n .>= 2f0)
end

# teacher targets for the ablation arms (non-default target settings only)
function lr_targets(teacher, x_clean, use_exp::Bool, o)
    outs, keep = block_outputs(teacher, x_clean, use_exp)
    K = min(o.target_k, length(outs))
    sel = outs[end-K+1:end]
    t = (K == 1 && !o.final_ln) ? sel[end] : sum(tok_ln.(sel)) ./ Float32(K)
    t = o.target_norm == "batch" ? _batch_std(t, keep) :
        o.target_norm == "token" ? tok_ln(t) : t
    return isnothing(keep) ? t : t .* reshape(Float32.(keep), 1, size(keep)...)
end

# student loss on masked positions (+ optional VICReg on the predictions); returns (loss, decoded, targets)
function lr_loss(m, x, targets, mask, o)
    h = encode(m, x)
    o.final_ln && (h = tok_ln(h))
    ed = size(h, 1)
    mflat = vec(mask) .> 0
    any(mflat) || return 0f0, nothing, nothing
    emb = reshape(h, ed, :)[:, mflat]
    tgt = reshape(targets, ed, :)[:, mflat]
    dec = m.decoder(emb)
    n = size(dec, 2)
    l = if o.loss == "cos"
        cs = sum(dec .* tgt, dims=1) ./ (sqrt.(sum(dec .^ 2, dims=1)) .* sqrt.(sum(tgt .^ 2, dims=1)) .+ 1f-6)
        mean(1f0 .- cs)
    else
        sum((dec .- tgt) .^ 2) / (n * ed)
    end
    if o.vic_var > 0 || o.vic_cov > 0
        dc = dec .- mean(dec, dims=2)
        sd = sqrt.(sum(dc .^ 2, dims=2) ./ max(n - 1, 1) .+ 1f-4)
        l += o.vic_var * mean(relu.(1f0 .- sd))
        C = (dc * dc') ./ max(n - 1, 1)
        l += o.vic_cov * (sum(C .^ 2) - sum(abs2, sd .^ 2 .- 1f-4)) / ed
    end
    return l, dec, tgt
end

# teacher decay at step (1-based) of total
ema_decay_at(step, total, base, o) =
    o.ema_sched == "cosine" ? Float32(o.ema_end - (o.ema_end - base) * (cos(pi * min(step, total) / total) + 1) / 2) :
    Float32(base)

end  # module LReconAblation
