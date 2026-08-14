module Train

using Flux, CUDA, Statistics, Random, StatsBase

let d = @__DIR__; d in LOAD_PATH || push!(LOAD_PATH, d); end
using Models: encode

export mask_input!, mask_input_exp!, mask_input_erecon!, mask_input_exp_erecon!
export corrupt_expr!
export compute_lr, masked_logitcrossentropy
export masked_mse_loss, masked_mse_loss_1d
export masked_mlm_loss, masked_erecon_loss, masked_lrecon_loss
export ema_update!


# masking

function mask_input!(X_masked, mask_labels, X::Matrix, mask_ratio::Float64, # rank to rank
                     mask_val, mask_id, offset::Bool = false)
    copyto!(X_masked, X)
    fill!(mask_labels, mask_val)
    n_rows, n_samples = size(X)
    idx = (1 + offset):n_rows
    num_masked = ceil(Int, length(idx) * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(idx, num_masked, replace=false)
        for pos in mask_pos
            mask_labels[pos, j] = X[pos, j]
            X_masked[pos, j] = mask_id
        end
    end
    return X_masked, mask_labels
end

function mask_input_exp!(X_masked, mask_labels, X_expr::Matrix{Float32}, X_ranks::Matrix, # exp to rank
                         mask_ratio::Float64, mask_val)
    copyto!(X_masked, X_expr)
    fill!(mask_labels, mask_val)
    n_rows, n_samples = size(X_expr)
    num_masked = ceil(Int, n_rows * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(1:n_rows, num_masked, replace=false)
        for pos in mask_pos
            mask_labels[pos, j] = X_ranks[pos, j]
            X_masked[pos, j] = -1f0
        end
    end
    return X_masked, mask_labels
end

function mask_input_erecon!(X_masked, expr_labels, X_ranks::Matrix, X_expr::Matrix{Float32}, # rank to exp
                            mask_ratio::Float64, mask_val, mask_id)
    copyto!(X_masked, X_ranks)
    fill!(expr_labels, mask_val)
    n_rows, n_samples = size(X_ranks)
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

function mask_input_exp_erecon!(X_masked, expr_labels, X_expr::Matrix{Float32}, # exp to exp
                                mask_ratio::Float64, mask_val)
    copyto!(X_masked, X_expr)
    fill!(expr_labels, mask_val)
    n_rows, n_samples = size(X_expr)
    num_masked = ceil(Int, n_rows * mask_ratio)
    for j in 1:n_samples
        mask_pos = sample(1:n_rows, num_masked, replace=false)
        for pos in mask_pos
            expr_labels[pos, j] = X_expr[pos, j]
            X_masked[pos, j] = -1f0
        end
    end
    return X_masked, expr_labels
end

function corrupt_expr!(X_corrupted, corrupt_mask, X_expr::Matrix{Float32}, corrupt_ratio::Float64)
    copyto!(X_corrupted, X_expr)
    fill!(corrupt_mask, false)
    n_genes, n_samples = size(X_expr)
    n_corrupt = ceil(Int, n_genes * corrupt_ratio)
    for j in 1:n_samples
        corrupt_pos = sample(1:n_genes, n_corrupt, replace=false)
        for pos in corrupt_pos
            donor = rand(1:n_samples)
            X_corrupted[pos, j] = X_expr[pos, donor]
            corrupt_mask[pos, j] = true
        end
    end
    return X_corrupted, corrupt_mask
end


# loss

function masked_logitcrossentropy(logits, y, n_classes)
    logits_flat = reshape(logits, size(logits, 1), :)
    y_flat = vec(y)
    mask = (y_flat .!= -100) .& (y_flat .<= n_classes) .& (y_flat .> 0)
    if !any(mask)
        return 0.0f0, nothing, nothing
    end

    y_masked = y_flat[mask]
    logits_masked = logits_flat[:, mask]

    y_oh = Flux.onehotbatch(y_masked, 1:n_classes)
    return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
end

function masked_mse_loss(decoded, target_embeds, mask_3d)
    diff = (decoded .- target_embeds) .* mask_3d
    n_masked = sum(mask_3d[1, :, :])
    return sum(diff .^ 2) / (n_masked * size(decoded, 1))
end

function masked_mse_loss_1d(preds, targets)
    mask = targets .!= -100f0
    n_masked = sum(mask)
    diff = (preds .- targets) .* mask
    return sum(diff .^ 2) / n_masked
end


# masked-only fwd pass

function _encode_exp(model, x)
    x3d = reshape(x, 1, size(x)...)
    projected = model.proj(x3d)
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ model.pos_emb(gene_ids)
    dropped = model.emb_dropout(combined)
    return model.transformer(dropped)
end

function _classify_masked(classifier, transformed, y_gpu, n_classes)
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

function _regress_masked(regressor, transformed, y_flat, mask_flat)
    ed = size(transformed, 1)
    transformed_2d = reshape(transformed, ed, :)
    if !any(mask_flat)
        return 0f0, nothing, nothing
    end
    masked_emb = transformed_2d[:, mask_flat]
    preds = vec(regressor(masked_emb))
    y_masked = y_flat[mask_flat]
    return sum((preds .- y_masked) .^ 2) / sum(mask_flat), preds, y_masked
end

function _decode_masked(decoder, transformed, target_embeds, mask_flat)
    ed = size(transformed, 1)
    transformed_2d = reshape(transformed, ed, :)
    target_2d = reshape(target_embeds, ed, :)
    if !any(mask_flat)
        return 0f0, nothing, nothing
    end
    masked_emb = transformed_2d[:, mask_flat]
    masked_tgt = target_2d[:, mask_flat]
    decoded = decoder(masked_emb)
    return sum((decoded .- masked_tgt) .^ 2) / (sum(mask_flat) * ed), decoded, masked_tgt
end

# MLM
function masked_mlm_loss(model, x_gpu, y_gpu, n_classes)
    transformed = encode(model, x_gpu)
    _classify_masked(model.classifier, transformed, y_gpu, n_classes)
end

# ER
function masked_erecon_loss(model, x_gpu, y_gpu)
    transformed = encode(model, x_gpu)
    y_flat = vec(y_gpu)
    mask_flat = y_flat .!= -100f0
    _regress_masked(model.regressor, transformed, y_flat, mask_flat)
end

function masked_erecon_loss(model, x_gpu, y_gpu, corrupt_mask_gpu)
    transformed = _encode_exp(model, x_gpu)
    y_flat = vec(y_gpu)
    mask_flat = vec(corrupt_mask_gpu) .> 0f0
    _regress_masked(model.regressor, transformed, y_flat, mask_flat)
end

# LR
function masked_lrecon_loss(model, x_gpu, target_embeds, mask_2d)
    transformed = encode(model, x_gpu)
    mask_flat = vec(mask_2d) .> 0
    _decode_masked(model.decoder, transformed, target_embeds, mask_flat)
end


# misc

function compute_lr(epoch::Int, n_epochs::Int, base_lr::Float64, warmup_epochs::Int)
    if epoch <= warmup_epochs
        return base_lr * epoch / warmup_epochs
    else
        progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return base_lr * 0.5 * (1.0 + cos(π * progress))
    end
end

function ema_update!(ema_model, model, decay::Float32)
    ps_ema = Flux.trainables(ema_model)
    ps_model = Flux.trainables(model)
    for (p_ema, p_model) in zip(ps_ema, ps_model)
        @. p_ema = decay * p_ema + (1f0 - decay) * p_model
    end
end


end  # module Train
