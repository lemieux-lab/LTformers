module Extract

using Flux, CUDA, Statistics, StatsBase

let d = @__DIR__; d in LOAD_PATH || push!(LOAD_PATH, d); end
using Models: encode, encode_rank, masked_mean_pool, ExpModel, ExpLReconModel, ExpEReconModel

export mpool, get_embeds, get_embeds_lrecon, get_embeds_exp, pool_embed, FINAL_LN

# --final_ln 1 (lrecon ablation arms): per-token LayerNorm (no affine) on encoder outputs before pooling
const FINAL_LN = Ref(false)
_out(h) = FINAL_LN[] ? (h .- mean(h, dims=1)) ./ (std(h, dims=1; corrected=false) .+ 1f-5) : h
_rank(m, x) = (r = encode_rank(m, x); (_out(r[1]), r[2]))


function _batched_embed(forward_fn, X, batch_size::Int)
    embeds = Matrix{Float32}[]
    n = size(X, 2)
    for i in 1:batch_size:n
        b_idx = i:min(i + batch_size - 1, n)
        push!(embeds, cpu(forward_fn(X[:, b_idx])))
    end
    return hcat(embeds...)
end

mpool(m, x) = masked_mean_pool(_rank(m, x)...)

get_embeds(pt_model, X_ranked, batch_size::Int) =
    _batched_embed(x -> mpool(pt_model, cu(Int32.(x))), X_ranked, batch_size)

function get_embeds_lrecon(pt_model, X_ranked, batch_size::Int)
    _batched_embed(X_ranked, batch_size) do x
        x_batch = cu(Int32.(x))
        masked_mean_pool(_rank(pt_model, x_batch)...)
    end
end

function get_embeds_exp(pt_model, X, batch_size::Int)
    _batched_embed(X, batch_size) do x
        x_batch = cu(Float32.(x))
        x3d = reshape(x_batch, 1, size(x_batch)...)
        projected = pt_model.proj(x3d)
        gene_ids = cu(Int32.(1:size(x_batch, 1)))
        combined = projected .+ pt_model.pos_emb(gene_ids)
        dropped = pt_model.emb_dropout(combined)
        transformed = _out(pt_model.transformer(dropped))
        dropdims(mean(transformed, dims=2), dims=2)
    end
end

# pooled embedding of a gpu batch (same pooling as get_embeds / get_embeds_exp)
function pool_embed(pt_model::Union{ExpModel, ExpLReconModel, ExpEReconModel}, x)
    x3d = reshape(x, 1, size(x)...)
    combined = pt_model.proj(x3d) .+ pt_model.pos_emb(cu(Int32.(1:size(x, 1))))
    return dropdims(mean(_out(pt_model.transformer(pt_model.emb_dropout(combined))), dims=2), dims=2)
end
pool_embed(pt_model, x) = mpool(pt_model, x)

end  # module Extract
