module Extract

using Flux, CUDA, Statistics, StatsBase

# encode was used below without being imported (mpool would throw UndefVarError)
let d = @__DIR__; d in LOAD_PATH || push!(LOAD_PATH, d); end
using Models: encode, encode_rank, masked_mean_pool

export mpool, get_embeds, get_embeds_lrecon, get_embeds_exp


function _batched_embed(forward_fn, X, batch_size::Int)
    embeds = Matrix{Float32}[]
    n = size(X, 2)
    for i in 1:batch_size:n
        b_idx = i:min(i + batch_size - 1, n)
        push!(embeds, cpu(forward_fn(X[:, b_idx])))
    end
    return hcat(embeds...)
end

# mpool(m, x) = dropdims(mean(encode(m, x), dims=2), dims=2)
mpool(m, x) = masked_mean_pool(encode_rank(m, x)...)   # PAD-aware encoder + mean over real tokens

get_embeds(pt_model, X_ranked, batch_size::Int) =
    _batched_embed(x -> mpool(pt_model, cu(Int32.(x))), X_ranked, batch_size)

function get_embeds_lrecon(pt_model, X_ranked, batch_size::Int)
    _batched_embed(X_ranked, batch_size) do x
        x_batch = cu(Int32.(x))
        # embedded = pt_model.embedding(x_batch)
        # pos_ids = cu(Int32.(1:size(embedded, 2)))
        # encoded = embedded .+ pt_model.pos_emb(pos_ids)
        # dropped = pt_model.emb_dropout(encoded)
        # transformed = pt_model.transformer(dropped)
        # dropdims(mean(transformed, dims=2), dims=2)
        masked_mean_pool(encode_rank(pt_model, x_batch)...)
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
        transformed = pt_model.transformer(dropped)
        dropdims(mean(transformed, dims=2), dims=2)
    end
end

end  # module Extract
