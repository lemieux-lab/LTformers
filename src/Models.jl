module Models

using Flux, CUDA, Functors, Statistics

export Transf, encode, fix_gpu_dropout
export RankModel, ExpModel
export RankLReconModel, ExpLReconModel
export RankEReconModel, ExpEReconModel


# transformer

struct Transf{A<:Flux.MultiHeadAttention, D<:Flux.Dropout, N<:Flux.LayerNorm, M<:Flux.Chain}
    mha::A
    att_dropout::D
    att_norm::N
    mlp::M
    mlp_norm::N
end

function Transf(embed_dim::Int, hidden_dim::Int; n_heads::Int, dropout_prob::Float64)
    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim,
                                  nheads=n_heads, dropout_prob=dropout_prob)
    att_dropout = Flux.Dropout(dropout_prob)
    att_norm = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu),
                     Flux.Dropout(dropout_prob),
                     Flux.Dense(hidden_dim => embed_dim),
                     Flux.Dropout(dropout_prob))
    mlp_norm = Flux.LayerNorm(embed_dim)
    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@layer Transf

function (tf::Transf)(input)
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1]
    residualed = input + tf.att_dropout(atted)
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size)
    mlp_out_reshaped = reshape(tf.mlp(reshaped), embed_dim, seq_len, batch_size)
    return residualed + mlp_out_reshaped
end

function encode(components, x)
    embedded = components.embedding(x)
    pos_ids = cu(Int32.(1:size(embedded, 2)))
    encoded = embedded .+ components.pos_emb(pos_ids)
    dropped = components.emb_dropout(encoded)
    transformed = components.transformer(dropped)
    return transformed
end



# mlm

struct RankModel{E<:Flux.Embedding, P<:Flux.Embedding, D<:Flux.Dropout, T<:Flux.Chain, C<:Flux.Chain}
    embedding::E
    pos_emb::P
    emb_dropout::D
    transformer::T
    classifier::C
end

Flux.@layer RankModel

function RankModel(; n_genes::Int, embed_dim::Int, n_layers::Int, n_classes::Int,
                     n_heads::Int, hidden_dim::Int, dropout_prob::Float64,
                     seq_len::Int = n_genes)
    embedding = Flux.Embedding(n_genes + 1 => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    classifier = Flux.Chain(Flux.Dense(embed_dim => embed_dim, gelu),
                            Flux.LayerNorm(embed_dim),
                            Flux.Dense(embed_dim => n_classes))
    return RankModel(embedding, pos_emb, emb_dropout, transformer, classifier)
end

function (model::RankModel)(input)
    transformed = encode(model, input)
    return model.classifier(transformed)
end


struct ExpModel{D<:Flux.Dense, E<:Flux.Embedding, O<:Flux.Dropout, T<:Flux.Chain, C<:Flux.Chain, M<:AbstractArray}
    proj::D
    pos_emb::E
    emb_dropout::O
    transformer::T
    classifier::C
    mask_emb::M
end

Flux.@layer ExpModel

function ExpModel(; n_genes::Int, embed_dim::Int, n_layers::Int, n_classes::Int,
                    n_heads::Int, hidden_dim::Int, dropout_prob::Float64,
                    seq_len::Int = n_genes)
    proj = Flux.Dense(1 => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    classifier = Flux.Chain(Flux.Dense(embed_dim => embed_dim, gelu),
                            Flux.LayerNorm(embed_dim),
                            Flux.Dense(embed_dim => n_classes))
    mask_emb = randn(Float32, embed_dim) .* 0.02f0
    return ExpModel(proj, pos_emb, emb_dropout, transformer, classifier, mask_emb)
end

function (m::ExpModel)(x)
    x3d = reshape(x, 1, size(x)...)
    projected = m.proj(x3d)
    mask_3d = reshape(x .== -1f0, 1, size(x)...)
    projected = projected .* (1f0 .- mask_3d) .+ m.mask_emb .* mask_3d
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ m.pos_emb(gene_ids)
    dropped = m.emb_dropout(combined)
    transformed = m.transformer(dropped)
    return m.classifier(transformed)
end


# lrecon

struct RankLReconModel{E<:Flux.Embedding, P<:Flux.Embedding, O<:Flux.Dropout, T<:Flux.Chain, H<:Flux.Chain}
    embedding::E
    pos_emb::P
    emb_dropout::O
    transformer::T
    decoder::H
end

Flux.@layer RankLReconModel

function RankLReconModel(; n_genes::Int, embed_dim::Int, n_layers::Int,
                           n_heads::Int, hidden_dim::Int, dropout_prob::Float64,
                           seq_len::Int = n_genes)
    embedding = Flux.Embedding(n_genes + 1 => embed_dim)
    # pos_emb = Flux.Embedding(n_genes => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    decoder = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu),
                         Flux.LayerNorm(hidden_dim),
                         Flux.Dense(hidden_dim => embed_dim))
    return RankLReconModel(embedding, pos_emb, emb_dropout, transformer, decoder)
end

function (m::RankLReconModel)(input)
    embedded = m.embedding(input)
    pos_ids = cu(Int32.(1:size(embedded, 2)))
    encoded = embedded .+ m.pos_emb(pos_ids)
    dropped = m.emb_dropout(encoded)
    transformed = m.transformer(dropped)
    embed_dim, seq_len, batch_size = size(transformed)
    flat = reshape(transformed, embed_dim, seq_len * batch_size)
    decoded = reshape(m.decoder(flat), embed_dim, seq_len, batch_size)
    return decoded
end


struct ExpLReconModel{D<:Flux.Dense, E<:Flux.Embedding, O<:Flux.Dropout, T<:Flux.Chain, H<:Flux.Chain, M<:AbstractArray}
    proj::D
    pos_emb::E
    emb_dropout::O
    transformer::T
    decoder::H
    mask_emb::M
end

Flux.@layer ExpLReconModel

function ExpLReconModel(; n_genes::Int, embed_dim::Int, n_layers::Int,
                          n_heads::Int, hidden_dim::Int, dropout_prob::Float64,
                          seq_len::Int = n_genes)
    proj = Flux.Dense(1 => embed_dim)
    # pos_emb = Flux.Embedding(n_genes => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    decoder = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu),
                         Flux.LayerNorm(hidden_dim),
                         Flux.Dense(hidden_dim => embed_dim))
    mask_emb = randn(Float32, embed_dim) .* 0.02f0
    return ExpLReconModel(proj, pos_emb, emb_dropout, transformer, decoder, mask_emb)
end

function (m::ExpLReconModel)(x)
    x3d = reshape(x, 1, size(x)...)
    projected = m.proj(x3d)
    mask_3d = reshape(x .== -1f0, 1, size(x)...)
    projected = projected .* (1f0 .- mask_3d) .+ m.mask_emb .* mask_3d
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ m.pos_emb(gene_ids)
    dropped = m.emb_dropout(combined)
    transformed = m.transformer(dropped)
    embed_dim, seq_len, batch_size = size(transformed)
    flat = reshape(transformed, embed_dim, seq_len * batch_size)
    decoded = reshape(m.decoder(flat), embed_dim, seq_len, batch_size)
    return decoded
end


# erecon

struct RankEReconModel{E<:Flux.Embedding, P<:Flux.Embedding, O<:Flux.Dropout, T<:Flux.Chain, R<:Flux.Chain}
    embedding::E
    pos_emb::P
    emb_dropout::O
    transformer::T
    regressor::R
end

Flux.@layer RankEReconModel

function RankEReconModel(; n_genes::Int, embed_dim::Int, n_layers::Int,
                           n_heads::Int, hidden_dim::Int, dropout_prob::Float64,
                           seq_len::Int = n_genes)
    embedding = Flux.Embedding(n_genes + 1 => embed_dim)
    # pos_emb = Flux.Embedding(n_genes => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    regressor = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu),
                           Flux.LayerNorm(hidden_dim),
                           Flux.Dense(hidden_dim => 1))
    return RankEReconModel(embedding, pos_emb, emb_dropout, transformer, regressor)
end

function (m::RankEReconModel)(input)
    embedded = m.embedding(input)
    pos_ids = cu(Int32.(1:size(embedded, 2)))
    encoded = embedded .+ m.pos_emb(pos_ids)
    dropped = m.emb_dropout(encoded)
    transformed = m.transformer(dropped)
    embed_dim, seq_len, batch_size = size(transformed)
    flat = reshape(transformed, embed_dim, seq_len * batch_size)
    out = reshape(m.regressor(flat), seq_len, batch_size)
    return out
end


struct ExpEReconModel{D<:Flux.Dense, E<:Flux.Embedding, O<:Flux.Dropout, T<:Flux.Chain, R<:Flux.Chain, M<:AbstractArray}
    proj::D
    pos_emb::E
    emb_dropout::O
    transformer::T
    regressor::R
    mask_emb::M
end

Flux.@layer ExpEReconModel

function ExpEReconModel(; n_genes::Int, embed_dim::Int, n_layers::Int,
                          n_heads::Int, hidden_dim::Int, dropout_prob::Float64,
                          seq_len::Int = n_genes)
    proj = Flux.Dense(1 => embed_dim)
    # pos_emb = Flux.Embedding(n_genes => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    regressor = Flux.Chain(Flux.Dense(embed_dim => hidden_dim, gelu),
                           Flux.LayerNorm(hidden_dim),
                           Flux.Dense(hidden_dim => 1))
    mask_emb = randn(Float32, embed_dim) .* 0.02f0
    return ExpEReconModel(proj, pos_emb, emb_dropout, transformer, regressor, mask_emb)
end

function (m::ExpEReconModel)(x)
    x3d = reshape(x, 1, size(x)...)
    projected = m.proj(x3d)
    mask_3d = reshape(x .== -1f0, 1, size(x)...)
    projected = projected .* (1f0 .- mask_3d) .+ m.mask_emb .* mask_3d
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ m.pos_emb(gene_ids)
    dropped = m.emb_dropout(combined)
    transformed = m.transformer(dropped)
    embed_dim, seq_len, batch_size = size(transformed)
    flat = reshape(transformed, embed_dim, seq_len * batch_size)
    out = reshape(m.regressor(flat), seq_len, batch_size)
    return out
end


# encode exp 
function encode(model::Union{ExpModel, ExpLReconModel, ExpEReconModel}, x)
    x3d = reshape(x, 1, size(x)...)
    projected = model.proj(x3d)
    mask_3d = reshape(x .== -1f0, 1, size(x)...)
    projected = projected .* (1f0 .- mask_3d) .+ model.mask_emb .* mask_3d
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ model.pos_emb(gene_ids)
    dropped = model.emb_dropout(combined)
    return model.transformer(dropped)
end


# fix dropout rng for gpu
function fix_gpu_dropout(model)
    return fmap(model; exclude = x -> x isa Flux.Dropout) do x
        if x isa Flux.Dropout
            return Flux.Dropout(x.p, x.dims, x.active, CUDA.default_rng())
        end
        return x
    end
end


end  # module Models
