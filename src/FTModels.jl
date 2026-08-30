module FTModels

using Flux, CUDA, Functors, Statistics, JLD2

let d = @__DIR__; d in LOAD_PATH || push!(LOAD_PATH, d); end
using Models: Transf, RankModel, ExpModel, RankLReconModel, ExpLReconModel,
              RankEReconModel, ExpEReconModel, encode, fix_gpu_dropout
using Extract: get_embeds, get_embeds_lrecon, get_embeds_exp

export RankClassifier, ExpClassifier
export RankFTModel, ExpFTModel
export build_embm, build_e2em


# emb only models

struct RankClassifier{E<:Flux.Embedding, P<:Flux.Embedding, D<:Flux.Dropout, T<:Flux.Chain, H<:Flux.Chain}
    embedding::E
    pos_emb::P
    emb_dropout::D
    transformer::T
    head::H
end

Flux.@layer RankClassifier

function RankClassifier(; n_genes::Int, embed_dim::Int, n_layers::Int,
                          n_classifications::Int, n_heads::Int, hidden_dim::Int,
                          dropout_prob::Float64, seq_len::Int = n_genes)
    embedding = Flux.Embedding(n_genes + 1 => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return RankClassifier(embedding, pos_emb, emb_dropout, transformer, head)
end

function (m::RankClassifier)(x)
    embedded = m.embedding(x)
    pos_ids = cu(Int32.(1:size(embedded, 2)))
    encoded = embedded .+ m.pos_emb(pos_ids)
    dropped = m.emb_dropout(encoded)
    transformed = m.transformer(dropped)
    pooled = dropdims(mean(transformed, dims=2), dims=2)
    return m.head(pooled)
end


struct ExpClassifier{D<:Flux.Dense, E<:Flux.Embedding, O<:Flux.Dropout, T<:Flux.Chain, H<:Flux.Chain}
    proj::D
    pos_emb::E
    emb_dropout::O
    transformer::T
    head::H
end

Flux.@layer ExpClassifier

function ExpClassifier(; n_genes::Int, embed_dim::Int, n_layers::Int,
                         n_classifications::Int, n_heads::Int, hidden_dim::Int,
                         dropout_prob::Float64, seq_len::Int = n_genes)
    proj = Flux.Dense(1 => embed_dim)
    pos_emb = Flux.Embedding(seq_len => embed_dim)
    emb_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return ExpClassifier(proj, pos_emb, emb_dropout, transformer, head)
end

function (m::ExpClassifier)(x)
    x3d = reshape(x, 1, size(x)...)
    projected = m.proj(x3d)
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ m.pos_emb(gene_ids)
    dropped = m.emb_dropout(combined)
    transformed = m.transformer(dropped)
    pooled = dropdims(mean(transformed, dims=2), dims=2)
    return m.head(pooled)
end


# e2e only models

struct RankFTModel{P, H<:Flux.Chain}
    pretrained::P
    head::H
end

Flux.@layer RankFTModel

function RankFTModel(pt_model::RankModel;
    embed_dim::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = (
        embedding = pt_model.embedding,
        pos_emb = pt_model.pos_emb,
        emb_dropout = pt_model.emb_dropout,
        transformer = pt_model.transformer)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return RankFTModel(pretrained, head)
end

function RankFTModel(pt_model::RankLReconModel;
    embed_dim::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = (
        embedding = pt_model.embedding,
        pos_emb = pt_model.pos_emb,
        emb_dropout = pt_model.emb_dropout,
        transformer = pt_model.transformer)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return RankFTModel(pretrained, head)
end

function RankFTModel(pt_model::RankEReconModel;
    embed_dim::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = (
        embedding = pt_model.embedding,
        pos_emb = pt_model.pos_emb,
        emb_dropout = pt_model.emb_dropout,
        transformer = pt_model.transformer)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return RankFTModel(pretrained, head)
end

function (m::RankFTModel)(x)
    transformed = encode(m.pretrained, x)
    pooled = dropdims(mean(transformed, dims=2), dims=2)
    return m.head(pooled)
end


struct ExpFTModel{P, H<:Flux.Chain}
    pretrained::P
    head::H
end

Flux.@layer ExpFTModel

function ExpFTModel(pt_model::ExpModel;
    embed_dim::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = (
        proj = pt_model.proj,
        pos_emb = pt_model.pos_emb,
        emb_dropout = pt_model.emb_dropout,
        transformer = pt_model.transformer)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return ExpFTModel(pretrained, head)
end

function ExpFTModel(pt_model::ExpLReconModel;
    embed_dim::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = (
        proj = pt_model.proj,
        pos_emb = pt_model.pos_emb,
        emb_dropout = pt_model.emb_dropout,
        transformer = pt_model.transformer)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return ExpFTModel(pretrained, head)
end

function ExpFTModel(pt_model::ExpEReconModel;
    embed_dim::Int, hidden_dim::Int, n_classifications::Int, drop_prob::Float64)
    pretrained = (
        proj = pt_model.proj,
        pos_emb = pt_model.pos_emb,
        emb_dropout = pt_model.emb_dropout,
        transformer = pt_model.transformer)
    head = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(drop_prob),
        Flux.Dense(hidden_dim => n_classifications))
    return ExpFTModel(pretrained, head)
end

function (m::ExpFTModel)(x)
    x3d = reshape(x, 1, size(x)...)
    projected = m.pretrained.proj(x3d)
    gene_ids = cu(Int32.(1:size(x, 1)))
    combined = projected .+ m.pretrained.pos_emb(gene_ids)
    dropped = m.pretrained.emb_dropout(combined)
    transformed = m.pretrained.transformer(dropped)
    pooled = dropdims(mean(transformed, dims=2), dims=2)
    return m.head(pooled)
end


# building

function _clean_state(state, model)
    model_keys = Set(keys(Flux.state(model)))
    return NamedTuple(k => v for (k, v) in pairs(state) if k in model_keys)
end

# load architecture config saved alongside pretrained weights
# tries model_config.jld2 first, falls back to parsing params.txt from parent dir
function _load_arch_config(model_dir::String, config::Dict)
    arch_keys = ["embed_dim", "hidden_dim", "n_heads", "n_layers", "drop_prob"]
    # try saved JLD2 arch config
    cfg_path = joinpath(model_dir, "model_config.jld2")
    if isfile(cfg_path)
        saved = load(cfg_path)["arch_config"]
        merged = copy(config)
        for k in arch_keys
            if haskey(saved, k)
                merged[k] = saved[k]
            end
        end
        return merged
    end
    # fallback: parse params.txt from parent dir (existing pretrained checkpoints)
    params_path = joinpath(dirname(model_dir), "params.txt")
    if isfile(params_path)
        merged = copy(config)
        for line in eachline(params_path)
            m = match(r"^(\w+)\s*=\s*(.+)$", line)
            isnothing(m) && continue
            k, v = m.captures
            k in arch_keys || continue
            if k in ["embed_dim", "hidden_dim", "n_heads", "n_layers"]
                merged[k] = parse(Int, strip(v))
            elseif k == "drop_prob"
                merged[k] = parse(Float64, strip(v))
            end
        end
        if any(merged[k] != config[k] for k in arch_keys if haskey(config, k) && haskey(merged, k))
            println("NOTE: using arch config from pretrained params.txt ($(params_path))")
        end
        return merged
    end
    @warn "no model_config.jld2 or params.txt found for $model_dir; using finetune config arch params"
    return config
end

function build_embm(config::Dict, X_train, X_test, n_genes, n_classifications; X_val=nothing, seq_len::Int=n_genes)
    if config["modeltype"] == "mlp"
        ft_model = Flux.Chain(
            Flux.Dense(n_genes => config["hidden_dim"], gelu),
            Flux.LayerNorm(config["hidden_dim"]),
            Flux.Dropout(config["drop_prob"]),
            Flux.Dense(config["hidden_dim"] => n_classifications))
        ft_model = fix_gpu_dropout(cu(ft_model))
        if X_val !== nothing
            return ft_model, Float32.(X_train), Float32.(X_val), Float32.(X_test)
        end
        return ft_model, Float32.(X_train), Float32.(X_test)
    end

    # load arch config from pretrained checkpoint (handles n_layers/dim mismatches)
    ac = _load_arch_config(config["model_dir"], config)
    state = load("$(config["model_dir"])/model_state.jld2")["model_state"]

    if config["modeltype"] == "etf" && config["task"] == "lrecon"
        pt_model = ExpLReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"],
            seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        pt_model = fix_gpu_dropout(cu(pt_model))
        Flux.testmode!(pt_model)
        train_input = get_embeds_exp(pt_model, X_train, config["batch_size"])
        test_input = get_embeds_exp(pt_model, X_test, config["batch_size"])

    elseif config["modeltype"] == "etf" && config["task"] == "erecon"
        pt_model = ExpEReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"],
            seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        pt_model = fix_gpu_dropout(cu(pt_model))
        Flux.testmode!(pt_model)
        train_input = get_embeds_exp(pt_model, X_train, config["batch_size"])
        test_input = get_embeds_exp(pt_model, X_test, config["batch_size"])

    elseif config["modeltype"] == "etf" && config["task"] == "mlm"
        pt_model = ExpModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_classes=n_genes,
            n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            dropout_prob=ac["drop_prob"], seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        pt_model = fix_gpu_dropout(cu(pt_model))
        Flux.testmode!(pt_model)
        train_input = get_embeds_exp(pt_model, X_train, config["batch_size"])
        test_input = get_embeds_exp(pt_model, X_test, config["batch_size"])

    elseif config["modeltype"] == "rtf" && config["task"] == "lrecon"
        pt_model = RankLReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"],
            seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        pt_model = fix_gpu_dropout(cu(pt_model))
        Flux.testmode!(pt_model)
        train_input = get_embeds_lrecon(pt_model, X_train, config["batch_size"])
        test_input = get_embeds_lrecon(pt_model, X_test, config["batch_size"])

    elseif config["modeltype"] == "rtf" && config["task"] == "erecon"
        pt_model = RankEReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"],
            seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        pt_model = fix_gpu_dropout(cu(pt_model))
        Flux.testmode!(pt_model)
        train_input = get_embeds_lrecon(pt_model, X_train, config["batch_size"])
        test_input = get_embeds_lrecon(pt_model, X_test, config["batch_size"])

    else  # rtf + mlm
        pt_model = RankModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_classes=n_genes,
            n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            dropout_prob=ac["drop_prob"], seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        pt_model = fix_gpu_dropout(cu(pt_model))
        Flux.testmode!(pt_model)
        train_input = get_embeds(pt_model, X_train, config["batch_size"])
        test_input = get_embeds(pt_model, X_test, config["batch_size"])
    end

    # extract val embeddings if X_val provided
    if X_val !== nothing
        if config["modeltype"] == "etf"
            val_input = get_embeds_exp(pt_model, X_val, config["batch_size"])
        elseif config["task"] in ("lrecon", "erecon")
            val_input = get_embeds_lrecon(pt_model, X_val, config["batch_size"])
        else
            val_input = get_embeds(pt_model, X_val, config["batch_size"])
        end
    end

    ft_model = Flux.Chain(
        Flux.Dense(ac["embed_dim"] => config["hidden_dim"], gelu),
        Flux.LayerNorm(config["hidden_dim"]),
        Flux.Dropout(config["drop_prob"]),
        Flux.Dense(config["hidden_dim"] => n_classifications))
    ft_model = fix_gpu_dropout(cu(ft_model))

    if X_val !== nothing
        return ft_model, train_input, val_input, test_input
    end
    return ft_model, train_input, test_input
end

function build_e2em(config::Dict, n_classifications; n_genes::Int, seq_len::Int=n_genes)
    # load arch config from pretrained checkpoint (handles n_layers/dim mismatches)
    ac = _load_arch_config(config["model_dir"], config)
    state = load("$(config["model_dir"])/model_state.jld2")["model_state"]

    if config["modeltype"] == "etf" && config["task"] == "lrecon"
        pt_model = ExpLReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"])
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        ft_model = ExpFTModel(pt_model;
            embed_dim=ac["embed_dim"], hidden_dim=config["hidden_dim"],
            n_classifications=n_classifications, drop_prob=config["drop_prob"])

    elseif config["modeltype"] == "etf" && config["task"] == "erecon"
        pt_model = ExpEReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"])
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        ft_model = ExpFTModel(pt_model;
            embed_dim=ac["embed_dim"], hidden_dim=config["hidden_dim"],
            n_classifications=n_classifications, drop_prob=config["drop_prob"])

    elseif config["modeltype"] == "etf" && config["task"] == "mlm"
        pt_model = ExpModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_classes=n_genes,
            n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            dropout_prob=ac["drop_prob"])
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        ft_model = ExpFTModel(pt_model;
            embed_dim=ac["embed_dim"], hidden_dim=config["hidden_dim"],
            n_classifications=n_classifications, drop_prob=config["drop_prob"])

    elseif config["modeltype"] == "rtf" && config["task"] == "lrecon"
        pt_model = RankLReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"],
            seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        ft_model = RankFTModel(pt_model;
            embed_dim=ac["embed_dim"], hidden_dim=config["hidden_dim"],
            n_classifications=n_classifications, drop_prob=config["drop_prob"])

    elseif config["modeltype"] == "rtf" && config["task"] == "erecon"
        pt_model = RankEReconModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_heads=ac["n_heads"],
            hidden_dim=ac["hidden_dim"], dropout_prob=ac["drop_prob"],
            seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        ft_model = RankFTModel(pt_model;
            embed_dim=ac["embed_dim"], hidden_dim=config["hidden_dim"],
            n_classifications=n_classifications, drop_prob=config["drop_prob"])

    else  # rtf + mlm
        pt_model = RankModel(n_genes=n_genes, embed_dim=ac["embed_dim"],
            n_layers=ac["n_layers"], n_classes=n_genes,
            n_heads=ac["n_heads"], hidden_dim=ac["hidden_dim"],
            dropout_prob=ac["drop_prob"], seq_len=seq_len)
        Flux.loadmodel!(pt_model, _clean_state(state, pt_model))
        ft_model = RankFTModel(pt_model;
            embed_dim=ac["embed_dim"], hidden_dim=config["hidden_dim"],
            n_classifications=n_classifications, drop_prob=config["drop_prob"])
    end

    return ft_model
end


end  # module FTModels
