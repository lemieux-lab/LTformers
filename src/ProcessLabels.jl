module ProcessLabels

using Flux, JLD2, Random, StatsBase, Statistics, DataFrames, MultivariateStats

let d = @__DIR__; d in LOAD_PATH || push!(LOAD_PATH, d); end
# using Preprocess: nonzero_medians
using Preprocess: nonzero_medians, gene_medians_for, rank_feature_k, rank_features

export get_labels, process_labels, oversmpl, downsmpl, dsplit, get_pt_idx, get_regression_pairs, get_regression_pairs_pca, identity_baseline


function get_labels(data::Matrix{Float32}, level::String, label_path::String)
    if level == "lvl1"
        mfc = load(label_path)["mfc"]
        return data, mfc, 1:size(data, 2)
    elseif level == "lvl2"
        pert_id = load(label_path)["pert_id"]
        y = pert_id
        counts = countmap(y)
        valid_labels = Set(k for (k, v) in counts if 1000 < v < 20000)
        idx = findall(l -> l in valid_labels, y)
        return data[:, idx], y[idx], idx
    end
end

# tahoe pseudobulk (cell_line, drug)
function get_labels(data_expr::Matrix{Float32}, df::DataFrame, level::String)
    if level == "lvl1"
        labels = String.(df.cell_line)
        return data_expr, labels, 1:size(data_expr, 2)
    elseif level == "lvl2"
        y = String.(df.drug)
        non_dmso = findall(l -> l != "DMSO", y)
        y_filt = y[non_dmso]
        data_filt = data_expr[:, non_dmso]
        counts = countmap(y_filt)
        valid = Set(k for (k, v) in counts if v >= 100)
        idx = findall(l -> l in valid, y_filt)
        return data_filt[:, idx], y_filt[idx], non_dmso[idx]
    end
end

# --- PCA-based lvl3: predict PC1 of target cell line compound-mean profiles ---

"""
    get_regression_pairs_pca(expr, source_mask, target_mask, source_drugs, target_drugs)

Core PCA-based regression pairing logic shared across data formats.
Finds shared compounds, computes compound-mean profiles, fits PCA on target,
and returns source means as X and target PC1 scores as y.

Returns: (X_paired, y_paired, shared_perts, pca_model, split)
PCA is fit on train-split target compounds only; `split` holds compound indices.
"""
# function _compound_split(n::Int; val_ratio::AbstractFloat=0.1f0, test_ratio::AbstractFloat=0.1f0)
#     idx = shuffle(1:n)
# seeded (local RNG, global stream untouched): every lvl3 run for a source/target pair gets the same compounds in
# train/val/test and hence the same PCA/PC1 targets and identity baseline, whatever the model or gene set
# (was: a new random split per run, so e.g. elog vs elog_full lvl3 results weren't comparable)
function _compound_split(n::Int; val_ratio::AbstractFloat=0.1f0, test_ratio::AbstractFloat=0.1f0, seed::Integer=42)
    idx = shuffle(MersenneTwister(seed), 1:n)
    n_test = floor(Int, n * test_ratio)
    n_val  = floor(Int, n * val_ratio)
    s_test = n - n_test
    s_val  = s_test - n_val
    return (; train_idx=idx[1:s_val], val_idx=idx[s_val+1:s_test], test_idx=idx[s_test+1:end])
end

function get_regression_pairs_pca(expr::Matrix{Float32},
                                   source_mask::BitVector, target_mask::BitVector,
                                   source_drugs::AbstractVector, target_drugs::AbstractVector;
                                   val_ratio::AbstractFloat=0.1f0, test_ratio::AbstractFloat=0.1f0)
    src_perts = Set(source_drugs[source_mask])
    tgt_perts = Set(target_drugs[target_mask])
    shared_perts = sort(collect(intersect(src_perts, tgt_perts)))
    filter!(p -> p != :DMSO && p != Symbol("DMSO") && string(p) != "DMSO", shared_perts)
    println("shared compounds (excl. DMSO): $(length(shared_perts))")
    length(shared_perts) < 5 && error("too few shared compounds ($(length(shared_perts))) for PCA regression")

    n_genes = size(expr, 1)
    n_compounds = length(shared_perts)

    # compute compound-mean profiles for source and target
    src_means = Matrix{Float32}(undef, n_genes, n_compounds)
    tgt_means = Matrix{Float32}(undef, n_genes, n_compounds)
    for (k, pid) in enumerate(shared_perts)
        src_idx = findall((source_drugs .== pid) .& source_mask)
        tgt_idx = findall((target_drugs .== pid) .& target_mask)
        src_means[:, k] = vec(mean(expr[:, src_idx], dims=2))
        tgt_means[:, k] = vec(mean(expr[:, tgt_idx], dims=2))
    end
    
    # fit on train compounds only so test compounds don't shape the PC1 axis
    split = _compound_split(n_compounds; val_ratio=val_ratio, test_ratio=test_ratio)
    pca_model = MultivariateStats.fit(PCA, Float64.(tgt_means[:, split.train_idx]); maxoutdim=2)
    pc_scores = Float32.(MultivariateStats.transform(pca_model, Float64.(tgt_means)))  # (2 x n_compounds)
    println("PCA fit on $(length(split.train_idx)) train compounds (val=$(length(split.val_idx)), test=$(length(split.test_idx)))")

    # variance explained diagnostics
    var_explained = principalvars(pca_model)
    total_var = tvar(pca_model)
    pct1 = round(var_explained[1] / total_var * 100, digits=1)
    pct2 = length(var_explained) >= 2 ? round(var_explained[2] / total_var * 100, digits=1) : 0.0
    println("PCA variance explained: PC1=$(pct1)%, PC2=$(pct2)%")

    # bimodality coefficient for PC1: BC = (skewness² + 1) / kurtosis
    pc1_vals = pc_scores[1, :]
    pc1_train = pc1_vals[split.train_idx]
    m = mean(pc1_train); s = std(pc1_train)
    if s > 0
        z = (pc1_train .- m) ./ s
        skw = mean(z .^ 3)
        krt = mean(z .^ 4)
        bc = krt > 0 ? (skw^2 + 1) / krt : 0.0
        println("PC1 bimodality coefficient: $(round(bc, digits=3)) (>0.555 suggests bimodality — consider using PC2)")
    end

    # y = PC1 scores (1 x n_compounds)
    y_paired = reshape(pc1_vals, 1, n_compounds)
    X_paired = src_means

    return X_paired, y_paired, shared_perts, pca_model, split
end


# lincs lvl3 (PCA-based)
function get_regression_pairs(expr::Matrix{Float32}, inst::DataFrame, gene_df::DataFrame,
                              source_cell::Symbol, target_cell::Symbol; dose::String="")
    src_mask = BitVector(inst.cell_iname .== source_cell)
    tgt_mask = BitVector(inst.cell_iname .== target_cell)

    # dose filtering — LINCS pert_dose is Vector{Symbol} e.g. Symbol("10"), Symbol("1.11111")
    if dose != ""
        dose_sym = Symbol(dose)
        dose_mask = BitVector(inst.pert_dose .== dose_sym)
        src_mask .&= dose_mask
        tgt_mask .&= dose_mask
        println("dose filter: $(dose) → source=$(sum(src_mask)), target=$(sum(tgt_mask)) samples")
    end

    println("source cell line $source_cell: $(sum(src_mask)) samples")
    println("target cell line $target_cell: $(sum(tgt_mask)) samples")

    X, y, perts, pca, split = get_regression_pairs_pca(expr, src_mask, tgt_mask,
                                                         inst.pert_id, inst.pert_id)
    return X, y, perts, pca, split
end

# tahoe pseudobulk lvl3 (PCA-based)
function get_regression_pairs(expr::Matrix{Float32}, df::DataFrame,
                              source_cell::Symbol, target_cell::Symbol; dose::String="")
    src_mask = BitVector(df.cell_line .== source_cell)
    tgt_mask = BitVector(df.cell_line .== target_cell)

    # dose filtering — Tahoe doses are Vector{Symbol} e.g. Symbol("5.0 uM"), Symbol("0.05 uM")
    if dose != ""
        dose_sym = Symbol("$(dose) uM")
        dose_mask = BitVector(df.dose .== dose_sym)
        if sum(dose_mask) == 0
            # fallback: try matching as plain Symbol (without " uM" suffix)
            dose_sym = Symbol(dose)
            dose_mask = BitVector(df.dose .== dose_sym)
        end
        src_mask .&= dose_mask
        tgt_mask .&= dose_mask
        println("dose filter: $(dose_sym) → source=$(sum(src_mask)), target=$(sum(tgt_mask)) samples")
    end

    println("source cell line $source_cell: $(sum(src_mask)) samples")
    println("target cell line $target_cell: $(sum(tgt_mask)) samples")

    X, y, perts, pca, split = get_regression_pairs_pca(expr, src_mask, tgt_mask,
                                                         df.drug, df.drug)
    return X, y, perts, pca, split
end


function identity_baseline(X_test::Matrix{Float32}, y_test::Matrix{Float32}, pca_model)
    isnothing(pca_model) && return (; r2=NaN, pearson=NaN, rmse=NaN)
    src_pc = Float32.(MultivariateStats.transform(pca_model, Float64.(X_test)))
    preds = src_pc[1, :]
    trues = vec(y_test)
    ss_res = sum((preds .- trues) .^ 2)
    ss_tot = sum((trues .- mean(trues)) .^ 2)
    r2 = 1.0 - ss_res / ss_tot
    pearson = length(preds) > 1 ? cor(preds, trues) : NaN
    rmse = sqrt(mean((preds .- trues) .^ 2))
    return (; r2, pearson, rmse)
end


function process_labels(y)
    labels = unique(y)
    ids = Dict(l => i for (i, l) in enumerate(labels))
    return Flux.onehotbatch([ids[l] for l in y], 1:length(labels)), length(labels)
end

function oversmpl(y_train)
    labels = Flux.onecold(cpu(y_train))
    d = Dict{Int, Vector{Int}}()
    for (i, label) in enumerate(labels)
        push!(get!(d, label, Int[]), i)
    end
    return d, collect(keys(d))
end

function downsmpl(data_expr, df, ratio::Float64, format::String)
    d = Dict{Tuple{String, String}, Vector{Int}}()
    if format == "lincs"
        for (i, (pt, ci)) in enumerate(zip(df.pert_type, df.cell_iname))
            push!(get!(d, (String(pt), String(ci)), Int[]), i)
        end
    else  # tahoe
        for (i, (drug, cl)) in enumerate(zip(df.drug, df.cell_line))
            push!(get!(d, (String(drug), String(cl)), Int[]), i)
        end
    end
    selected = Int[]
    for idx in values(d)
        n_select = max(1, round(Int, length(idx) * ratio))
        append!(selected, sample(idx, n_select, replace=false))
    end
    sort!(selected)
    return data_expr[:, selected], selected
end


# splitting

function get_pt_idx(label_idx, model_dir::String)
    if model_dir == ""
        println("no model_dir set, using new random split")
        return nothing, nothing, nothing, nothing
    end
    indices_path = "$model_dir/indices.jld2"
    # checkpoints live in <run>/best or <run>/final but indices.jld2 is written to <run>/ -> fall back to the parent
    if !isfile(indices_path) && basename(rstrip(model_dir, '/')) in ("best", "final")
        parent_path = joinpath(dirname(rstrip(model_dir, '/')), "indices.jld2")
        if isfile(parent_path)
            println("using pretrain split from $parent_path")
            indices_path = parent_path
        end
    end
    if !isfile(indices_path)
        println("no $indices_path, using new random split")
        return nothing, nothing, nothing, nothing
    end
    pt_idx = load(indices_path)
    d = Dict(orig_i => new_i for (new_i, orig_i) in enumerate(label_idx))
    train_idx = [d[i] for i in pt_idx["train_indices"] if haskey(d, i)]
    test_idx = [d[i] for i in pt_idx["test_indices"] if haskey(d, i)]
    val_idx = if haskey(pt_idx, "val_indices")
        [d[i] for i in pt_idx["val_indices"] if haskey(d, i)]
    else
        println("  (no val_indices in pretrain checkpoint — will create val from train)")
        nothing
    end
    return train_idx, test_idx, val_idx, pt_idx
end

function dsplit(data::Matrix{Float32}, config::Dict; label_path::String = "",
                label_source = nothing,
                inst_df = nothing, gene_df = nothing,
                ttsplit_fn = error("ttsplit_fn required"),
                tvsplit_fn = nothing,
                rank_genes_fn = error("rank_genes_fn required"),
                inverse_ranks_fn = nothing,
                hvg_idx_lvl3 = nothing)   # expression models, lvl3: HVG subset applied after pairing (targets use all genes)
    fmt = get(config, "data_format", "tahoe")

    if config["level"] == "lvl3"
        src = Symbol(get(config, "source_cell", "MCF7"))
        tgt = Symbol(get(config, "target_cell", "PC3"))
        dose = get(config, "dose", "")
        if fmt == "lincs"
            isnothing(inst_df) && error("dsplit lvl3: inst_df required for LINCS regression")
            X, y, shared_perts, pca_model, split = get_regression_pairs(data, inst_df, gene_df, src, tgt; dose=dose)
        else  # tahoe PB
            isnothing(label_source) && error("dsplit lvl3: label_source (DataFrame) required for Tahoe PB regression")
            X, y, shared_perts, pca_model, split = get_regression_pairs(data, label_source, src, tgt; dose=dose)
        end
        n_genes = size(X, 1)
        train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx

        # computed on raw expression, before any rank transform
        id_baseline = identity_baseline(X[:, test_idx], y[:, test_idx], pca_model)

        if !isnothing(hvg_idx_lvl3) && config["modeltype"] in ("etf", "emlp", "elog")
            X = X[hvg_idx_lvl3, :]
            n_genes = size(X, 1)
        end

        if config["modeltype"] == "rtf"
            # gene_medians = nonzero_medians(X)
            gene_medians = gene_medians_for(config, X)   # train-split medians from file (Preprocess)
            X = rank_genes_fn(X, gene_medians)
        elseif config["modeltype"] in ("rmlp", "rlog")
            isnothing(inverse_ranks_fn) && error("dsplit lvl3: inverse_ranks_fn required for $(config["modeltype"])")
            # gene_medians = nonzero_medians(X)
            # X_ranked = rank_genes_fn(X, gene_medians)
            # X = Float32.(inverse_ranks_fn(X_ranked)) ./ Float32(n_genes)
            gene_medians = gene_medians_for(config, X)
            X_ranked = rank_genes_fn(X, gene_medians)
            # top-k: (k+1-r)/k, absent 0; full (--rank_top_k 0): (n_det+1-r)/n, undetected 0 (Preprocess.rank_features)
            X = rank_features(X_ranked, vec(sum(X .> 0, dims=1)), n_genes, rank_feature_k(config, n_genes);
                         encoding=Symbol(get(config, "rank_encoding", "rev")))
        end

        X_train, X_val, X_test = X[:, train_idx], X[:, val_idx], X[:, test_idx]
        y_train, y_val, y_test = y[:, train_idx], y[:, val_idx], y[:, test_idx]

        return (; X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx,
                  n_genes, n_classifications=1, cidx_dict=nothing, cs=nothing, pca_model, id_baseline)
    end

    if fmt == "lincs"
        X, y, label_idx = get_labels(data, config["level"], label_path)
    else  # tahoe
        X, y, label_idx = get_labels(data, label_source, config["level"])
    end
    y_oh, n_cls = process_labels(y)
    n_genes = size(X, 1)

    model_dir = get(config, "model_dir", "")

    # get val from train when pretrain checkpoint lacks val_indices
    function _split_val_from_train(train_idx)
        n_val = floor(Int, length(train_idx) * 0.125)
        shuffled = shuffle(train_idx)
        return shuffled[n_val+1:end], shuffled[1:n_val]
    end

    if config["modeltype"] in ("emlp", "elog")
        train_idx, test_idx, val_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_val, X_test, train_idx, val_idx, test_idx = tvsplit_fn(X, 0.1f0, 0.1f0)
        else
            if isnothing(val_idx)
                train_idx, val_idx = _split_val_from_train(train_idx)
            end
            X_train, X_val, X_test = X[:, train_idx], X[:, val_idx], X[:, test_idx]
        end

    elseif config["modeltype"] == "etf"
        train_idx, test_idx, val_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_val, X_test, train_idx, val_idx, test_idx = tvsplit_fn(X, 0.1f0, 0.1f0)
        else
            if isnothing(val_idx)
                train_idx, val_idx = _split_val_from_train(train_idx)
            end
            X_train, X_val, X_test = X[:, train_idx], X[:, val_idx], X[:, test_idx]
        end

    elseif config["modeltype"] in ("rmlp", "rlog")
        # gene_medians = nonzero_medians(X)
        # X_ranked = rank_genes_fn(X, gene_medians)
        # X_inv = Float32.(inverse_ranks_fn(X_ranked)) ./ Float32(n_genes)
        gene_medians = gene_medians_for(config, X)   # train-split medians from file (Preprocess)
        X_ranked = rank_genes_fn(X, gene_medians)
        # top-k: (k+1-r)/k, absent 0; full (--rank_top_k 0): (n_det+1-r)/n, undetected 0 (Preprocess.rank_features)
        X_inv = rank_features(X_ranked, vec(sum(X .> 0, dims=1)), n_genes, rank_feature_k(config, n_genes);
                         encoding=Symbol(get(config, "rank_encoding", "rev")))
        train_idx, test_idx, val_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_val, X_test, train_idx, val_idx, test_idx = tvsplit_fn(X_inv, 0.1f0, 0.1f0)
        else
            if isnothing(val_idx)
                train_idx, val_idx = _split_val_from_train(train_idx)
            end
            X_train, X_val, X_test = X_inv[:, train_idx], X_inv[:, val_idx], X_inv[:, test_idx]
        end

    else  # rtf
        # gene_medians = nonzero_medians(X)
        gene_medians = gene_medians_for(config, X)   # train-split medians from file, same as pretraining
        X_ranked = rank_genes_fn(X, gene_medians)
        train_idx, test_idx, val_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_val, X_test, train_idx, val_idx, test_idx = tvsplit_fn(X_ranked, 0.1f0, 0.1f0)
        else
            if isnothing(val_idx)
                train_idx, val_idx = _split_val_from_train(train_idx)
            end
            X_train, X_val, X_test = X_ranked[:, train_idx], X_ranked[:, val_idx], X_ranked[:, test_idx]
        end
    end

    y_train, y_val, y_test = y_oh[:, train_idx], y_oh[:, val_idx], y_oh[:, test_idx]

    cidx_dict, cs = config["level"] == "lvl2" ? oversmpl(y_train) : (nothing, nothing)
    
    return (; X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx,
              n_genes, n_classifications=n_cls, cidx_dict, cs, pca_model=nothing, id_baseline=nothing)
end


end  # module ProcessLabels
