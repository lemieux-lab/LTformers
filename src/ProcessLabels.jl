module ProcessLabels

using Flux, JLD2, Random, StatsBase, Statistics, DataFrames, MultivariateStats

export get_labels, process_labels, oversmpl, downsmpl, dsplit, get_pt_idx, get_regression_pairs, get_regression_pairs_pca


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

# --- old single-gene lvl3 (replaced by PCA-based approach) ---
# # lincs lvl3: need to pair compounds across two cell lines
# # TODO need to still implement tahoe!!!
# function get_regression_pairs(expr::Matrix{Float32}, inst::DataFrame, gene_df::DataFrame,
#                               source_cell::Symbol, target_cell::Symbol, target_gene::String)
#     gene_idx = findfirst(gene_df.gene_symbol .== target_gene)
#     isnothing(gene_idx) && error("target gene '$target_gene' not found in gene DataFrame")
#     println("target gene: $target_gene (index $gene_idx)")
#
#     src_mask = inst.cell_iname .== source_cell
#     tgt_mask = inst.cell_iname .== target_cell
#     println("source cell line $source_cell: $(sum(src_mask)) samples")
#     println("target cell line $target_cell: $(sum(tgt_mask)) samples")
#
#     src_perts = Set(inst.pert_id[src_mask])
#     tgt_perts = Set(inst.pert_id[tgt_mask])
#     shared_perts = sort(collect(intersect(src_perts, tgt_perts)))
#
#     filter!(p -> p != :DMSO && p != Symbol("DMSO"), shared_perts)
#     println("shared pert_ids (excl. DMSO): $(length(shared_perts))")
#
#     n_genes = size(expr, 1)
#     n_compounds = length(shared_perts)
#     X_paired = Matrix{Float32}(undef, n_genes, n_compounds) # source cell line means
#     y_paired = Matrix{Float32}(undef, 1, n_compounds) # target gene in target cell
#
#     for (k, pid) in enumerate(shared_perts)
#         src_idx = findall((inst.pert_id .== pid) .& src_mask)
#         tgt_idx = findall((inst.pert_id .== pid) .& tgt_mask)
#         X_paired[:, k] = vec(mean(expr[:, src_idx], dims=2))
#         y_paired[1, k] = mean(expr[gene_idx, tgt_idx])
#     end
#
#     return X_paired, y_paired, shared_perts
# end

# --- PCA-based lvl3: predict PC1 of target cell line compound-mean profiles ---

"""
    get_regression_pairs_pca(expr, source_mask, target_mask, source_drugs, target_drugs)

Core PCA-based regression pairing logic shared across data formats.
Finds shared compounds, computes compound-mean profiles, fits PCA on target,
and returns source means as X and target PC1 scores as y.

Returns: (X_paired, y_paired, shared_perts, pca_model)
"""
function get_regression_pairs_pca(expr::Matrix{Float32},
                                   source_mask::BitVector, target_mask::BitVector,
                                   source_drugs::AbstractVector, target_drugs::AbstractVector)
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

    # fit PCA on target compound-means (observations as columns)
    pca_model = MultivariateStats.fit(PCA, Float64.(tgt_means); maxoutdim=2)
    pc_scores = Float32.(MultivariateStats.transform(pca_model, Float64.(tgt_means)))  # (2 x n_compounds)

    # variance explained diagnostics
    var_explained = principalvars(pca_model)
    total_var = tvar(pca_model)
    pct1 = round(var_explained[1] / total_var * 100, digits=1)
    pct2 = length(var_explained) >= 2 ? round(var_explained[2] / total_var * 100, digits=1) : 0.0
    println("PCA variance explained: PC1=$(pct1)%, PC2=$(pct2)%")

    # bimodality coefficient for PC1: BC = (skewness² + 1) / kurtosis
    pc1_vals = pc_scores[1, :]
    m = mean(pc1_vals); s = std(pc1_vals)
    if s > 0
        z = (pc1_vals .- m) ./ s
        skw = mean(z .^ 3)
        krt = mean(z .^ 4)
        bc = krt > 0 ? (skw^2 + 1) / krt : 0.0
        println("PC1 bimodality coefficient: $(round(bc, digits=3)) (>0.555 suggests bimodality — consider using PC2)")
    end

    # y = PC1 scores (1 x n_compounds)
    y_paired = reshape(pc1_vals, 1, n_compounds)
    X_paired = src_means

    return X_paired, y_paired, shared_perts, pca_model
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

    X, y, perts, pca = get_regression_pairs_pca(expr, src_mask, tgt_mask,
                                                  inst.pert_id, inst.pert_id)
    return X, y, perts
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

    X, y, perts, pca = get_regression_pairs_pca(expr, src_mask, tgt_mask,
                                                  df.drug, df.drug)
    return X, y, perts
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
                inverse_ranks_fn = nothing)
    fmt = get(config, "data_format", "tahoe")

    if config["level"] == "lvl3"
        src = Symbol(get(config, "source_cell", "MCF7"))
        tgt = Symbol(get(config, "target_cell", "PC3"))
        dose = get(config, "dose", "")
        if fmt == "lincs"
            isnothing(inst_df) && error("dsplit lvl3: inst_df required for LINCS regression")
            X, y, shared_perts = get_regression_pairs(data, inst_df, gene_df, src, tgt; dose=dose)
        else  # tahoe PB
            isnothing(label_source) && error("dsplit lvl3: label_source (DataFrame) required for Tahoe PB regression")
            X, y, shared_perts = get_regression_pairs(data, label_source, src, tgt; dose=dose)
        end
        n_genes = size(X, 1)

        if config["modeltype"] == "rtf"
            gene_medians = vec(median(X, dims=2)) .+ 1f-10
            X = rank_genes_fn(X, gene_medians)
        end

        X_train, X_val, X_test, train_idx, val_idx, test_idx = tvsplit_fn(X, 0.1f0, 0.1f0)
        y_train, y_val, y_test = y[:, train_idx], y[:, val_idx], y[:, test_idx]

        return (; X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx,
                  n_genes, n_classifications=1, cidx_dict=nothing, cs=nothing)
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

    if config["modeltype"] == "emlp"
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

    elseif config["modeltype"] == "rmlp"
        gene_medians = vec(median(X, dims=2)) .+ 1f-10
        X_ranked = rank_genes_fn(X, gene_medians)
        X_inv = inverse_ranks_fn(X_ranked)
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
        gene_medians = vec(median(X, dims=2)) .+ 1f-10
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
              n_genes, n_classifications=n_cls, cidx_dict, cs)
end


end  # module ProcessLabels
