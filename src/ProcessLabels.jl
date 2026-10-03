module ProcessLabels

using Flux, JLD2, Random, StatsBase, Statistics, DataFrames, MultivariateStats

let d = @__DIR__; d in LOAD_PATH || push!(LOAD_PATH, d); end
using Preprocess: nonzero_medians, gene_medians_for, rank_feature_k, rank_features, plate_delta

export get_labels, process_labels, oversmpl, downsmpl, dsplit, get_pt_idx, get_regression_pairs, get_regression_pairs_pca, identity_baseline


# keep (optional): columns allowed before labels are chosen (bad plates, delta filters)
# lvl2: trt_cp compounds with > min_n profiles
function get_labels(data::Matrix{Float32}, level::String, label_path::String; keep = nothing,
                    pert_type = nothing, min_n::Integer = 500)
    if level == "lvl1"
        mfc = load(label_path)["mfc"]
        isnothing(keep) && return data, mfc, 1:size(data, 2)
        idx = findall(keep)
        return data[:, idx], mfc[idx], idx
    elseif level == "lvl2"
        pert_id = load(label_path)["pert_id"]
        y = pert_id
        isnothing(pert_type) && error("get_labels LINCS lvl2: pert_type (inst_df.pert_type) required to keep trt_cp only")
        ok = BitVector(string.(pert_type) .== "trt_cp")
        isnothing(keep) || (ok .&= keep)
        counts = countmap(y[ok])
        valid_labels = Set(k for (k, v) in counts if v > min_n)
        idx = findall(i -> ok[i] && y[i] in valid_labels, eachindex(y))
        println("LINCS lvl2: trt_cp compounds with > $min_n profiles: $(length(valid_labels)) classes, $(length(idx)) profiles")
        return data[:, idx], y[idx], idx
    end
end

# tahoe pseudobulk (cell_line, drug)
function get_labels(data_expr::Matrix{Float32}, df::DataFrame, level::String; keep = nothing)
    if level == "lvl1"
        labels = String.(df.cell_line)
        isnothing(keep) && return data_expr, labels, 1:size(data_expr, 2)
        idx = findall(keep)
        return data_expr[:, idx], labels[idx], idx
    elseif level == "lvl2"
        y = String.(df.drug)
        non_dmso = isnothing(keep) ? findall(l -> l != "DMSO", y) : findall(i -> keep[i] && y[i] != "DMSO", eachindex(y))
        y_filt = y[non_dmso]
        data_filt = data_expr[:, non_dmso]
        counts = countmap(y_filt)
        valid = Set(k for (k, v) in counts if v >= 100)
        idx = findall(l -> l in valid, y_filt)
        return data_filt[:, idx], y_filt[idx], non_dmso[idx]
    end
end

# PCA lvl3: predict PC1 of target compound means

# seeded compound split
function _compound_split(n::Int; val_ratio::AbstractFloat=0.1f0, test_ratio::AbstractFloat=0.1f0, seed::Integer=42)
    idx = shuffle(MersenneTwister(seed), 1:n)
    n_test = floor(Int, n * test_ratio)
    n_val  = floor(Int, n * val_ratio)
    s_test = n - n_test
    s_val  = s_test - n_val
    return (; train_idx=idx[1:s_val], val_idx=idx[s_val+1:s_test], test_idx=idx[s_test+1:end])
end

# source compound means -> target PC1 scores
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

    # compound-mean profiles
    src_means = Matrix{Float32}(undef, n_genes, n_compounds)
    tgt_means = Matrix{Float32}(undef, n_genes, n_compounds)
    for (k, pid) in enumerate(shared_perts)
        src_idx = findall((source_drugs .== pid) .& source_mask)
        tgt_idx = findall((target_drugs .== pid) .& target_mask)
        src_means[:, k] = vec(mean(expr[:, src_idx], dims=2))
        tgt_means[:, k] = vec(mean(expr[:, tgt_idx], dims=2))
    end
    
    # fit on train compounds only
    split = _compound_split(n_compounds; val_ratio=val_ratio, test_ratio=test_ratio)
    pca_model = MultivariateStats.fit(PCA, Float64.(tgt_means[:, split.train_idx]); maxoutdim=2)
    pc_scores = Float32.(MultivariateStats.transform(pca_model, Float64.(tgt_means)))  # (2 x n_compounds)
    println("PCA fit on $(length(split.train_idx)) train compounds (val=$(length(split.val_idx)), test=$(length(split.test_idx)))")

    # variance explained
    var_explained = principalvars(pca_model)
    total_var = tvar(pca_model)
    pct1 = round(var_explained[1] / total_var * 100, digits=1)
    pct2 = length(var_explained) >= 2 ? round(var_explained[2] / total_var * 100, digits=1) : 0.0
    println("PCA variance explained: PC1=$(pct1)%, PC2=$(pct2)%")

    # PC1 bimodality coefficient
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

    # dose filter
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

    # dose filter
    if dose != ""
        dose_sym = Symbol("$(dose) uM")
        dose_mask = BitVector(df.dose .== dose_sym)
        if sum(dose_mask) == 0
            # fallback: plain symbol
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

# seeded group split (indices only)
function _group_split_idx(groups::AbstractVector, val_ratio::AbstractFloat, test_ratio::AbstractFloat;
                          seed::Integer = 42)
    ug = shuffle(MersenneTwister(seed), sort(unique(groups)))
    n_test = floor(Int, length(ug) * test_ratio)
    n_val = floor(Int, length(ug) * val_ratio)
    test_g, val_g = Set(ug[1:n_test]), Set(ug[n_test+1:n_test+n_val])
    test_idx = findall(in(test_g), groups)
    val_idx = findall(in(val_g), groups)
    train_idx = findall(g -> !(g in test_g) && !(g in val_g), groups)
    return train_idx, val_idx, test_idx
end

# val/test = whole groups (wells); fractions are of groups, not samples
# seeded
function _group_tvsplit(X::Matrix, groups::AbstractVector, val_ratio::AbstractFloat, test_ratio::AbstractFloat;
                        seed::Integer = 42)
    ug = shuffle(MersenneTwister(seed), sort(unique(groups)))
    n_test = floor(Int, length(ug) * test_ratio)
    n_val = floor(Int, length(ug) * val_ratio)
    test_g, val_g = Set(ug[1:n_test]), Set(ug[n_test+1:n_test+n_val])
    test_idx = findall(in(test_g), groups)
    val_idx = findall(in(val_g), groups)
    train_idx = findall(g -> !(g in test_g) && !(g in val_g), groups)
    return X[:, train_idx], X[:, val_idx], X[:, test_idx], train_idx, val_idx, test_idx
end

# pretrain split, used by every finetune
const CANONICAL_SPLITS = Dict("lincs" => "data/lincs/pretrain_split.jld2",
                              "tahoe" => "data/tahoe/pb_pretrain_split.jld2")
const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

function split_path_for(config::Dict)
    p = something(get(config, "split_path", nothing), get(CANONICAL_SPLITS, get(config, "data_format", "tahoe"), ""))
    p == "" && return ""
    return isabspath(p) ? p : joinpath(REPO_ROOT, p)
end

function load_pt_split(path::String)
    isfile(path) || error("pretrain split file not found: $path")
    s = load(path)
    return (; train=Vector{Int}(s["train_indices"]), val=Vector{Int}(s["val_indices"]), test=Vector{Int}(s["test_indices"]))
end

# pretrain split -> positions in label_idx
function _map_split(label_idx, s)
    d = Dict(orig_i => new_i for (new_i, orig_i) in enumerate(label_idx))
    f(v) = [d[i] for i in v if haskey(d, i)]
    return f(s.train), f(s.test), f(s.val)
end

function get_pt_idx(label_idx, model_dir::String; split_path::String = "")
    if split_path != ""
        s = load_pt_split(split_path)
        println("using canonical pretrain split $split_path ($(length(s.train))/$(length(s.val))/$(length(s.test)))")
        # checkpoint split must match
        if model_dir != ""
            for p in ("$model_dir/indices.jld2", joinpath(dirname(rstrip(model_dir, '/')), "indices.jld2"))
                isfile(p) || continue
                ck = load(p)
                haskey(ck, "train_indices") || continue
                (sort(ck["train_indices"]) == sort(s.train) && sort(ck["test_indices"]) == sort(s.test)) ||
                    error("checkpoint split $p differs from canonical split $split_path")
                println("  checkpoint split $p matches")
                break
            end
        end
        train_idx, test_idx, val_idx = _map_split(label_idx, s)
        return train_idx, test_idx, val_idx, s
    end
    if model_dir == ""
        println("no model_dir set, using new random split")
        return nothing, nothing, nothing, nothing
    end
    indices_path = "$model_dir/indices.jld2"
    # indices.jld2 is in parent of best/final
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

# LINCS plates whose stored level 3 values are broken (flat, sd ~5e-4)
# excluded from every LINCS task by default
const LINCS_BAD_PLATES = ["REP.A010_JURKAT_24H_X3_B32"]

input_mode(config) = string(something(get(config, "input", nothing), "abs"))

# per-sample keep mask (bad plates; for --input delta also the delta filters) + DMSO / cell line / plate keys
function _sample_filters(data, config, fmt, inst_df, label_source)
    n = size(data, 2); delta = input_mode(config) == "delta"
    input_mode(config) in ("abs", "delta") || error("--input must be abs or delta, got $(input_mode(config))")
    if fmt == "lincs"
        isnothing(inst_df) && return nothing, nothing, nothing, nothing
        plate = string.(inst_df.det_plate); cl = string.(inst_df.cell_iname)
        bad = Set(string.(something(get(config, "exclude_plates", nothing), LINCS_BAD_PLATES)))
        keep = BitVector([!(p in bad) for p in plate])
        println("excluded plates $(collect(bad)): $(count(.!keep)) samples")
        is_dmso = BitVector((string.(inst_df.pert_type) .== "ctl_vehicle") .& (string.(inst_df.pert_id) .== "DMSO")) .& keep
        if delta
            # treated compounds with dose and time (controls all have an empty dose, so this only touches trt_cp)
            trt = (string.(inst_df.pert_type) .== "trt_cp") .& (string.(inst_df.pert_dose) .!= "") .&
                  (string.(inst_df.pert_time) .!= "-666")
            has_dmso = Set(zip(cl[is_dmso], plate[is_dmso]))
            matched = BitVector([(c, p) in has_dmso for (c, p) in zip(cl, plate)])
            n0 = count(keep)
            keep = keep .& trt .& matched
            println("delta filters: kept $(count(keep)) of $n0 (trt_cp with dose/time and a same cell line + plate DMSO)")
        end
        return keep, is_dmso, cl, plate
    else  # tahoe PB
        isnothing(label_source) && return nothing, nothing, nothing, nothing
        plate = string.(label_source.plate); cl = string.(label_source.cell_line)
        is_dmso = BitVector(string.(label_source.drug) .== "DMSO")
        delta || return nothing, is_dmso, cl, plate
        has_dmso = Set(zip(cl[is_dmso], plate[is_dmso]))
        keep = BitVector([(c, p) in has_dmso for (c, p) in zip(cl, plate)])
        println("delta filters: dropped $(count(.!keep)) pseudobulks without a same cell line + plate DMSO")
        return keep, is_dmso, cl, plate
    end
end

function dsplit(data::Matrix{Float32}, config::Dict; label_path::String = "",
                label_source = nothing,
                inst_df = nothing, gene_df = nothing,
                ttsplit_fn = error("ttsplit_fn required"),
                tvsplit_fn = nothing,
                rank_genes_fn = error("rank_genes_fn required"),
                inverse_ranks_fn = nothing,
                hvg_idx_lvl3 = nothing)  # lvl3 hvg applied after pairing
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

        # on raw expression
        id_baseline = identity_baseline(X[:, test_idx], y[:, test_idx], pca_model)

        if !isnothing(hvg_idx_lvl3) && config["modeltype"] in ("etf", "emlp", "elog")
            X = X[hvg_idx_lvl3, :]
            n_genes = size(X, 1)
        end

        if config["modeltype"] == "rtf"
            gene_medians = gene_medians_for(config, X)
            X = rank_genes_fn(X, gene_medians)
        elseif config["modeltype"] in ("rmlp", "rlog")
            isnothing(inverse_ranks_fn) && error("dsplit lvl3: inverse_ranks_fn required for $(config["modeltype"])")
            gene_medians = gene_medians_for(config, X)
            X_ranked = rank_genes_fn(X, gene_medians)
            X = rank_features(X_ranked, vec(sum(X .> 0, dims=1)), n_genes, rank_feature_k(config, n_genes))
        end

        X_train, X_val, X_test = X[:, train_idx], X[:, val_idx], X[:, test_idx]
        y_train, y_val, y_test = y[:, train_idx], y[:, val_idx], y[:, test_idx]

        return (; X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx,
                  n_genes, n_classifications=1, cidx_dict=nothing, cs=nothing, pca_model, id_baseline)
    end

    keep, is_dmso, cl_key, plate_key = _sample_filters(data, config, fmt, inst_df, label_source)
    if fmt == "lincs"
        X, y, label_idx = get_labels(data, config["level"], label_path; keep=keep,
                                     pert_type=(isnothing(inst_df) ? nothing : inst_df.pert_type),
                                     min_n=something(get(config, "lincs_min_n", nothing), 500))
    else  # tahoe
        X, y, label_idx = get_labels(data, label_source, config["level"]; keep=keep)
    end
    # --input delta: plate-matched DMSO delta of each kept sample (expression models only; rank delta not decided)
    if input_mode(config) == "delta"
        config["modeltype"] in ("elog", "emlp", "etf") ||
            error("--input delta is only implemented for expression models (elog, emlp, etf), got $(config["modeltype"])")
        X, matched = plate_delta(data, cl_key, plate_key, is_dmso, label_idx)
        all(matched) || error("delta: $(count(.!matched)) samples without plate-matched DMSO survived the filters")
        println("input = delta: plate-matched DMSO delta for $(length(label_idx)) samples")
    end
    y_oh, n_cls = process_labels(y)
    label_names = unique(y)  # same order as process_labels
    n_genes = size(X, 1)

    model_dir = get(config, "model_dir", "")
    groups = nothing

    # --group_split 1 (tahoe PB): all pseudobulks of a (drug, dose) go to one split, as in SC. a well holds all cell
    # lines, so a random pseudobulk split puts the same well/drug/dose in train (other cell lines) and test
    if something(get(config, "group_split", nothing), 0) == 1 && fmt == "lincs"
        # LINCS: whole detection plates held out (each plate is one cell line; plate narrows the lvl2 drugs to a
        # handful, so a random split lets a model use plate identity). folder tag _gpl
        isnothing(inst_df) && error("--group_split for LINCS needs inst_df")
        groups = String.(string.(inst_df.det_plate[label_idx]))
        tvsplit_fn = (X, v, t) -> _group_tvsplit(X, groups, v, t)
        model_dir = ""
        println("group split by plate: $(length(unique(groups))) plates, $(length(groups)) samples")
    elseif something(get(config, "group_split", nothing), 0) == 1
        # group = (drug, dose): replicate wells of a drug-dose stay on one side; DMSO grouped per well
        smp = String.(label_source.sample[label_idx]); drg = String.(label_source.drug[label_idx]); dse = String.(label_source.dose[label_idx])
        groups = [d == "DMSO" ? s : d * "|" * x for (s, d, x) in zip(smp, drg, dse)]
        tvsplit_fn = (X, v, t) -> _group_tvsplit(X, groups, v, t)
        model_dir = ""                              # don't reuse the pretrain (random) split
        println("group split by (drug, dose): $(length(unique(groups))) groups, $(length(unique(smp))) wells, $(length(groups)) pseudobulks")
    end

    # val from train if missing
    function _split_val_from_train(train_idx)
        n_val = floor(Int, length(train_idx) * 0.125)
        shuffled = shuffle(MersenneTwister(42), train_idx)
        return shuffled[n_val+1:end], shuffled[1:n_val]
    end

    # features
    Xf = if config["modeltype"] in ("emlp", "elog", "etf")
        X
    elseif config["modeltype"] in ("rmlp", "rlog")
        gene_medians = gene_medians_for(config, X)
        X_ranked = rank_genes_fn(X, gene_medians)
        rank_features(X_ranked, vec(sum(X .> 0, dims=1)), n_genes, rank_feature_k(config, n_genes))
    else  # rtf
        gene_medians = gene_medians_for(config, X)
        rank_genes_fn(X, gene_medians)
    end

    # pretrain split, or group split (test_full = all test groups)
    split_path = split_path_for(config)
    if !isnothing(groups)
        train_idx, val_full_idx, test_full_idx = _group_split_idx(groups, 0.1f0, 0.1f0)
        if split_path == ""
            @warn "no pretrain split file for $fmt: grouped test set not restricted to pretrain-unseen samples"
            val_idx, test_idx = val_full_idx, test_full_idx
            split_tag = "group"
        else
            s = load_pt_split(split_path)
            unseen = Set(vcat(s.val, s.test))
            val_idx = [k for k in val_full_idx if label_idx[k] in unseen]
            test_idx = [k for k in test_full_idx if label_idx[k] in unseen]
            split_tag = "group_ptunseen"
            println("pretrain-unseen subsets: val $(length(val_idx))/$(length(val_full_idx)), " *
                    "test $(length(test_idx))/$(length(test_full_idx)) samples " *
                    "($(length(unique(argmax.(eachcol(y_oh[:, test_idx]))))) classes in unseen test)")
        end
    else
        train_idx, test_idx, val_idx, _ = get_pt_idx(label_idx, model_dir; split_path=split_path)
        if isnothing(train_idx)
            @warn "no pretrain split for $fmt: falling back to a random split that depends on --seed"
            _, _, _, train_idx, val_idx, test_idx = tvsplit_fn(Xf, 0.1f0, 0.1f0)
            split_tag = "random"
        else
            isnothing(val_idx) && ((train_idx, val_idx) = _split_val_from_train(train_idx))
            split_tag = "pretrain"
        end
        val_full_idx, test_full_idx = val_idx, test_idx
    end
    X_train, X_val, X_test = Xf[:, train_idx], Xf[:, val_idx], Xf[:, test_idx]
    X_test_full = test_full_idx === test_idx ? X_test : Xf[:, test_full_idx]
    println("split ($split_tag): $(length(train_idx)) train, $(length(val_idx)) val, $(length(test_idx)) test" *
            (test_full_idx === test_idx ? "" : " ($(length(test_full_idx)) in full test groups)"))


    y_train, y_val, y_test = y_oh[:, train_idx], y_oh[:, val_idx], y_oh[:, test_idx]
    y_test_full = y_oh[:, test_full_idx]

    cidx_dict, cs = config["level"] == "lvl2" ? oversmpl(y_train) : (nothing, nothing)

    # *_idx index into label_idx
    return (; X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx,
              X_test_full, y_test_full, test_full_idx, val_full_idx, split_tag, label_idx, label_names,
              n_genes, n_classifications=n_cls, cidx_dict, cs, pca_model=nothing, id_baseline=nothing)
end


end  # module ProcessLabels
