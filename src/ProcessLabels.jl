module ProcessLabels

using Flux, JLD2, Random, StatsBase, Statistics, DataFrames

export get_labels, process_labels, oversmpl, downsmpl, dsplit, get_pt_idx, get_regression_pairs


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

# lincs lvl3: need to pair compounds across two cell lines
# TODO need to still implement tahoe!!!
function get_regression_pairs(expr::Matrix{Float32}, inst::DataFrame, gene_df::DataFrame,
                              source_cell::Symbol, target_cell::Symbol, target_gene::String)
    gene_idx = findfirst(gene_df.gene_symbol .== target_gene)
    isnothing(gene_idx) && error("target gene '$target_gene' not found in gene DataFrame")
    println("target gene: $target_gene (index $gene_idx)")

    src_mask = inst.cell_iname .== source_cell
    tgt_mask = inst.cell_iname .== target_cell
    println("source cell line $source_cell: $(sum(src_mask)) samples")
    println("target cell line $target_cell: $(sum(tgt_mask)) samples")

    src_perts = Set(inst.pert_id[src_mask])
    tgt_perts = Set(inst.pert_id[tgt_mask])
    shared_perts = sort(collect(intersect(src_perts, tgt_perts)))

    filter!(p -> p != :DMSO && p != Symbol("DMSO"), shared_perts)
    println("shared pert_ids (excl. DMSO): $(length(shared_perts))")

    n_genes = size(expr, 1)
    n_compounds = length(shared_perts)
    X_paired = Matrix{Float32}(undef, n_genes, n_compounds) # source cell line means
    y_paired = Matrix{Float32}(undef, 1, n_compounds) # target gene in target cell

    for (k, pid) in enumerate(shared_perts)
        src_idx = findall((inst.pert_id .== pid) .& src_mask)
        tgt_idx = findall((inst.pert_id .== pid) .& tgt_mask)
        X_paired[:, k] = vec(mean(expr[:, src_idx], dims=2))
        y_paired[1, k] = mean(expr[gene_idx, tgt_idx])
    end

    return X_paired, y_paired, shared_perts
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
        return nothing, nothing, nothing
    end
    indices_path = "$model_dir/indices.jld2"
    if !isfile(indices_path)
        println("no $indices_path, using new random split")
        return nothing, nothing, nothing
    end
    pt_idx = load(indices_path)
    d = Dict(orig_i => new_i for (new_i, orig_i) in enumerate(label_idx))
    train_idx = [d[i] for i in pt_idx["train_indices"] if haskey(d, i)]
    test_idx = [d[i] for i in pt_idx["test_indices"] if haskey(d, i)]
    return train_idx, test_idx, pt_idx
end

function dsplit(data::Matrix{Float32}, config::Dict; label_path::String = "",
                label_source = nothing,
                inst_df = nothing, gene_df = nothing,
                ttsplit_fn = error("ttsplit_fn required"),
                rank_genes_fn = error("rank_genes_fn required"))
    fmt = get(config, "data_format", "tahoe")

    if config["level"] == "lvl3"
        if fmt == "lincs"
            isnothing(inst_df) && error("dsplit lvl3: inst_df required for LINCS regression")
            isnothing(gene_df) && error("dsplit lvl3: gene_df required for LINCS regression")
            src = Symbol(get(config, "source_cell", "MCF7"))
            tgt = Symbol(get(config, "target_cell", "PC3"))
            target_gene = get(config, "target_gene", "IGFBP3")
            X, y, shared_perts = get_regression_pairs(data, inst_df, gene_df, src, tgt, target_gene)
        else  # tahoe
            return # TODO
        end
        n_genes = size(X, 1)

        if config["modeltype"] == "rtf"
            gene_medians = vec(median(X, dims=2)) .+ 1f-10
            X = rank_genes_fn(X, gene_medians)
        end

        X_train, X_test, train_idx, test_idx = ttsplit_fn(X, 0.2f0)
        y_train, y_test = y[:, train_idx], y[:, test_idx]

        return (; X_train, X_test, y_train, y_test, train_idx, test_idx,
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

    if config["modeltype"] == "mlp"
        train_idx, test_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_test, train_idx, test_idx = ttsplit_fn(X, 0.2f0)
        else
            X_train, X_test = X[:, train_idx], X[:, test_idx]
        end

    elseif config["modeltype"] == "etf"
        train_idx, test_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_test, train_idx, test_idx = ttsplit_fn(X, 0.2f0)
        else
            X_train, X_test = X[:, train_idx], X[:, test_idx]
        end

    else  # rtf
        gene_medians = vec(median(X, dims=2)) .+ 1f-10
        X_ranked = rank_genes_fn(X, gene_medians)
        train_idx, test_idx, pt_idx = get_pt_idx(label_idx, model_dir)
        if isnothing(train_idx)
            X_train, X_test, train_idx, test_idx = ttsplit_fn(X_ranked, 0.2f0)
        else
            X_train, X_test = X_ranked[:, train_idx], X_ranked[:, test_idx]
        end
    end

    y_train, y_test = y_oh[:, train_idx], y_oh[:, test_idx]

    cidx_dict, cs = config["level"] == "lvl2" ? oversmpl(y_train) : (nothing, nothing)
    return (; X_train, X_test, y_train, y_test, train_idx, test_idx,
              n_genes, n_classifications=n_cls, cidx_dict, cs)
end


end  # module ProcessLabels
