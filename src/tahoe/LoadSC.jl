module LoadSC

using CSV, DataFrames, Random, PyCall, SparseArrays, JLD2, Flux, StatsBase

# ranking rules + SC train-shard medians
let d = @__DIR__, s = joinpath(@__DIR__, ".."); d in LOAD_PATH || push!(LOAD_PATH, d); s in LOAD_PATH || push!(LOAD_PATH, s); end
using Preprocess: rank_features!, pad_token_id
using ProcessSC: sc_gene_medians

export load_gene_vocab, list_shards, shard_train_test_split, load_shard_split
export shard_train_val_test_split, load_shard_val_split
export split_shards, load_hvg_idx
export load_shard_pyarrow, load_shard_metadata, load_sc_finetune_data
export sc_finetune_metadata_scan, materialize_finetune_split
export prepare_shard_cell_map, finetune_batches_from_shard
export load_sc_finetune_data_streaming
export draw_train_cells, finetune_train_batches, _build_ft_batch, memlog

const _pq = PyNULL()
const _np = PyNULL()

function __init__()
    copy!(_pq, pyimport("pyarrow.parquet"))
    copy!(_np, pyimport("numpy"))
    # cap pyarrow's decode/IO thread pools to the SLURM allocation (default = every core on the node, 256 on oni,
    # which burst far past the job's cpus on each shard read). output is identical, only decode parallelism changes
    nt = something(tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "")), 8)
    pa = pyimport("pyarrow")
    pa.set_cpu_count(nt)
    pa.set_io_thread_count(nt)
end


function load_gene_vocab(meta_dir::String, coding_gene_path::String)
    py_json = pyimport("json")

    gene_vocab = Dict{Int, String}()
    open(joinpath(meta_dir, "gene_vocabulary.jsonl")) do f
        for line in eachline(f)
            d = py_json.loads(line)
            gene_vocab[convert(Int, d["token_id"])] = convert(String, d["gene_symbol"])
        end
    end
    # GC here w/o GIL segfaults
    GC.gc()

    df_coding = CSV.read(coding_gene_path, DataFrame; delim='\t')
    coding_symbols = Set(df_coding.symbol)
    coding_tokens_set = Set(tid for (tid, sym) in gene_vocab if sym in coding_symbols)

    sorted_coding = sort(collect(coding_tokens_set))
    token_to_idx = Dict{Int, Int}(tid => i for (i, tid) in enumerate(sorted_coding))
    n_coding = length(sorted_coding)

    println("Gene vocab loaded: $(length(gene_vocab)) total, $n_coding protein-coding")
    return sorted_coding, token_to_idx, n_coding
end

function list_shards(data_dir::String)
    files = sort(filter(f -> endswith(f, ".parquet"), readdir(data_dir)))
    return [joinpath(data_dir, f) for f in files]
end

function shard_train_test_split(shard_paths::Vector{String}, test_ratio::Float64 = 0.2)
    n = length(shard_paths)
    idx = shuffle(1:n)
    n_test = floor(Int, n * test_ratio)
    test_idx = sort(idx[1:n_test])
    train_idx = sort(idx[n_test+1:end])
    return shard_paths[train_idx], shard_paths[test_idx]
end

function load_shard_split(model_dir::String, all_shards::Vector{String}, test_ratio::Float64 = 0.2)
    split_path = joinpath(model_dir, "shard_split.jld2")
    if model_dir != "" && isfile(split_path)
        saved = load(split_path)
        println("loaded pretrain shard split from $split_path")
        return saved["train_shards"], saved["test_shards"]
    end
    println("no shard split found at $split_path, using new random split")
    return shard_train_test_split(all_shards, test_ratio)
end

# seeded shard split
function shard_train_val_test_split(shard_paths::Vector{String}, val_ratio::Float64 = 0.1, test_ratio::Float64 = 0.1;
                                    seed::Integer = 42)
    n = length(shard_paths)
    idx = shuffle(MersenneTwister(seed), 1:n)
    n_test = floor(Int, n * test_ratio)
    n_val  = floor(Int, n * val_ratio)
    test_idx  = sort(idx[1:n_test])
    val_idx   = sort(idx[n_test+1:n_test+n_val])
    train_idx = sort(idx[n_test+n_val+1:end])
    return shard_paths[train_idx], shard_paths[val_idx], shard_paths[test_idx]
end

function load_shard_val_split(model_dir::String, all_shards::Vector{String}, val_ratio::Float64 = 0.1, test_ratio::Float64 = 0.1)
    split_path = joinpath(model_dir, "shard_split.jld2")
    if model_dir != "" && isfile(split_path)
        saved = load(split_path)
        println("loaded pretrain shard split from $split_path")
        if haskey(saved, "val_shards")
            return saved["train_shards"], saved["val_shards"], saved["test_shards"]
        else
            println("  (no val_shards found — falling back to new 3-way split)")
        end
    else
        println("no shard split found at $split_path, using new random split")
    end
    return shard_train_val_test_split(all_shards, val_ratio, test_ratio)
end

# plate-matched DMSO delta for SC finetuning (--input delta): when set, every dense feature built from a shard is
# minus the mean DMSO_TF profile of the cell's (cell line, plate) (scripts/pretrain/sc/compute_dmso_means.jl).
# nothing = off (default; pretraining and all existing runs unchanged)
const SC_DELTA = Ref{Any}(nothing)
function set_sc_delta!(path::AbstractString)
    d = load(path)
    SC_DELTA[] = (; means=Float32.(d["means"]), key_to_col=Dict(k => i for (i, k) in enumerate(d["keys"])))
    println("SC delta: DMSO_TF means for $(length(d["keys"])) (cell line, plate) groups from $path"); flush(stdout)
end
export SC_DELTA, set_sc_delta!

function load_shard_pyarrow(path::String)
    t = _pq.read_table(path)
    genes_combined = t.column("genes").combine_chunks()
    expr_combined = t.column("expressions").combine_chunks()
    genes_flat = convert(Vector{Int64}, _np.array(genes_combined.values, copy=true))
    offsets = convert(Vector{Int64}, _np.array(genes_combined.offsets, copy=true))
    expr_flat = convert(Vector{Float32}, _np.array(expr_combined.values, copy=true))
    n_cells = length(offsets) - 1
    # return (; genes_flat, offsets, expr_flat, n_cells)
    isnothing(SC_DELTA[]) && return (; genes_flat, offsets, expr_flat, n_cells)
    # delta on: column of each cell's (cell line, plate) DMSO mean (0 = no DMSO for that group)
    cl = convert(Vector{String}, t.column("cell_line_id").to_pylist())
    pl = convert(Vector{String}, t.column("plate").to_pylist())
    k2c = SC_DELTA[].key_to_col
    delta_col = [get(k2c, (c, p), 0) for (c, p) in zip(cl, pl)]
    return (; genes_flat, offsets, expr_flat, n_cells, delta_col)
end

# subtract the cell's plate-matched DMSO mean in place (no-op unless the shard was loaded with delta on)
function _apply_delta!(dense::AbstractVector{Float32}, shard, ci::Int)
    hasproperty(shard, :delta_col) || return dense
    k = shard.delta_col[ci]
    k == 0 && error("SC delta: cell $ci has no DMSO_TF mean for its (cell line, plate)")
    dense .-= view(SC_DELTA[].means, :, k)
    return dense
end


function load_shard_metadata(path::String)
    t = _pq.read_table(path, columns=["drug", "sample", "cell_line_id"])
    n_cells = t.num_rows
    drug = convert(Vector{String}, t.column("drug").to_pylist())
    sample = convert(Vector{String}, t.column("sample").to_pylist())
    cell_line_id = convert(Vector{String}, t.column("cell_line_id").to_pylist())
    return (; n_cells, drug, sample, cell_line_id)
end


# materialize SC finetune data from shards
function load_sc_finetune_data(all_shards::Vector{String}, level::String,
                                token_to_idx::Dict{Int,Int}, n_coding::Int,
                                top_k::Int, modeltype::String;
                                pb_data_path::String = "",
                                hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                                subset_shards::Int = 0,
                                process_cell_topk_flat_fn = nothing,
                                cell_to_dense_flat_fn = nothing,
                                oversmpl_fn = nothing,
                                source_cell::String = "",
                                target_cell::String = "",
                                dose::String = "",
                                meta_dir::String = "",
                                regression_pairs_fn = nothing)

    # validate args
    if modeltype == "rtf" && isnothing(process_cell_topk_flat_fn)
        error("load_sc_finetune_data: process_cell_topk_flat_fn required for rtf modeltype")
    end
    if modeltype != "rtf" && isnothing(cell_to_dense_flat_fn)
        error("load_sc_finetune_data: cell_to_dense_flat_fn required for etf/mlp modeltype")
    end
    if level == "lvl2" && isnothing(oversmpl_fn)
        error("load_sc_finetune_data: oversmpl_fn required for lvl2")
    end

    # lvl3: pseudo-bulk + PCA regression
    if level == "lvl3"
        (source_cell == "" || target_cell == "") && error("load_sc_finetune_data lvl3: source_cell and target_cell required")
        isnothing(regression_pairs_fn) && error("load_sc_finetune_data lvl3: regression_pairs_fn required")
        isnothing(cell_to_dense_flat_fn) && error("load_sc_finetune_data lvl3: cell_to_dense_flat_fn required")
        return _load_sc_lvl3(all_shards, token_to_idx, n_coding, top_k, modeltype,
                              source_cell, target_cell, dose, meta_dir,
                              cell_to_dense_flat_fn, process_cell_topk_flat_fn,
                              regression_pairs_fn;
                              subset_shards=subset_shards, hvg_idx=hvg_idx)
    end

    # valid labels
    valid_drugs = nothing
    if level == "lvl2"
        pb_data_path == "" && error("load_sc_finetune_data: pb_data_path required for lvl2")
        pb_df = JLD2.load(pb_data_path)["df"]
        pb_drugs = String.(pb_df.drug)
        non_dmso = filter(d -> d != "DMSO", pb_drugs)
        counts = countmap(non_dmso)
        valid_drugs = Set(k for (k, v) in counts if v >= 100)
        println("lvl2: $(length(valid_drugs)) valid drugs from PB data (≥100 PB samples, non-DMSO)")
    end

    # pass 1 metadata
    shards_to_scan = subset_shards > 0 ? all_shards[1:min(subset_shards, length(all_shards))] : all_shards
    println("[pass 1] scanning metadata from $(length(shards_to_scan)) shards...")
    flush(stdout)

    # (shard, cell_idx, label, sample_id)
    cell_records = Tuple{String, Int, String, String}[]
    for (si, sp) in enumerate(shards_to_scan)
        meta = load_shard_metadata(sp)
        for i in 1:meta.n_cells
            if level == "lvl1"
                label = meta.cell_line_id[i]
            else
                label = meta.drug[i]
                (label == "DMSO" || !(label in valid_drugs)) && continue
            end
            push!(cell_records, (sp, i, label, meta.sample[i]))
        end
        if si % 500 == 0
            println("  scanned $si / $(length(shards_to_scan)) shards, $(length(cell_records)) valid cells so far")
            flush(stdout)
        end
    end
    println("[pass 1] done: $(length(cell_records)) valid cells from $(length(shards_to_scan)) shards")
    flush(stdout)

    # sample-level split
    unique_samples = unique(r[4] for r in cell_records)
    # shuffle!(unique_samples)  # global RNG: split depended on --seed
    shuffle!(MersenneTwister(42), unique_samples)  # fixed split (split_seed 42), as in sc_finetune_metadata_scan
    n_test = floor(Int, length(unique_samples) * 0.1)
    n_val  = floor(Int, length(unique_samples) * 0.1)
    test_samples  = Set(unique_samples[1:n_test])
    val_samples   = Set(unique_samples[n_test+1:n_test+n_val])
    train_samples = Set(unique_samples[n_test+n_val+1:end])
    println("sample split: $(length(train_samples)) train, $(length(val_samples)) val, $(length(test_samples)) test samples")

    train_cells = filter(r -> r[4] in train_samples, cell_records)
    val_cells   = filter(r -> r[4] in val_samples, cell_records)
    test_cells  = filter(r -> r[4] in test_samples, cell_records)
    println("cell split: $(length(train_cells)) train, $(length(val_cells)) val, $(length(test_cells)) test cells")
    flush(stdout)

    # process labels
    all_labels = [r[3] for r in cell_records]
    unique_labels = sort(unique(all_labels))
    label_to_id = Dict(l => i for (i, l) in enumerate(unique_labels))
    n_cls = length(unique_labels)
    println("n_classifications: $n_cls")

    # pass 2 expression
    println("[pass 2] extracting expression features...")
    flush(stdout)

    _process_cell_topk_flat = process_cell_topk_flat_fn
    _cell_to_dense_flat! = cell_to_dense_flat_fn

    function _materialize_split(cells, label_to_id, n_cls, token_to_idx, n_coding, top_k, modeltype, hvg_idx)
        n = length(cells)
        n == 0 && error("empty split")

        # feature dim
        if modeltype == "rtf"
            feat_dim = top_k
            X = Matrix{Int32}(undef, feat_dim, n)
        elseif !isnothing(hvg_idx)
            feat_dim = length(hvg_idx)
            X = Matrix{Float32}(undef, feat_dim, n)
        else
            feat_dim = n_coding
            X = Matrix{Float32}(undef, feat_dim, n)
        end

        # one-hot labels
        label_ids = [label_to_id[r[3]] for r in cells]
        y_oh = Flux.onehotbatch(label_ids, 1:n_cls)

        # group by shard
        by_shard = Dict{String, Vector{Tuple{Int, Int}}}()
        for (j, (sp, ci, _, _)) in enumerate(cells)
            push!(get!(by_shard, sp, Tuple{Int,Int}[]), (ci, j))
        end

        n_shards_done = 0
        for (sp, pairs) in by_shard
            shard = load_shard_pyarrow(sp)
            dense = Vector{Float32}(undef, n_coding)
            for (ci, j) in pairs
                if modeltype == "rtf"
                    gene_ids, _ = _process_cell_topk_flat(dense, shard.genes_flat, shard.offsets,
                                                           shard.expr_flat, ci, token_to_idx, n_coding, top_k)
                    X[:, j] = gene_ids
                elseif !isnothing(hvg_idx)
                    _cell_to_dense_flat!(dense, shard.genes_flat, shard.offsets,
                                         shard.expr_flat, ci, token_to_idx)
                    X[:, j] = dense[hvg_idx]
                else
                    _cell_to_dense_flat!(dense, shard.genes_flat, shard.offsets,
                                         shard.expr_flat, ci, token_to_idx)
                    X[:, j] = dense
                end
            end
            n_shards_done += 1
            if n_shards_done % 200 == 0
                println("  materialized $n_shards_done / $(length(by_shard)) shards")
                flush(stdout)
            end
        end
        return X, y_oh
    end

    X_train, y_train = _materialize_split(train_cells, label_to_id, n_cls, token_to_idx, n_coding, top_k, modeltype, hvg_idx)
    println("  train: $(size(X_train))")
    X_val, y_val = _materialize_split(val_cells, label_to_id, n_cls, token_to_idx, n_coding, top_k, modeltype, hvg_idx)
    println("  val: $(size(X_val))")
    X_test, y_test = _materialize_split(test_cells, label_to_id, n_cls, token_to_idx, n_coding, top_k, modeltype, hvg_idx)
    println("  test: $(size(X_test))")
    println("[pass 2] done")
    flush(stdout)

    n_genes = modeltype == "rtf" ? n_coding : size(X_train, 1)

    # oversampling (lvl2)
    cidx_dict, cs = if level == "lvl2"
        oversmpl_fn(y_train)
    else
        (nothing, nothing)
    end

    return (; X_train, X_val, X_test, y_train, y_val, y_test,
              n_genes, n_classifications=n_cls,
              train_idx=collect(1:size(X_train, 2)),
              val_idx=collect(1:size(X_val, 2)),
              test_idx=collect(1:size(X_test, 2)),
              cidx_dict, cs)
end


# sample_id -> dose match map
function _sc_sample_dose_map(meta_dir::String, dose::String)
    sample_dose_map = nothing
    if dose != ""
        if meta_dir == ""
            @warn "dose filter requested but no meta_dir provided; skipping SC dose filter"
        else
            sample_meta_path = joinpath(meta_dir, "sample_metadata.parquet")
            if isfile(sample_meta_path)
                println("[SC lvl3] loading sample_metadata.parquet for dose filtering...")
                t = _pq.read_table(sample_meta_path)
                cols = [string(c) for c in t.column_names]
                # dose column
                dose_col = nothing
                for candidate in ["dose", "pert_dose", "dose_um", "Dose"]
                    if candidate in cols
                        dose_col = candidate
                        break
                    end
                end
                # sample column
                sample_col = nothing
                for candidate in ["sample", "sample_id", "Sample"]
                    if candidate in cols
                        sample_col = candidate
                        break
                    end
                end
                # fallback: parse dose from drugname_drugconc
                drugconc_col = nothing
                if isnothing(dose_col) && "drugname_drugconc" in cols
                    drugconc_col = "drugname_drugconc"
                    println("[SC lvl3] no dedicated dose column; parsing dose from drugname_drugconc")
                end
                if !isnothing(sample_col) && (!isnothing(dose_col) || !isnothing(drugconc_col))
                    n_rows = convert(Int, t.num_rows)
                    sample_arr = t.column(sample_col)
                    dose_val = parse(Float64, dose)
                    sample_dose_map = Dict{String, Bool}()
                    if !isnothing(dose_col)
                        dose_arr = t.column(dose_col)
                        for i in 0:(n_rows - 1)
                            s = string(sample_arr[i].as_py())
                            d = dose_arr[i].as_py()
                            d_float = isa(d, Number) ? Float64(d) : tryparse(Float64, string(d))
                            if !isnothing(d_float)
                                sample_dose_map[s] = isapprox(d_float, dose_val; atol=0.01)
                            end
                        end
                    else
                        # parse float from "[('Drug', 0.05, 'uM')]"
                        dc_arr = t.column(drugconc_col)
                        dose_re = r",\s*([\d.]+)\s*,"
                        for i in 0:(n_rows - 1)
                            s = string(sample_arr[i].as_py())
                            dc_str = string(dc_arr[i].as_py())
                            m = match(dose_re, dc_str)
                            d_float = isnothing(m) ? nothing : tryparse(Float64, m.captures[1])
                            if !isnothing(d_float)
                                sample_dose_map[s] = isapprox(d_float, dose_val; atol=0.01)
                            end
                        end
                    end
                    n_matching = count(values(sample_dose_map))
                    println("[SC lvl3] dose map: $(length(sample_dose_map)) samples, $n_matching matching dose=$dose")
                else
                    @warn "could not find dose/sample columns in sample_metadata.parquet (cols=$cols)"
                end
            else
                @warn "sample_metadata.parquet not found at $sample_meta_path"
            end
        end
    end
    return sample_dose_map
end


# SC lvl3: pseudo-bulk per (cell_line, drug) + PCA
function _load_sc_lvl3(all_shards::Vector{String}, token_to_idx::Dict{Int,Int},
                        n_coding::Int, top_k::Int, modeltype::String,
                        source_cell::String, target_cell::String,
                        dose::String, meta_dir::String,
                        cell_to_dense_flat_fn, process_cell_topk_flat_fn,
                        regression_pairs_fn;
                        subset_shards::Int = 0,
                        hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                        identity_baseline_fn = nothing)

    shards_to_scan = subset_shards > 0 ? all_shards[1:min(subset_shards, length(all_shards))] : all_shards
    sample_dose_map = _sc_sample_dose_map(meta_dir, dose)

    # pass 1 metadata, source + target lines
    println("[SC lvl3 pass 1] scanning metadata from $(length(shards_to_scan)) shards...")
    flush(stdout)

    # (shard, cell_idx, drug, cell_line)
    cell_records = Tuple{String, Int, String, String}[]
    for (si, sp) in enumerate(shards_to_scan)
        meta = load_shard_metadata(sp)
        for i in 1:meta.n_cells
            cl = meta.cell_line_id[i]
            (cl != source_cell && cl != target_cell) && continue
            drug = meta.drug[i]
            drug == "DMSO" && continue

            # dose filter
            if !isnothing(sample_dose_map)
                sample_id = meta.sample[i]
                dose_ok = get(sample_dose_map, sample_id, false)
                !dose_ok && continue
            end

            push!(cell_records, (sp, i, drug, cl))
        end
        if si % 500 == 0
            println("  scanned $si / $(length(shards_to_scan)) shards, $(length(cell_records)) valid cells")
            flush(stdout)
        end
    end
    println("[SC lvl3 pass 1] done: $(length(cell_records)) valid cells")
    flush(stdout)

    # pass 2 pseudo-bulk
    println("[SC lvl3 pass 2] pseudo-bulking by (cell_line, drug)...")
    flush(stdout)

    # running sums per (cell_line, drug)
    pb_sums = Dict{Tuple{String,String}, Vector{Float64}}()
    pb_counts = Dict{Tuple{String,String}, Int}()

    # group by shard
    by_shard = Dict{String, Vector{Tuple{Int, String, String}}}()
    for (sp, ci, drug, cl) in cell_records
        push!(get!(by_shard, sp, Tuple{Int,String,String}[]), (ci, drug, cl))
    end

    dense = Vector{Float32}(undef, n_coding)
    n_shards_done = 0
    for (sp, pairs) in by_shard
        shard = load_shard_pyarrow(sp)
        for (ci, drug, cl) in pairs
            cell_to_dense_flat_fn(dense, shard.genes_flat, shard.offsets,
                                  shard.expr_flat, ci, token_to_idx)
            key = (cl, drug)
            if !haskey(pb_sums, key)
                pb_sums[key] = zeros(Float64, n_coding)
                pb_counts[key] = 0
            end
            pb_sums[key] .+= Float64.(dense)
            pb_counts[key] += 1
        end
        n_shards_done += 1
        if n_shards_done % 200 == 0
            println("  pseudo-bulked $n_shards_done / $(length(by_shard)) shards")
            flush(stdout)
        end
    end
    println("[SC lvl3 pass 2] done: $(length(pb_sums)) pseudo-bulk groups")

    # pseudo-bulk matrix
    groups = collect(keys(pb_sums))
    n_groups = length(groups)
    pb_expr = Matrix{Float32}(undef, n_coding, n_groups)
    pb_cl = String[]
    pb_drug = String[]
    for (j, key) in enumerate(groups)
        pb_expr[:, j] = Float32.(pb_sums[key] ./ pb_counts[key])
        push!(pb_cl, key[1])
        push!(pb_drug, key[2])
    end
    println("  pseudo-bulk matrix: $(size(pb_expr))")
    println("  source groups: $(count(pb_cl .== source_cell)), target groups: $(count(pb_cl .== target_cell))")

    # PCA pairing
    src_mask = BitVector(pb_cl .== source_cell)
    tgt_mask = BitVector(pb_cl .== target_cell)
    pb_drug_sym = Symbol.(pb_drug)

    X, y, shared_perts, pca_model, split = regression_pairs_fn(pb_expr, src_mask, tgt_mask,
                                                                pb_drug_sym, pb_drug_sym)

    n_genes = size(X, 1)
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx

    # identity baseline on raw expression
    id_baseline = isnothing(identity_baseline_fn) ? nothing :
        identity_baseline_fn(X[:, test_idx], y[:, test_idx], pca_model)

    # rank if RTF
    if modeltype == "rtf"
        # train-shard medians
        gene_medians = something(sc_gene_medians(n_coding), ones(Float32, size(X, 1)))
        n, m = size(X)
        k = min(top_k, n)
        X_ranked = Matrix{Int32}(undef, k, m)
        normalized_col = Vector{Float32}(undef, n)
        sorted_ind_col = Vector{Int32}(undef, n)
        pad = pad_token_id(n_coding)
        for j in 1:m
            @. normalized_col = X[:, j] / gene_medians
            sortperm!(sorted_ind_col, normalized_col, rev=true)
            X_ranked[:, j] .= view(sorted_ind_col, 1:k)
            # undetected -> PAD
            n_det_j = count(>(0f0), view(X, :, j))
            n_det_j < k && (X_ranked[max(n_det_j, 1)+1:k, j] .= pad)
        end
        X = X_ranked
    end

    # HVG filter for ETF
    if modeltype != "rtf" && !isnothing(hvg_idx)
        X = X[hvg_idx, :]
        n_genes = size(X, 1)
    end

    # split

    X_train = X[:, train_idx]
    X_val   = X[:, val_idx]
    X_test  = X[:, test_idx]
    y_train = y[:, train_idx]
    y_val   = y[:, val_idx]
    y_test  = y[:, test_idx]

    println("SC lvl3 split: train=$(size(X_train,2)), val=$(size(X_val,2)), test=$(size(X_test,2))")

    return (; X_train, X_val, X_test, y_train, y_val, y_test,
              n_genes=modeltype == "rtf" ? n_coding : n_genes,
              n_classifications=1,
              train_idx, val_idx, test_idx,
              cidx_dict=nothing, cs=nothing, id_baseline)
end


# per-cell SC lvl3 with PB PC1 targets
function _load_sc_lvl3_percell(all_shards::Vector{String}, token_to_idx::Dict{Int,Int},
                                n_coding::Int, top_k::Int, modeltype::String,
                                source_cell::String, target_cell::String,
                                dose::String, meta_dir::String,
                                cell_to_dense_flat_fn, process_cell_topk_flat_fn;
                                pb_expr::Matrix{Float32},
                                pb_df::DataFrame,
                                regression_pairs_fn,
                                subset_shards::Int = 0,
                                hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                                identity_baseline_fn = nothing)

    println("[SC lvl3 percell] per-cell mode: individual source cells → PB PC1 targets")
    flush(stdout)

    shards_to_scan = subset_shards > 0 ? all_shards[1:min(subset_shards, length(all_shards))] : all_shards

    # step 1 PB PC1 targets
    println("[SC lvl3 percell] computing PC1 targets from PB data...")
    src_sym = Symbol(source_cell)
    tgt_sym = Symbol(target_cell)
    src_mask = BitVector(pb_df.cell_line .== src_sym)
    tgt_mask = BitVector(pb_df.cell_line .== tgt_sym)

    # dose filter
    if dose != ""
        dose_sym = Symbol("$(dose) uM")
        dose_mask = BitVector(pb_df.dose .== dose_sym)
        if sum(dose_mask) == 0
            dose_sym = Symbol(dose)
            dose_mask = BitVector(pb_df.dose .== dose_sym)
        end
        src_mask .&= dose_mask
        tgt_mask .&= dose_mask
        println("  PB dose filter: $(dose_sym) → source=$(sum(src_mask)), target=$(sum(tgt_mask)) samples")
    end

    println("  PB source $(source_cell): $(sum(src_mask)) samples")
    println("  PB target $(target_cell): $(sum(tgt_mask)) samples")

    _, y_pb, shared_perts, pca_model, pca_split = regression_pairs_fn(pb_expr, src_mask, tgt_mask,
                                                                       pb_df.drug, pb_df.drug)
    # compound split shared with PCA
    train_drugs = Set(string.(shared_perts[pca_split.train_idx]))
    val_drugs   = Set(string.(shared_perts[pca_split.val_idx]))
    test_drugs  = Set(string.(shared_perts[pca_split.test_idx]))

    # gene spaces must match for identity baseline
    do_id_baseline = !isnothing(identity_baseline_fn) && size(pb_expr, 1) == n_coding
    if !isnothing(identity_baseline_fn) && !do_id_baseline
        @warn "PB genes ($(size(pb_expr, 1))) ≠ SC coding genes ($n_coding); identity baseline skipped"
    end

    # drug -> PC1
    drug_to_pc1 = Dict{String, Float32}()
    for (i, pert) in enumerate(shared_perts)
        drug_to_pc1[string(pert)] = y_pb[1, i]
    end
    shared_drug_set = Set(string.(shared_perts))
    println("  PC1 targets for $(length(drug_to_pc1)) shared compounds")
    flush(stdout)

    # source cells at PB dose
    sample_dose_map = _sc_sample_dose_map(meta_dir, dose)
    if dose != "" && isnothing(sample_dose_map)
        @warn "[SC lvl3 percell] dose=$dose requested but no dose map available; SC cells use all doses"
    end

    # step 3 scan SC shards
    println("[SC lvl3 percell] scanning shards for source cells ($(source_cell))...")
    flush(stdout)

    # (shard, cell_idx, drug)
    cell_records = Tuple{String, Int, String}[]
    for (si, sp) in enumerate(shards_to_scan)
        meta = load_shard_metadata(sp)
        for i in 1:meta.n_cells
            cl = meta.cell_line_id[i]
            cl != source_cell && continue
            drug = meta.drug[i]
            !(drug in shared_drug_set) && continue

            # dose filter
            if !isnothing(sample_dose_map)
                sample_id = meta.sample[i]
                dose_ok = get(sample_dose_map, sample_id, false)
                !dose_ok && continue
            end

            push!(cell_records, (sp, i, drug))
        end
        if si % 500 == 0
            println("  scanned $si / $(length(shards_to_scan)) shards, $(length(cell_records)) source cells")
            flush(stdout)
        end
    end
    n_cells = length(cell_records)
    println("[SC lvl3 percell] found $n_cells source cells across $(length(unique(r[3] for r in cell_records))) compounds")
    flush(stdout)
    n_cells < 5 && error("too few source cells ($n_cells) for per-cell SC lvl3")

    # step 4 materialize
    println("[SC lvl3 percell] materializing $n_cells cells...")
    flush(stdout)

    y = Matrix{Float32}(undef, 1, n_cells)
    cell_drugs = Vector{String}(undef, n_cells)

    # raw test expression for identity baseline
    test_cols = [col for (col, r) in enumerate(cell_records) if r[3] in test_drugs]
    test_pos = Dict(col => j for (j, col) in enumerate(test_cols))
    X_id = do_id_baseline ? Matrix{Float32}(undef, n_coding, length(test_cols)) : nothing

    # group by shard
    by_shard = Dict{String, Vector{Tuple{Int, Int, String}}}()
    for (col, (sp, ci, drug)) in enumerate(cell_records)
        push!(get!(by_shard, sp, Tuple{Int,Int,String}[]), (ci, col, drug))
    end

    # RTF: (top_k, n) Int32, others: (n_genes, n) Float32
    n_genes = n_coding
    if modeltype == "rtf" && !isnothing(process_cell_topk_flat_fn)
        # RTF: top-k gene ids
        X = Matrix{Int32}(undef, top_k, n_cells)
        dense = Vector{Float32}(undef, n_coding)
        n_shards_done = 0
        for (sp, entries) in by_shard
            shard = load_shard_pyarrow(sp)
            for (ci, col, drug) in entries
                gene_ids, _ = process_cell_topk_flat_fn(dense, shard.genes_flat, shard.offsets,
                                                         shard.expr_flat, ci, token_to_idx, n_coding, top_k)
                X[:, col] = gene_ids
                y[1, col] = drug_to_pc1[drug]
                cell_drugs[col] = drug
                # raw expression + tie noise
                if do_id_baseline && haskey(test_pos, col)
                    X_id[:, test_pos[col]] = dense
                end
            end
            n_shards_done += 1
            if n_shards_done % 200 == 0; println("  loaded $n_shards_done / $(length(by_shard)) shards"); flush(stdout); end
        end
        n_genes = n_coding
    else
        # MLP/ETF: dense
        X = Matrix{Float32}(undef, n_coding, n_cells)
        dense = Vector{Float32}(undef, n_coding)
        n_shards_done = 0
        for (sp, entries) in by_shard
            shard = load_shard_pyarrow(sp)
            for (ci, col, drug) in entries
                if modeltype in ("rmlp", "rlog") && !isnothing(process_cell_topk_flat_fn)
                    # rank features
                    gene_ids, _, n_det = process_cell_topk_flat_fn(dense, shard.genes_flat, shard.offsets,
                                                                   shard.expr_flat, ci, token_to_idx, n_coding, top_k)
                    rank_features!(view(X, :, col), gene_ids, n_det, top_k)
                else
                    cell_to_dense_flat_fn(dense, shard.genes_flat, shard.offsets,
                                          shard.expr_flat, ci, token_to_idx)
                    X[:, col] = dense
                end
                y[1, col] = drug_to_pc1[drug]
                cell_drugs[col] = drug
                if do_id_baseline && haskey(test_pos, col)
                    X_id[:, test_pos[col]] = dense
                end
            end
            n_shards_done += 1
            if n_shards_done % 200 == 0; println("  loaded $n_shards_done / $(length(by_shard)) shards"); flush(stdout); end
        end

        # step 5 model transforms
        if modeltype in ("rmlp", "rlog") && !isnothing(process_cell_topk_flat_fn)
            nothing
        elseif !isnothing(hvg_idx)
            X = X[hvg_idx, :]
            n_genes = size(X, 1)
        end
    end
    println("[SC lvl3 percell] materialized: $(size(X))")

    # step 6 compound split
    present = Set(cell_drugs)
    test_compounds  = intersect(test_drugs, present)
    val_compounds   = intersect(val_drugs, present)
    train_compounds = intersect(train_drugs, present)

    train_idx = Int[]; val_idx = Int[]; test_idx = Int[]
    for i in 1:n_cells
        drug = cell_drugs[i]
        if drug in test_compounds
            push!(test_idx, i)
        elseif drug in val_compounds
            push!(val_idx, i)
        else
            push!(train_idx, i)
        end
    end

    X_train = X[:, train_idx]; y_train = y[:, train_idx]
    X_val   = X[:, val_idx];   y_val   = y[:, val_idx]
    X_test  = X[:, test_idx];  y_test  = y[:, test_idx]

    println("[SC lvl3 percell] compound-level split:")
    println("  train: $(length(train_idx)) cells, $(length(train_compounds)) compounds")
    println("  val:   $(length(val_idx)) cells, $(length(val_compounds)) compounds")
    println("  test:  $(length(test_idx)) cells, $(length(test_compounds)) compounds")

    # X_id columns align with y_test
    @assert test_idx == test_cols
    id_baseline = do_id_baseline ? identity_baseline_fn(X_id, y_test, pca_model) :
        (isnothing(identity_baseline_fn) ? nothing : (; r2=NaN, pearson=NaN, rmse=NaN))

    return (; X_train, X_val, X_test, y_train, y_val, y_test,
              n_genes=n_genes,
              n_classifications=1,
              train_idx, val_idx, test_idx,
              cidx_dict=nothing, cs=nothing, id_baseline)
end


# SC finetune metadata scan + sample split
function sc_finetune_metadata_scan(all_shards::Vector{String}, level::String;
                                    pb_data_path::String = "",
                                    subset_shards::Int = 0,
                                    split_by::String = "drug_dose",
                                    split_seed::Integer = 42)  # fixed split for every run/model; --seed only affects training (as PB split_seed)

    # valid labels
    valid_drugs = nothing
    if level == "lvl2"
        pb_data_path == "" && error("sc_finetune_metadata_scan: pb_data_path required for lvl2")
        pb_df = JLD2.load(pb_data_path)["df"]
        pb_drugs = String.(pb_df.drug)
        non_dmso = filter(d -> d != "DMSO", pb_drugs)
        counts = countmap(non_dmso)
        valid_drugs = Set(k for (k, v) in counts if v >= 100)
        println("lvl2: $(length(valid_drugs)) valid drugs from PB data (≥100 PB samples, non-DMSO)")
    end

    # pass 1 metadata
    # seeded random subset (+1: not the pretrain shard split)
    shards_to_scan = subset_shards > 0 ?
        sort(shuffle(MersenneTwister(split_seed + 1), sort(all_shards))[1:min(subset_shards, length(all_shards))]) : all_shards
    println("[pass 1] scanning metadata from $(length(shards_to_scan)) shards...")
    flush(stdout)

    # (shard, cell_idx, label, sample_id, cell_line). strings interned: one object per unique value instead of one
    # per cell (93M cells -> ~10G of duplicate Strings otherwise)
    cell_records = Tuple{String, Int, String, String, String}[]
    str_pool = Dict{String, String}()
    intern(x) = get!(str_pool, x, x)
    for (si, sp) in enumerate(shards_to_scan)
        meta = load_shard_metadata(sp)
        for i in 1:meta.n_cells
            if level == "lvl1"
                label = meta.cell_line_id[i]
            else
                label = meta.drug[i]
                (label == "DMSO" || !(label in valid_drugs)) && continue
            end
            push!(cell_records, (sp, i, intern(label), intern(meta.sample[i]), intern(meta.cell_line_id[i])))
        end
        if si % 500 == 0
            println("  scanned $si / $(length(shards_to_scan)) shards, $(length(cell_records)) valid cells so far")
            flush(stdout)
        end
    end
    println("[pass 1] done: $(length(cell_records)) valid cells from $(length(shards_to_scan)) shards")
    flush(stdout)
    memlog("after metadata scan")

    # split unit: "drug_dose" (default) = all wells of a (drug, dose) together, so replicate wells can't straddle
    # train/test (DMSO kept per well); "well" = whole wells; "well_cl" = (well, cell line) units, like a random PB split
    split_by in ("drug_dose", "well", "well_cl") || error("split_by must be drug_dose, well or well_cl (got $split_by)")
    dd = Dict{String, String}()
    if split_by == "drug_dose"
        pb_data_path == "" && error("split_by=drug_dose needs pb_data_path (sample -> drug, dose map)")
        pbm = JLD2.load(pb_data_path)["df"]
        for (smp, drg, dse) in zip(String.(pbm.sample), String.(pbm.drug), String.(pbm.dose))
            dd[smp] = drg == "DMSO" ? smp : drg * "|" * dse
        end
        n_miss = length(setdiff(Set(r[4] for r in cell_records), keys(dd)))
        n_miss > 0 && println("  drug_dose split: $n_miss SC samples not in PB map -> grouped by well")
    end
    unit(r) = split_by == "well_cl" ? r[4] * "|" * r[5] : split_by == "drug_dose" ? get(dd, r[4], r[4]) : r[4]
    unique_samples = unique(unit(r) for r in cell_records)
    # shuffle!(unique_samples)  # global RNG: split depended on --seed and on RNG calls made before this point
    shuffle!(MersenneTwister(split_seed), unique_samples)
    n_test = floor(Int, length(unique_samples) * 0.1)
    n_val  = floor(Int, length(unique_samples) * 0.1)
    test_samples  = Set(unique_samples[1:n_test])
    val_samples   = Set(unique_samples[n_test+1:n_test+n_val])
    train_samples = Set(unique_samples[n_test+n_val+1:end])
    println("$(split_by) split: $(length(train_samples)) train, $(length(val_samples)) val, $(length(test_samples)) test units")

    # process labels (before the full record list is dropped)
    unique_labels = sort(collect(Set(r[3] for r in cell_records)))
    label_to_id = Dict(l => i for (i, l) in enumerate(unique_labels))
    n_cls = length(unique_labels)

    train_cells = filter(r -> unit(r) in train_samples, cell_records)
    val_cells   = filter(r -> unit(r) in val_samples, cell_records)
    test_cells  = filter(r -> unit(r) in test_samples, cell_records)
    # warn on val/test classes with no training cells
    train_labels = Set(r[3] for r in train_cells)
    missing_tr = setdiff(Set(r[3] for r in Iterators.flatten((val_cells, test_cells))), train_labels)
    isempty(missing_tr) || @warn "$(length(missing_tr)) val/test classes have no training cells: $(first(collect(missing_tr), 5))"
    println("classes: $(length(train_labels)) in train, $(length(missing_tr)) val/test-only")
    empty!(cell_records); sizehint!(cell_records, 0); cell_records = nothing
    GC.gc()
    println("cell split: $(length(train_cells)) train, $(length(val_cells)) val, $(length(test_cells)) test cells")
    flush(stdout)

    println("n_classifications: $n_cls")
    memlog("after split")

    return (; train_cells, val_cells, test_cells, label_to_id, n_cls, valid_drugs)
end


# materialize cells into features + one-hot labels
function materialize_finetune_split(cells, label_to_id::Dict, n_cls::Int,
                                     token_to_idx::Dict{Int,Int}, n_coding::Int,
                                     top_k::Int, modeltype::String,
                                     hvg_idx::Union{Vector{Int}, Nothing};
                                     process_cell_topk_flat_fn = nothing,
                                     cell_to_dense_flat_fn = nothing)
    n = length(cells)
    n == 0 && error("materialize_finetune_split: empty split")

    # feature dim
    if modeltype == "rtf"
        feat_dim = top_k
        X = Matrix{Int32}(undef, feat_dim, n)
    elseif !isnothing(hvg_idx)
        feat_dim = length(hvg_idx)
        X = Matrix{Float32}(undef, feat_dim, n)
    else
        feat_dim = n_coding
        X = Matrix{Float32}(undef, feat_dim, n)
    end

    # one-hot labels
    label_ids = [label_to_id[r[3]] for r in cells]
    y_oh = Flux.onehotbatch(label_ids, 1:n_cls)

    # group by shard
    by_shard = Dict{String, Vector{Tuple{Int, Int}}}()
    for (j, (sp, ci, _, _)) in enumerate(cells)
        push!(get!(by_shard, sp, Tuple{Int,Int}[]), (ci, j))
    end

    n_shards_done = 0
    for (sp, pairs) in by_shard
        shard = load_shard_pyarrow(sp)
        dense = Vector{Float32}(undef, n_coding)
        for (ci, j) in pairs
            if modeltype == "rtf"
                gene_ids, _ = process_cell_topk_flat_fn(dense, shard.genes_flat, shard.offsets,
                                                         shard.expr_flat, ci, token_to_idx, n_coding, top_k)
                X[:, j] = gene_ids
            elseif !isnothing(hvg_idx)
                cell_to_dense_flat_fn(dense, shard.genes_flat, shard.offsets,
                                       shard.expr_flat, ci, token_to_idx)
                _apply_delta!(dense, shard, ci)   # no-op unless --input delta
                X[:, j] = dense[hvg_idx]
            else
                cell_to_dense_flat_fn(dense, shard.genes_flat, shard.offsets,
                                       shard.expr_flat, ci, token_to_idx)
                _apply_delta!(dense, shard, ci)   # no-op unless --input delta
                X[:, j] = dense
            end
        end
        n_shards_done += 1
        if n_shards_done % 200 == 0
            println("  materialized $n_shards_done / $(length(by_shard)) shards")
            flush(stdout)
        end
    end
    return X, y_oh
end


# shard -> (cell idx, labels) for train
function prepare_shard_cell_map(train_cells, label_to_id::Dict)
    shard_map = Dict{String, Tuple{Vector{Int}, Vector{Int}}}()
    for (sp, ci, label, _) in train_cells
        lid = label_to_id[label]
        if !haskey(shard_map, sp)
            shard_map[sp] = (Int[], Int[])
        end
        push!(shard_map[sp][1], ci)
        push!(shard_map[sp][2], lid)
    end
    return shard_map
end


# batch channel for one finetune shard
function finetune_batches_from_shard(shard_path::String,
                                      cell_indices::Vector{Int},
                                      cell_labels::Vector{Int},
                                      token_to_idx::Dict{Int,Int}, n_coding::Int,
                                      top_k::Int, batch_size::Int, modeltype::String,
                                      n_cls::Int;
                                      hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                                      process_cell_topk_flat_fn = nothing,
                                      cell_to_dense_flat_fn = nothing)

    shard = load_shard_pyarrow(shard_path)
    n_valid = length(cell_indices)
    n_valid == 0 && return Channel{Any}(0)

    return Channel{Any}(1) do ch
        for start_idx in 1:batch_size:n_valid
            end_idx = min(start_idx + batch_size - 1, n_valid)
            ci = cell_indices[start_idx:end_idx]
            labels = cell_labels[start_idx:end_idx]
            X_batch = _build_ft_batch(shard, ci, token_to_idx, n_coding,
                                       top_k, modeltype, hvg_idx,
                                       process_cell_topk_flat_fn, cell_to_dense_flat_fn)
            y_batch = Flux.onehotbatch(labels, 1:n_cls)
            put!(ch, (X_batch, y_batch))
        end
    end
end


# feature matrix for a batch
function _build_ft_batch(shard, cell_indices::AbstractVector{Int},
                          token_to_idx::Dict{Int,Int}, n_coding::Int,
                          top_k::Int, modeltype::String,
                          hvg_idx::Union{Vector{Int}, Nothing},
                          process_cell_topk_flat_fn, cell_to_dense_flat_fn)
    bs = length(cell_indices)
    if modeltype == "rankfeat"
        # rank baselines: (n_coding, bs) Float32
        batch = Matrix{Float32}(undef, n_coding, bs)
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            gene_ids, _, n_det = process_cell_topk_flat_fn(dense, shard.genes_flat, shard.offsets,
                                                           shard.expr_flat, ci, token_to_idx, n_coding, top_k)
            rank_features!(view(batch, :, j), gene_ids, n_det, top_k)
        end
        return batch
    elseif modeltype == "rtf"
        # RTF: (top_k, bs) Int32
        batch = Matrix{Int32}(undef, top_k, bs)
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            gene_ids, _ = process_cell_topk_flat_fn(dense, shard.genes_flat, shard.offsets,
                                                     shard.expr_flat, ci, token_to_idx, n_coding, top_k)
            batch[:, j] = gene_ids
        end
        return batch
    elseif !isnothing(hvg_idx)
        # ETF-HVG: (n_hvg, bs) Float32
        n_hvg = length(hvg_idx)
        batch = Matrix{Float32}(undef, n_hvg, bs)
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            cell_to_dense_flat_fn(dense, shard.genes_flat, shard.offsets,
                                   shard.expr_flat, ci, token_to_idx)
            _apply_delta!(dense, shard, ci)   # no-op unless --input delta
            batch[:, j] = dense[hvg_idx]
        end
        return batch
    else
        # ETF: (n_coding, bs) Float32
        batch = Matrix{Float32}(undef, n_coding, bs)
        dense = Vector{Float32}(undef, n_coding)
        for (j, ci) in enumerate(cell_indices)
            cell_to_dense_flat_fn(dense, shard.genes_flat, shard.offsets,
                                   shard.expr_flat, ci, token_to_idx)
            _apply_delta!(dense, shard, ci)   # no-op unless --input delta
            batch[:, j] = copy(dense)
        end
        return batch
    end
end


# memory checkpoint: julia live heap, peak RSS, arrow (python) allocations
function memlog(tag::AbstractString)
    arrow = try Float64(pyimport("pyarrow").total_allocated_bytes()) / 2^30 catch; NaN end
    println("[mem] $tag: julia live=$(round(Base.gc_live_bytes() / 2^30, digits=1))G, " *
            "maxrss=$(round(Sys.maxrss() / 2^30, digits=1))G, arrow=$(round(arrow, digits=1))G")
    flush(stdout)
end

# training cells drawn across all shards (class-balanced for lvl2), shuffled in pools of group_shards shards.
# shard-by-shard training gave label-skewed batches: each shard holds ~30-70 of 376 drugs (lvl2).
function draw_train_cells(train_cells, label_to_id::Dict, n_cells::Int; balanced::Bool = false)
    n_train = length(train_cells)
    labels = [label_to_id[r[3]] for r in train_cells]
    if !balanced
        # uniform, without replacement while possible
        n_cells <= n_train && return randperm(n_train)[1:n_cells]
        return vcat(randperm(n_train), rand(1:n_train, n_cells - n_train))
    end
    # equal count per class, with replacement only for classes smaller than their quota
    by_cls = Dict{Int, Vector{Int}}()
    for (j, l) in enumerate(labels)
        push!(get!(by_cls, l, Int[]), j)
    end
    cls = sort(collect(keys(by_cls)))
    per_cls = fill(div(n_cells, length(cls)), length(cls))
    per_cls[randperm(length(cls))[1:rem(n_cells, length(cls))]] .+= 1
    picked = Int[]
    for (c, q) in zip(cls, per_cls)
        idx = by_cls[c]
        append!(picked, q <= length(idx) ? shuffle(idx)[1:q] : rand(idx, q))
    end
    return picked
end

function finetune_train_batches(train_cells, label_to_id::Dict, n_cls::Int, n_cells::Int,
                                token_to_idx::Dict{Int,Int}, n_coding::Int,
                                top_k::Int, batch_size::Int, modeltype::String;
                                balanced::Bool = false,
                                group_shards::Int = 64,
                                hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                                process_cell_topk_flat_fn = nothing,
                                cell_to_dense_flat_fn = nothing)
    picked = draw_train_cells(train_cells, label_to_id, n_cells; balanced=balanced)

    # shard -> (cell idx, label) of the drawn cells
    by_shard = Dict{String, Tuple{Vector{Int}, Vector{Int}}}()
    for j in picked
        sp, ci, lab, _ = train_cells[j]
        e = get!(by_shard, sp, (Int[], Int[]))
        push!(e[1], ci); push!(e[2], label_to_id[lab])
    end
    shard_order = shuffle(sort(collect(keys(by_shard))))
    groups = collect(Iterators.partition(shard_order, max(group_shards, 1)))
    println("[train mix] $(length(picked)) cells from $(length(shard_order)) shards, $(length(groups)) groups of ≤$group_shards shards (balanced=$balanced)")
    flush(stdout)

    feat_dim = modeltype == "rtf" ? top_k : (!isnothing(hvg_idx) ? length(hvg_idx) : n_coding)
    T = modeltype == "rtf" ? Int32 : Float32
    return Channel{Any}(2) do ch
        carry_X = Matrix{T}(undef, feat_dim, 0); carry_y = Int[]
        for (gi, grp) in enumerate(groups)
            # pool features for the group, preallocated and filled in place
            n_new = sum(length(by_shard[sp][1]) for sp in grp)
            n_carry = length(carry_y)
            X = Matrix{T}(undef, feat_dim, n_carry + n_new)
            X[:, 1:n_carry] .= carry_X
            ys = vcat(carry_y, Vector{Int}(undef, n_new))
            col = n_carry
            for sp in grp
                cis, labs = by_shard[sp]
                shard = load_shard_pyarrow(sp)
                Xs = _build_ft_batch(shard, cis, token_to_idx, n_coding, top_k, modeltype, hvg_idx,
                                     process_cell_topk_flat_fn, cell_to_dense_flat_fn)
                shard = nothing
                X[:, col+1:col+length(cis)] .= Xs
                ys[col+1:col+length(cis)] .= labs
                col += length(cis)
            end
            carry_X = Matrix{T}(undef, feat_dim, 0)
            (gi == 1 || gi % 10 == 0) && (GC.gc(false); memlog("train pool $gi/$(length(groups)) ($(length(ys)) cells)"))
            perm = randperm(length(ys))
            n_full = div(length(ys), batch_size) * batch_size
            # label-mixing check on the first pool
            if gi == 1 && n_full > 0
                nb = min(50, div(n_full, batch_size))
                u = sum(length(unique(ys[perm[(b-1)*batch_size+1:b*batch_size]])) for b in 1:nb) / nb
                println("[train mix] first pool: $(length(ys)) cells, mean unique labels per batch = $(round(u, digits=1)) / $batch_size")
                flush(stdout)
            end
            for s in 1:batch_size:n_full
                idx = perm[s:s+batch_size-1]
                put!(ch, (X[:, idx], Flux.onehotbatch(ys[idx], 1:n_cls)))
            end
            # leftover cells roll into the next pool; last group emits a partial batch
            rest = perm[n_full+1:end]
            if gi == length(groups)
                isempty(rest) || put!(ch, (X[:, rest], Flux.onehotbatch(ys[rest], 1:n_cls)))
            else
                carry_X = X[:, rest]; carry_y = ys[rest]
            end
            X = nothing
        end
    end
end


# streaming SC finetune data (shard maps)
function load_sc_finetune_data_streaming(all_shards::Vector{String}, level::String,
                                          token_to_idx::Dict{Int,Int}, n_coding::Int,
                                          top_k::Int, modeltype::String;
                                          pb_data_path::String = "",
                                          hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                                          subset_shards::Int = 0,
                                          process_cell_topk_flat_fn = nothing,
                                          cell_to_dense_flat_fn = nothing,
                                          oversmpl_fn = nothing,
                                          source_cell::String = "",
                                          target_cell::String = "",
                                          dose::String = "",
                                          meta_dir::String = "",
                                          regression_pairs_fn = nothing,
                                          sc_lvl3_percell::Bool = false,
                                          pb_expr::Union{Matrix{Float32}, Nothing} = nothing,
                                          pb_df::Union{DataFrame, Nothing} = nothing,
                                          actual_modeltype::String = "",
                                          identity_baseline_fn = nothing,
                                          split_by::String = "drug_dose")

    # validate args
    if modeltype == "rtf" && isnothing(process_cell_topk_flat_fn)
        error("load_sc_finetune_data_streaming: process_cell_topk_flat_fn required for rtf modeltype")
    end
    if modeltype != "rtf" && isnothing(cell_to_dense_flat_fn)
        error("load_sc_finetune_data_streaming: cell_to_dense_flat_fn required for etf/mlp modeltype")
    end

    # lvl3 (no streaming)
    if level == "lvl3"
        (source_cell == "" || target_cell == "") && error("load_sc_finetune_data_streaming lvl3: source_cell and target_cell required")
        isnothing(regression_pairs_fn) && error("load_sc_finetune_data_streaming lvl3: regression_pairs_fn required")
        isnothing(cell_to_dense_flat_fn) && error("load_sc_finetune_data_streaming lvl3: cell_to_dense_flat_fn required")
        if sc_lvl3_percell
            # per-cell lvl3
            isnothing(pb_expr) && error("load_sc_finetune_data_streaming lvl3 percell: pb_expr required")
            isnothing(pb_df) && error("load_sc_finetune_data_streaming lvl3 percell: pb_df required")
            percell_mt = actual_modeltype != "" ? actual_modeltype : modeltype
            d = _load_sc_lvl3_percell(all_shards, token_to_idx, n_coding, top_k, percell_mt,
                                       source_cell, target_cell, dose, meta_dir,
                                       cell_to_dense_flat_fn, process_cell_topk_flat_fn;
                                       pb_expr=pb_expr, pb_df=pb_df,
                                       regression_pairs_fn=regression_pairs_fn,
                                       subset_shards=subset_shards, hvg_idx=hvg_idx,
                                       identity_baseline_fn=identity_baseline_fn)
        else
            # pseudo-bulk lvl3
            d = _load_sc_lvl3(all_shards, token_to_idx, n_coding, top_k, modeltype,
                               source_cell, target_cell, dose, meta_dir,
                               cell_to_dense_flat_fn, process_cell_topk_flat_fn,
                               regression_pairs_fn;
                               subset_shards=subset_shards, hvg_idx=hvg_idx,
                               identity_baseline_fn=identity_baseline_fn)
        end
        # streaming-compatible shape
        return (; X_train=d.X_train, X_val=d.X_val, X_test=d.X_test,
                  y_train=d.y_train, y_val=d.y_val, y_test=d.y_test,
                  n_genes=d.n_genes, n_classifications=d.n_classifications,
                  label_to_id=nothing,
                  train_shard_map=nothing, train_shard_paths=nothing,
                  n_train_cells=size(d.X_train, 2),
                  val_shard_map=nothing, val_shard_paths=nothing,
                  n_val_cells=size(d.X_val, 2),
                  test_shard_map=nothing, test_shard_paths=nothing,
                  n_test_cells=size(d.X_test, 2),
                  use_oversmpl=false,
                  cidx_dict=nothing, cs=nothing,
                  train_cells=nothing, test_group_map=nothing,
                  train_idx=d.train_idx, val_idx=d.val_idx, test_idx=d.test_idx,
                  id_baseline=d.id_baseline)
    end

    # pass 1 metadata + split
    scan = sc_finetune_metadata_scan(all_shards, level;
                                      pb_data_path=pb_data_path,
                                      subset_shards=subset_shards,
                                      split_by=split_by)
    # shard maps
    train_shard_map = prepare_shard_cell_map(scan.train_cells, scan.label_to_id)
    train_shard_paths = collect(keys(train_shard_map))
    n_train_cells = length(scan.train_cells)

    val_shard_map = prepare_shard_cell_map(scan.val_cells, scan.label_to_id)
    val_shard_paths = collect(keys(val_shard_map))
    n_val_cells = length(scan.val_cells)

    test_shard_map = prepare_shard_cell_map(scan.test_cells, scan.label_to_id)
    test_shard_paths = collect(keys(test_shard_map))
    # per-cell group keys aligned with test_shard_map (same record order): well, (well, cell line)
    test_group_map = Dict{String, Tuple{Vector{String}, Vector{String}}}()
    wcl_pool = Dict{String, String}()
    for (sp, _, _, smp, cl) in scan.test_cells
        g = get!(test_group_map, sp, (String[], String[]))
        k = smp * "|" * cl
        push!(g[1], smp); push!(g[2], get!(wcl_pool, k, k))
    end
    memlog("after shard maps")
    n_test_cells = length(scan.test_cells)

    println("[streaming] train: $n_train_cells cells across $(length(train_shard_paths)) shards")
    println("[streaming] val: $n_val_cells cells across $(length(val_shard_paths)) shards")
    println("[streaming] test: $n_test_cells cells across $(length(test_shard_paths)) shards")
    println("[streaming] all splits use shard-level streaming (no materialization)")
    flush(stdout)

    n_genes = modeltype == "rtf" ? n_coding : (!isnothing(hvg_idx) ? length(hvg_idx) : n_coding)
    _use_oversmpl = level == "lvl2"

    return (; n_genes, n_classifications=scan.n_cls, label_to_id=scan.label_to_id,
              train_shard_map, train_shard_paths, n_train_cells,
              val_shard_map, val_shard_paths, n_val_cells,
              test_shard_map, test_shard_paths, n_test_cells,
              use_oversmpl=_use_oversmpl,
              cidx_dict=nothing, cs=nothing,
              train_cells=scan.train_cells,
              test_group_map,
              train_idx=collect(1:n_train_cells),
              val_idx=collect(1:n_val_cells),
              test_idx=collect(1:n_test_cells),
              id_baseline=nothing)
end


# seeded shard split
function split_shards(all_shards::Vector{String}, subset::Int = 0)
    train, val, test = shard_train_val_test_split(all_shards, 0.1, 0.1)
    if subset > 0
        n_eval = max(1, div(subset, 8))
        train = train[1:min(subset, length(train))]
        val = val[1:min(n_eval, length(val))]
        test = test[1:min(n_eval, length(test))]
    end
    println("Shards: $(length(train)) train, $(length(val)) val, $(length(test)) test")
    return train, val, test
end

# HVG indices
function load_hvg_idx(path::String)
    (path != "" && isfile(path)) || error("SC ETF requires hvg_path (got '$path')")
    d = load(path)
    println("  loaded $(length(d["hvg_idx"])) HVG indices from $path (computed from $(d["n_cells_scanned"]) cells)")
    return d["hvg_idx"]
end


end  # module LoadSC
