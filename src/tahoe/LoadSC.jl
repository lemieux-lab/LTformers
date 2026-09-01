module LoadSC

using CSV, DataFrames, Random, PyCall, SparseArrays, JLD2, Flux, StatsBase

export load_gene_vocab, list_shards, shard_train_test_split, load_shard_split
export shard_train_val_test_split, load_shard_val_split
export load_shard_pyarrow, load_shard_metadata, load_sc_finetune_data

const _pq = PyNULL()
const _np = PyNULL()

function __init__()
    copy!(_pq, pyimport("pyarrow.parquet"))
    copy!(_np, pyimport("numpy"))
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

function shard_train_val_test_split(shard_paths::Vector{String}, val_ratio::Float64 = 0.1, test_ratio::Float64 = 0.1)
    n = length(shard_paths)
    idx = shuffle(1:n)
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

function load_shard_pyarrow(path::String)
    t = _pq.read_table(path)
    genes_combined = t.column("genes").combine_chunks()
    expr_combined = t.column("expressions").combine_chunks()
    genes_flat = convert(Vector{Int64}, _np.array(genes_combined.values, copy=true))
    offsets = convert(Vector{Int64}, _np.array(genes_combined.offsets, copy=true))
    expr_flat = convert(Vector{Float32}, _np.array(expr_combined.values, copy=true))
    n_cells = length(offsets) - 1
    return (; genes_flat, offsets, expr_flat, n_cells)
end


function load_shard_metadata(path::String)
    t = _pq.read_table(path, columns=["drug", "sample", "cell_line_id"])
    n_cells = t.num_rows
    drug = convert(Vector{String}, t.column("drug").to_pylist())
    sample = convert(Vector{String}, t.column("sample").to_pylist())
    cell_line_id = convert(Vector{String}, t.column("cell_line_id").to_pylist())
    return (; n_cells, drug, sample, cell_line_id)
end


"""
    load_sc_finetune_data(all_shards, level, token_to_idx, n_coding, top_k, modeltype; ...)

Pre-materialize SC finetune data from parquet shards into in-memory matrices.

Two-pass approach:
  1. Metadata scan — reads only drug/sample/cell_line_id columns to identify valid cells and build sample-level splits
  2. Expression extraction — loads full shards only for those containing valid cells, extracts features per cell

Returns a named tuple matching PB's `dsplit` output shape:
  `(; X_train, X_val, X_test, y_train, y_val, y_test, n_genes, n_classifications,
     train_idx, val_idx, test_idx, cidx_dict, cs)`
"""
function load_sc_finetune_data(all_shards::Vector{String}, level::String,
                                token_to_idx::Dict{Int,Int}, n_coding::Int,
                                top_k::Int, modeltype::String;
                                pb_data_path::String = "",
                                hvg_idx::Union{Vector{Int}, Nothing} = nothing,
                                subset_shards::Int = 0,
                                process_cell_topk_flat_fn = nothing,
                                cell_to_dense_flat_fn = nothing,
                                oversmpl_fn = nothing)

    # validate required function parameters
    if modeltype == "rtf" && isnothing(process_cell_topk_flat_fn)
        error("load_sc_finetune_data: process_cell_topk_flat_fn required for rtf modeltype")
    end
    if modeltype != "rtf" && isnothing(cell_to_dense_flat_fn)
        error("load_sc_finetune_data: cell_to_dense_flat_fn required for etf/mlp modeltype")
    end
    if level == "lvl2" && isnothing(oversmpl_fn)
        error("load_sc_finetune_data: oversmpl_fn required for lvl2")
    end

    # -- determine valid labels --
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

    # -- pass 1: metadata scan --
    shards_to_scan = subset_shards > 0 ? all_shards[1:min(subset_shards, length(all_shards))] : all_shards
    println("[pass 1] scanning metadata from $(length(shards_to_scan)) shards...")
    flush(stdout)

    # cell_records: (shard_path, cell_idx_in_shard, label, sample_id)
    cell_records = Tuple{String, Int, String, String}[]
    for (si, sp) in enumerate(shards_to_scan)
        meta = load_shard_metadata(sp)
        for i in 1:meta.n_cells
            if level == "lvl1"
                label = meta.cell_line_id[i]
            else  # lvl2
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

    # -- sample-level split --
    unique_samples = unique(r[4] for r in cell_records)
    shuffle!(unique_samples)
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

    # -- process labels --
    all_labels = [r[3] for r in cell_records]
    unique_labels = sort(unique(all_labels))
    label_to_id = Dict(l => i for (i, l) in enumerate(unique_labels))
    n_cls = length(unique_labels)
    println("n_classifications: $n_cls")

    # -- pass 2: expression extraction --
    println("[pass 2] extracting expression features...")
    flush(stdout)

    # ProcessSC functions passed in from caller (avoids fragile @eval Main pattern)
    # import ProcessSC functions
    # if !isdefined(Main, :ProcessSC)
    #     push!(LOAD_PATH, @__DIR__)
    #     @eval Main using ProcessSC
    # end
    # _process_cell_topk_flat = Main.ProcessSC.process_cell_topk_flat
    # _cell_to_dense_flat! = Main.ProcessSC.cell_to_dense_flat!
    _process_cell_topk_flat = process_cell_topk_flat_fn
    _cell_to_dense_flat! = cell_to_dense_flat_fn

    function _materialize_split(cells, label_to_id, n_cls, token_to_idx, n_coding, top_k, modeltype, hvg_idx)
        n = length(cells)
        n == 0 && error("empty split")

        # determine feature dimension
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

        # group cells by shard for efficient loading
        by_shard = Dict{String, Vector{Tuple{Int, Int}}}()  # shard_path => [(cell_idx_in_shard, col_in_X)]
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

    # oversampling setup (for lvl2)
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


end  # module LoadSC
