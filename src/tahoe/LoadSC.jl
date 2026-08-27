module LoadSC

using CSV, DataFrames, Random, PyCall, SparseArrays, JLD2

export load_gene_vocab, list_shards, shard_train_test_split, load_shard_split
export shard_train_val_test_split, load_shard_val_split
export load_shard_pyarrow

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


end  # module LoadSC
