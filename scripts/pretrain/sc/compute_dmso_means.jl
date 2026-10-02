# mean DMSO_TF profile per (cell line, plate) over Tahoe SC cells, for plate-matched delta inputs
# (delta = cell's log1p CP10k expression - mean of the DMSO_TF cells of the same cell line on the same plate)
# expression = log1p(1e4 x / total) over the 19,020 protein-coding genes (ProcessSC.cell_to_dense_flat!, as in training)
# controls are matched exactly on "DMSO_TF" ("Trametinib (DMSO_TF solvate)" is a drug)
#
# chunked over all shards (a plate's DMSO wells are spread across its shard block), then merged:
#   SC_CHUNK=i SC_NCHUNKS=n julia scripts/pretrain/sc/compute_dmso_means.jl      -> data/tahoe/sc_dmso_chunks/chunk_<i>.jld2
#   SC_MERGE=1 SC_NCHUNKS=n julia scripts/pretrain/sc/compute_dmso_means.jl      -> data/tahoe/sc_dmso_means.jld2
#   SC_NSHARDS=5 limits each chunk to its first shards (tests); SC_OUT redirects outputs

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(get(ENV, "JULIA_PROJECT", joinpath(@__DIR__, "../../..", arch_dir)))
using JLD2, Statistics, Printf, PyCall

push!(LOAD_PATH, joinpath(@__DIR__, "../../../src"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../../src/tahoe"))
using LoadSC, ProcessSC

logln(xs...) = (println(xs...); flush(stdout))
data_dir = get(ENV, "SC_DATA_DIR", "data/tahoe/data")
meta_dir = get(ENV, "SC_META_DIR", "data/tahoe/metadata")
coding_path = get(ENV, "SC_CODING", "data/tahoe/protein-coding_gene.txt")
out_root = get(ENV, "SC_OUT", "data/tahoe")
chunk_dir = joinpath(out_root, "sc_dmso_chunks"); mkpath(chunk_dir)
n_chunks = parse(Int, get(ENV, "SC_NCHUNKS", "1"))
const CTRL = "DMSO_TF"

if get(ENV, "SC_MERGE", "0") == "1"
    sums = Dict{Tuple{String,String},Vector{Float64}}(); counts = Dict{Tuple{String,String},Int}(); n_shards = 0
    for i in 1:n_chunks
        f = joinpath(chunk_dir, "chunk_$i.jld2"); isfile(f) || error("missing $f")
        c = load(f)
        for (k, s) in c["sums"]
            haskey(sums, k) ? (sums[k] .+= s) : (sums[k] = copy(s)); counts[k] = get(counts, k, 0) + c["counts"][k]
        end
        global n_shards += c["n_shards"]
    end
    ks = sort(collect(keys(sums)))
    means = reduce(hcat, [Float32.(sums[k] ./ counts[k]) for k in ks])   # genes × groups
    out = joinpath(out_root, "sc_dmso_means.jld2")
    jldsave(out; means, keys=ks, cell_lines=first.(ks), plates=last.(ks), counts=[counts[k] for k in ks], n_shards,
            control=CTRL, expression="log1p(1e4 x / total), protein-coding genes")
    logln("merged $n_chunks chunks, $n_shards shards, $(length(ks)) (cell line, plate) groups; DMSO cells per group: ",
          "min $(minimum(values(counts))), median $(round(Int, median(collect(values(counts))))) -> $out")
    exit(0)
end

chunk = parse(Int, get(ENV, "SC_CHUNK", "1"))
_, token_to_idx, n_coding = load_gene_vocab(meta_dir, coding_path)
all_shards = list_shards(data_dir)
mine = all_shards[chunk:n_chunks:end]                                   # round-robin: spreads plates over chunks
haskey(ENV, "SC_NSHARDS") && (mine = mine[1:min(end, parse(Int, ENV["SC_NSHARDS"]))])
logln("chunk $chunk / $n_chunks: $(length(mine)) of $(length(all_shards)) shards, $n_coding genes")

pq = pyimport("pyarrow.parquet")
sums = Dict{Tuple{String,String},Vector{Float64}}(); counts = Dict{Tuple{String,String},Int}()
dense = Vector{Float32}(undef, n_coding)
t0 = time(); n_ctrl = 0
for (si, sp) in enumerate(mine)
    t = pq.read_table(sp, columns=["drug", "cell_line_id", "plate"])
    drug = convert(Vector{String}, t.column("drug").to_pylist())
    ctrl = findall(==(CTRL), drug)
    if !isempty(ctrl)
        cl = convert(Vector{String}, t.column("cell_line_id").to_pylist()); pl = convert(Vector{String}, t.column("plate").to_pylist())
        sh = load_shard_pyarrow(sp)
        @assert sh.n_cells == length(drug)
        for ci in ctrl
            cell_to_dense_flat!(dense, sh.genes_flat, sh.offsets, sh.expr_flat, ci, token_to_idx)
            k = (cl[ci], pl[ci])
            s = get!(() -> zeros(n_coding), sums, k)
            s .+= dense; counts[k] = get(counts, k, 0) + 1
        end
        global n_ctrl += length(ctrl)
    end
    (si % 10 == 0 || si == length(mine)) && logln(@sprintf("  shard %d / %d, %d DMSO_TF cells so far, %.0f s", si, length(mine), n_ctrl, time() - t0))
end
jldsave(joinpath(chunk_dir, "chunk_$chunk.jld2"); sums, counts, n_shards=length(mine), shards=mine)
logln("chunk $chunk done: $(length(sums)) groups, $n_ctrl DMSO_TF cells, $(round(time() - t0)) s")
