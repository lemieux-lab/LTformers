#=
Plate confound check: label x plate cross-tab (chi2, Cramér's V), majority-class and plate-only lookup baselines.
Same samples/splits as finetuning; test = pretrain-unseen and full test groups.
Usage: julia --project=x86_64 scripts/analyses/plate_crosstab.jl lincs|tahoe|tahoe_sc
=#

using Pkg
arch_dir = Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"
Pkg.activate(joinpath(@__DIR__, "../..", arch_dir))
using JLD2, DataFrames, StatsBase, Statistics, CSV, Random, Printf
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"), joinpath(@__DIR__, "../../src/tahoe"))
using Preprocess: ttsplit, tvsplit, rank_genes
using ProcessLabels: dsplit
using Config: load_config

const DATASET = isempty(ARGS) ? "lincs" : ARGS[1]
@assert DATASET in ("lincs", "tahoe", "tahoe_sc")
const REPO = abspath(joinpath(@__DIR__, "../.."))
const OUT = joinpath(REPO, "results", DATASET == "lincs" ? "lincs" : DATASET == "tahoe" ? "tahoe/pb" : "tahoe/sc",
                     "analyses", "plate")
mkpath(OUT)


# label x plate stats
function crosstab_stats(labels::AbstractVector, plates::AbstractVector)
    n = length(labels)
    cell = countmap(collect(zip(labels, plates)))
    rows = countmap(labels); cols = countmap(plates)
    R, C = length(rows), length(cols)
    # chi2 = sum(O^2/E) - n over non-zero cells
    chi2 = sum(o^2 / (rows[l] * cols[p] / n) for ((l, p), o) in cell) - n
    dof = (R - 1) * (C - 1)
    v = sqrt(chi2 / (n * (min(R, C) - 1)))
    # Wilson-Hilferty z
    z = ((chi2 / dof)^(1 / 3) - (1 - 2 / (9dof))) / sqrt(2 / (9dof))
    per_plate = Dict{eltype(plates), Int}(); per_label = Dict{eltype(labels), Int}()
    for (l, p) in keys(cell)
        per_plate[p] = get(per_plate, p, 0) + 1
        per_label[l] = get(per_label, l, 0) + 1
    end
    lp = collect(values(per_plate)); pl = collect(values(per_label))
    return (; n, n_labels=R, n_plates=C, chi2, dof, cramers_v=v, wh_z=z,
              labels_per_plate_median=median(lp), labels_per_plate_mean=mean(lp), labels_per_plate_max=maximum(lp),
              plates_per_label_median=median(pl), plates_per_label_mean=mean(pl), plates_per_label_min=minimum(pl))
end

function majority_class(train_y, test_y)
    maj = first(sort(collect(countmap(train_y)), by=last, rev=true))[1]
    return mean(test_y .== maj), maj
end

function plate_lookup(train_y, train_p, test_y, test_p)
    _, maj = majority_class(train_y, test_y)
    by_plate = Dict{eltype(train_p), Dict{eltype(train_y), Int}}()
    for (y, p) in zip(train_y, train_p)
        c = get!(by_plate, p, Dict{eltype(train_y), Int}())
        c[y] = get(c, y, 0) + 1
    end
    best = Dict(p => first(sort(collect(c), by=last, rev=true))[1] for (p, c) in by_plate)
    pred = [get(best, p, maj) for p in test_p]
    return mean(pred .== test_y), mean(p in keys(best) for p in test_p)
end

rows = DataFrame(); xt = DataFrame()

function add_rows!(level, split, labels_tr, plates_tr, sets, n_cls)
    for (set, (yt, pt)) in sets
        isempty(yt) && continue
        z, _ = majority_class(labels_tr, yt)
        lk, seen = plate_lookup(labels_tr, plates_tr, yt, pt)
        push!(rows, (; dataset=DATASET, level, split, test_set=set, n_train=length(labels_tr), n_test=length(yt),
                       n_classes=n_cls, chance=1 / n_cls, majority_class=z, plate_lookup=lk, frac_test_on_train_plates=seen))
        @printf("%-8s %-5s %-14s %-12s n_test=%-8d chance=%.4f majority=%.4f plate-lookup=%.4f (test on train plates %.3f)\n",
                DATASET, level, split, set, length(yt), 1 / n_cls, z, lk, seen)
    end
end

if DATASET in ("lincs", "tahoe")
    if DATASET == "lincs"
        lincs = load(joinpath(REPO, "data/lincs/lincs_trt_data.jld2"))["filtered_data"]
        inst = lincs.inst; n = nrow(inst)
        plates = string.(inst.det_plate)
        lab = load(joinpath(REPO, "data/lincs/lincs_trt_inst.jld2"))
        labels = Dict("lvl1" => string.(lab["mfc"]), "lvl2" => string.(lab["pert_id"]))
        kw = (; label_path=joinpath(REPO, "data/lincs/lincs_trt_inst.jld2"), inst_df=inst, gene_df=lincs.gene)
        split_names = Dict(0 => "random", 1 => "group_plate")
    else
        df = load(joinpath(REPO, "data/tahoe/filtered_pseudobulks_alpha_10000.jld2"))["df"]; select!(df, Not(:expr))
        n = nrow(df); plates = string.(df.plate)
        labels = Dict("lvl1" => string.(df.cell_line), "lvl2" => string.(df.drug))
        kw = (; label_source=df)
        split_names = Dict(0 => "random", 1 => "group_drugdose")
    end
    X0 = zeros(Float32, 1, n)  # dummy matrix for dsplit
    for (level, gss) in (("lvl1", (0,)), ("lvl2", (0, 1)))
        for gs in gss
            cfg = Dict{String,Any}("data_format" => DATASET, "level" => level, "modeltype" => "elog", "group_split" => gs)
            d = dsplit(X0, cfg; kw..., ttsplit_fn=ttsplit, tvsplit_fn=tvsplit, rank_genes_fn=rank_genes)
            g(v) = d.label_idx[v]
            L, P = labels[level], plates
            sets = [("headline", (L[g(d.test_idx)], P[g(d.test_idx)]))]
            d.test_full_idx === d.test_idx || push!(sets, ("full_groups", (L[g(d.test_full_idx)], P[g(d.test_full_idx)])))
            add_rows!(level, split_names[gs], L[g(d.train_idx)], P[g(d.train_idx)], sets, d.n_classifications)
            if gs == 0
                st = crosstab_stats(L[d.label_idx], P[d.label_idx])
                push!(xt, (; dataset=DATASET, level, st...))
                println("  crosstab $level: ", st)
            end
        end
    end
else  # tahoe_sc
    using LoadSC: list_shards, sc_finetune_metadata_scan
    using SCTrain: sc_pt_unseen_shards
    config = load_config(joinpath(REPO, "config/default.toml"))
    pb = load(config["pb_data_path"])["df"]; select!(pb, Not(:expr))
    smp_plate = Dict(string(s) => string(p) for (s, p) in zip(pb.sample, pb.plate))
    unseen = sc_pt_unseen_shards(config)
    shards = list_shards(config["data_dir"])
    for (level, split_by) in (("lvl1", "well_cl"), ("lvl2", "drug_dose"))
        scan = sc_finetune_metadata_scan(shards, level; pb_data_path=config["pb_data_path"], split_by=split_by)
        plate(r) = get(smp_plate, r[4], "NA")
        ytr = [r[3] for r in scan.train_cells]; ptr = plate.(scan.train_cells)
        te = scan.test_cells
        u = [basename(r[1]) in unseen for r in te]
        yte = [r[3] for r in te]; pte = plate.(te)
        sets = [("headline", (yte[u], pte[u])), ("full_groups", (yte, pte))]
        add_rows!(level, split_by, ytr, ptr, sets, scan.n_cls)
        allr = vcat(scan.train_cells, scan.val_cells, scan.test_cells)
        st = crosstab_stats([r[3] for r in allr], plate.(allr))
        push!(xt, (; dataset=DATASET, level, st...))
        println("  crosstab $level: ", st)
        scan = nothing; allr = nothing; GC.gc()
    end
end

CSV.write(joinpath(OUT, "baselines.csv"), rows)
CSV.write(joinpath(OUT, "crosstab.csv"), xt)
println("wrote $(joinpath(OUT, "baselines.csv")) and crosstab.csv")
