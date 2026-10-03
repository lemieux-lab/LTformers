module SCTrain

# shared SC finetune loop: cross-shard mixed train batches, step-based val eval on a fixed val-shard subset,
# best-val checkpointing, final + best test eval (best test skipped when best == final)

using Flux, CUDA, JLD2, Statistics, Random, Dates

let d = @__DIR__, s = joinpath(@__DIR__, ".."); d in LOAD_PATH || push!(LOAD_PATH, d); s in LOAD_PATH || push!(LOAD_PATH, s); end
using Log: log_model, log_params, load_best_cpu
using Plot: plot_loss
using Models: fix_gpu_dropout
using LoadSC: finetune_batches_from_shard, finetune_train_batches, draw_train_cells, load_shard_pyarrow, _build_ft_batch, memlog

export sc_eval_paths, sc_finetune!, sc_feature_stats, sc_pt_unseen_shards


# shards pretraining didn't train on
const SC_PT_SPLIT = joinpath(@__DIR__, "..", "..", "data", "tahoe", "sc_pretrain_shard_split.jld2")
function sc_pt_unseen_shards(config)
    p = something(get(config, "sc_split_path", nothing), SC_PT_SPLIT)
    isfile(p) || (@warn "no SC pretrain shard split at $p: val/test not restricted to pretrain-unseen shards"; return nothing)
    s = load(p)
    return Set(vcat(basename.(s["val_shards"]), basename.(s["test_shards"])))
end


# val: fixed seeded subset of ft_eval_shards shards (default 100, 0 = all); test: ft_test_shards (default 0 = all)
function sc_eval_paths(d, config)
    d.train_shard_map === nothing && return nothing, nothing
    n_val = something(get(config, "ft_eval_shards", nothing), 100)
    n_test = something(get(config, "ft_test_shards", nothing), 0)
    unseen = sc_pt_unseen_shards(config)
    vp_all = sort(d.val_shard_paths)
    vp_keep = isnothing(unseen) ? vp_all : filter(sp -> basename(sp) in unseen, vp_all)
    isnothing(unseen) || println("val shards restricted to pretrain-unseen: $(length(vp_keep))/$(length(vp_all))")
    vp = shuffle(MersenneTwister(42), vp_keep)
    tp = sort(d.test_shard_paths)
    val_paths = n_val > 0 ? vp[1:min(n_val, length(vp))] : vp
    test_paths = n_test > 0 ? shuffle(MersenneTwister(42), tp)[1:min(n_test, length(tp))] : tp
    println("eval shards: val=$(length(val_paths))/$(length(vp)) test=$(length(test_paths))/$(length(tp))")
    return val_paths, test_paths
end

# per-feature train mean/sd from a cell sample (n_shards random train shards × ≤n_per_shard cells), for --standardize
function sc_feature_stats(d, fk; n_shards::Int = 100, n_per_shard::Int = 2000)
    by_shard = Dict{String, Vector{Int}}()
    for (sp, ci) in ((r[1], r[2]) for r in d.train_cells)
        push!(get!(by_shard, sp, Int[]), ci)
    end
    shards = shuffle(MersenneTwister(7), sort(collect(keys(by_shard))))[1:min(n_shards, length(by_shard))]
    s1 = nothing; s2 = nothing; n = 0
    for sp in shards
        cis = by_shard[sp]
        cis = length(cis) > n_per_shard ? shuffle(MersenneTwister(7), cis)[1:n_per_shard] : cis
        X = Float64.(_build_ft_batch(load_shard_pyarrow(sp), cis, fk.token_to_idx, fk.n_coding, fk.top_k, fk.feat_mt,
                                     fk.hvg_idx, fk.process_cell_topk_flat_fn, fk.cell_to_dense_flat_fn))
        s1 = isnothing(s1) ? vec(sum(X, dims=2)) : s1 .+ vec(sum(X, dims=2))
        s2 = isnothing(s2) ? vec(sum(X .^ 2, dims=2)) : s2 .+ vec(sum(X .^ 2, dims=2))
        n += size(X, 2)
    end
    μ = s1 ./ n
    σ = sqrt.(max.(s2 ./ n .- μ .^ 2, 0.0))
    σ[σ .< 1e-6] .= 1.0                      # constant features -> centered only
    println("feature stats: $n cells from $(length(shards)) train shards, mean sd = $(round(mean(σ), digits=4))"); flush(stdout)
    memlog("after feature stats")
    return Float32.(μ), Float32.(σ)
end

_loss(logits, y, is_regression) = is_regression ? Flux.mse(logits, y) : Flux.logitcrossentropy(logits, y)

# loss (+ preds/trues) over streamed shards or an in-memory matrix
# group_map (test only): shard -> (well keys, well|cell-line keys) per cell -> also returns accuracy of
# mean softmax per group (pseudobulk-comparable: a PB sample = one (well, cell line))
function _eval(m, fk, is_regression; paths = nothing, shard_map = nothing, X = nothing, Y = nothing, label = "val",
               group_map = nothing, unseen = nothing)
    preds = is_regression ? Float32[] : Int[]
    trues = is_regression ? Float32[] : Int[]
    losses = Float32[]; ns = Int[]
    agg = [Dict{String, Vector{Float32}}(), Dict{String, Vector{Float32}}()]   # well, well|cl -> summed probs
    agg_lab = [Dict{String, Int}(), Dict{String, Int}()]                        # group -> label (-1 = mixed)
    # same, pretrain-unseen cells only
    agg_u = [Dict{String, Vector{Float32}}(), Dict{String, Vector{Float32}}()]
    agg_lab_u = [Dict{String, Int}(), Dict{String, Int}()]
    pred_unseen = Bool[]; cur_unseen = false
    pred_shard = Int32[]; pred_cell = Int32[]  # shard, cell per prediction
    function step!(x, y, gkeys = nothing)
        x_gpu = fk.to_x(x); y_gpu = CuArray(y)
        logits = m(x_gpu)
        push!(losses, Float32(cpu(_loss(logits, y_gpu, is_regression)))); push!(ns, size(y, 2))
        if is_regression
            append!(preds, vec(cpu(logits))); append!(trues, vec(cpu(y_gpu)))
        else
            append!(preds, Flux.onecold(cpu(logits))); append!(trues, Flux.onecold(cpu(y)))
            if !isnothing(gkeys)
                probs = cpu(Flux.softmax(logits)); labs = Flux.onecold(cpu(y))
                for k in 1:2, j in axes(probs, 2)
                    g = gkeys[k][j]
                    acc_v = get!(() -> zeros(Float32, size(probs, 1)), agg[k], g)
                    acc_v .+= view(probs, :, j)
                    l = get!(agg_lab[k], g, labs[j]); l != labs[j] && (agg_lab[k][g] = -1)
                    if cur_unseen
                        acc_u = get!(() -> zeros(Float32, size(probs, 1)), agg_u[k], g)
                        acc_u .+= view(probs, :, j)
                        lu = get!(agg_lab_u[k], g, labs[j]); lu != labs[j] && (agg_lab_u[k][g] = -1)
                    end
                end
            end
        end
        append!(pred_unseen, fill(cur_unseen, size(y, 2)))
        CUDA.unsafe_free!(x_gpu); CUDA.unsafe_free!(y_gpu)
    end
    if !isnothing(paths)
        for (i, sp) in enumerate(paths)
            cur_unseen = isnothing(unseen) || basename(sp) in unseen
            cis, labs = shard_map[sp]
            gk = isnothing(group_map) ? nothing : group_map[sp]
            off = 0
            for (x, y) in finetune_batches_from_shard(sp, cis, labs, fk.token_to_idx, fk.n_coding, fk.top_k,
                                                       fk.batch_size, fk.feat_mt, fk.n_cls; hvg_idx=fk.hvg_idx,
                                                       process_cell_topk_flat_fn=fk.process_cell_topk_flat_fn,
                                                       cell_to_dense_flat_fn=fk.cell_to_dense_flat_fn)
                bs = size(y, 2)
                step!(x, y, isnothing(gk) ? nothing : (gk[1][off+1:off+bs], gk[2][off+1:off+bs]))
                append!(pred_shard, fill(Int32(i), bs)); append!(pred_cell, Int32.(cis[off+1:off+bs]))
                off += bs
            end
            i % 200 == 0 && (println("    $label shard $i/$(length(paths))"); flush(stdout))
        end
    else
        for s in 1:fk.batch_size:size(X, 2)
            e = min(s + fk.batch_size - 1, size(X, 2))
            step!(X[:, s:e], Y[:, s:e])
        end
    end
    loss = isempty(losses) ? NaN32 : Float32(sum(losses .* ns) / sum(ns))
    acc = is_regression ? NaN : mean(preds .== trues)
    isnothing(group_map) && return loss, acc, preds, trues
    # group accuracy (NaN if a grouping mixes labels, e.g. per-well for lvl1 cell-line ID)
    gacc(k) = any(==(-1), values(agg_lab[k])) ? NaN :
              mean(argmax(v) == agg_lab[k][g] for (g, v) in agg[k])
    gacc_u(k) = isempty(agg_u[k]) || any(==(-1), values(agg_lab_u[k])) ? NaN :
                mean(argmax(v) == agg_lab_u[k][g] for (g, v) in agg_u[k])
    ga = (well_acc = gacc(1), n_wells = length(agg[1]), wellcl_acc = gacc(2), n_wellcl = length(agg[2]),
          well_acc_u = gacc_u(1), n_wells_u = length(agg_u[1]), wellcl_acc_u = gacc_u(2), n_wellcl_u = length(agg_u[2]),
          unseen_mask = BitVector(pred_unseen), pred_shard = pred_shard, pred_cell = pred_cell)
    println("  $label aggregated: per-well acc=$(round(ga.well_acc, digits=4)) (n=$(ga.n_wells)), " *
            "per-(well, cell line) acc=$(round(ga.wellcl_acc, digits=4)) (n=$(ga.n_wellcl))"); flush(stdout)
    isnothing(unseen) || println("  $label pretrain-unseen cells: $(count(pred_unseen))/$(length(pred_unseen)), " *
            "acc=$(round(mean(preds[pred_unseen] .== trues[pred_unseen]), digits=4)), per-(well, cell line) acc=$(round(ga.wellcl_acc_u, digits=4)) (n=$(ga.n_wellcl_u))")
    return loss, acc, preds, trues, ga
end

# train batches: streaming -> class-balanced (lvl2) cross-shard mix; in-memory (lvl3) -> shuffled epochs
function _train_batches(d, fk, config, is_regression, n_cells)
    if d.train_shard_map !== nothing
        return finetune_train_batches(d.train_cells, d.label_to_id, fk.n_cls, n_cells,
                                      fk.token_to_idx, fk.n_coding, fk.top_k, fk.batch_size, fk.feat_mt;
                                      balanced=(config["level"] == "lvl2" && !is_regression),
                                      group_shards=something(get(config, "group_shards", nothing), 64),
                                      hvg_idx=fk.hvg_idx,
                                      process_cell_topk_flat_fn=fk.process_cell_topk_flat_fn,
                                      cell_to_dense_flat_fn=fk.cell_to_dense_flat_fn)
    end
    n = size(d.X_train, 2)
    perm = randperm(n)
    return ((d.X_train[:, perm[s:min(s + fk.batch_size - 1, n)]], d.y_train[:, perm[s:min(s + fk.batch_size - 1, n)]])
            for s in 1:fk.batch_size:(div(n, fk.batch_size) * fk.batch_size))
end

"""
    sc_finetune!(model, opt, d, fk, config; is_regression, save_dir, gpu_info, skip, loss_name, wb, val_paths, test_paths)

fk: (; token_to_idx, n_coding, top_k, batch_size, feat_mt, n_cls, hvg_idx, to_x,
       process_cell_topk_flat_fn, cell_to_dense_flat_fn)
Budget: max_ft_steps > 0 -> that many steps (one pass of drawn cells); else n_epochs passes over all train cells.
Val every ft_eval_every steps (default 4000; 0 = only at the end / epoch ends) and at the last step.
"""
function sc_finetune!(model, opt, d, fk, config; is_regression::Bool, save_dir::String, gpu_info::String,
                      skip, loss_name::String, wb = nothing, val_paths = nothing, test_paths = nothing)
    streaming = d.train_shard_map !== nothing
    max_steps = get(config, "max_ft_steps", 0)
    eval_every = something(get(config, "ft_eval_every", nothing), 4000)
    n_passes = max_steps > 0 ? (streaming ? 1 : typemax(Int)) : config["n_epochs"]
    n_cells = streaming ? (max_steps > 0 ? max_steps * fk.batch_size : d.n_train_cells) : 0

    train_losses = Float32[]; val_losses = Float32[]; val_accs = Float64[]; val_steps = Int[]
    best_val_loss = Inf32; best_val_acc = -Inf; best_step = 0
    best_state = nothing  # best-val weights on cpu in RAM; written to best/ once after training (not at every improvement)
    save_model = something(get(config, "save_model", nothing), 1) != 0  # 0: no model_state.jld2 files (sweeps)
    step = 0; run_losses = Float32[]; done = false

    # --cache_val 1 (classification, streaming): read the val shards once, keep raw features in memory, reuse them at
    # every val eval (same cells/order/transform -> same val acc; saves ~20x100 shard reads per run). ~1G at 1024
    # features, ~19G at full genes, so only for small feature sets
    cache_val = streaming && !is_regression && something(get(config, "cache_val", nothing), 0) == 1
    val_X = nothing; val_Y = nothing
    function build_val_cache!()
        xs = Any[]; labs_all = Int[]
        for sp in val_paths
            cis, labs = d.val_shard_map[sp]
            for (x, y) in finetune_batches_from_shard(sp, cis, labs, fk.token_to_idx, fk.n_coding, fk.top_k,
                                                       fk.batch_size, fk.feat_mt, fk.n_cls; hvg_idx=fk.hvg_idx,
                                                       process_cell_topk_flat_fn=fk.process_cell_topk_flat_fn,
                                                       cell_to_dense_flat_fn=fk.cell_to_dense_flat_fn)
                push!(xs, x); append!(labs_all, Flux.onecold(y))
            end
        end
        val_X = reduce(hcat, xs); val_Y = Flux.onehotbatch(labs_all, 1:fk.n_cls)
        memlog("val cache built ($(length(labs_all)) cells from $(length(val_paths)) shards)")
    end

    function do_val!()
        Flux.testmode!(model)
        cache_val && isnothing(val_X) && build_val_cache!()
        vl, va, _, _ = cache_val ? _eval(model, fk, is_regression; X=val_X, Y=val_Y) :
                       streaming ? _eval(model, fk, is_regression; paths=val_paths, shard_map=d.val_shard_map) :
                                   _eval(model, fk, is_regression; X=d.X_val, Y=d.y_val)
        Flux.trainmode!(model)
        push!(train_losses, isempty(run_losses) ? NaN32 : mean(run_losses)); empty!(run_losses)
        push!(val_losses, vl); push!(val_accs, va); push!(val_steps, step)
        println("  step $step: train=$(round(train_losses[end], digits=4)) val=$(round(vl, digits=4))" *
                (is_regression ? "" : " val_acc=$(round(va, digits=4))")); flush(stdout)
        # regression: lowest val loss; classification: highest val accuracy (ties -> lower loss).
        # val loss bottoms out early under held-out wells while accuracy keeps rising
        improved = is_regression ? vl < best_val_loss : (va > best_val_acc || (va == best_val_acc && vl < best_val_loss))
        if improved
            best_val_loss = vl; best_val_acc = va; best_step = step
            best_dir = joinpath(save_dir, "best"); mkpath(best_dir)
            # log_model(model, best_dir)
            best_state = Flux.state(cpu(model))
            plot_loss(length(train_losses), train_losses, val_losses, best_dir, loss_name)
            jldsave(joinpath(best_dir, "losses.jld2"); val_steps, train_losses, val_losses, val_accs)
            log_params(config, gpu_info, 0, 0, best_dir; skip=skip, total_steps=step,
                       best_step=best_step, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        end
        if wb !== nothing
            ld = Dict("global_step" => step, "train_loss" => train_losses[end], "val_loss" => vl,
                      "best_val_loss" => best_val_loss)  # per-eval so hyperband early_terminate can see it
            is_regression || (ld["val_acc"] = va)
            is_regression || (ld["best_val_acc"] = best_val_acc)  # sweep metric (classification): matches best/ selection
            wb.log(ld)
        end
    end

    memlog("start train")
    Flux.trainmode!(model)
    for pass in 1:n_passes
        done && break
        for (x, y) in _train_batches(d, fk, config, is_regression, n_cells)
            x_gpu = fk.to_x(x); y_gpu = CuArray(y)
            lv, grads = Flux.withgradient(m -> _loss(m(x_gpu), y_gpu, is_regression), model)
            Flux.update!(opt, model, grads[1])
            CUDA.unsafe_free!(x_gpu); CUDA.unsafe_free!(y_gpu)
            push!(run_losses, Float32(cpu(lv)))
            step += 1
            step % 1000 == 0 && (println("  step $step train_loss(last 1k)=$(round(mean(run_losses[max(1, end-999):end]), digits=4))"); flush(stdout))
            eval_every > 0 && step % eval_every == 0 && do_val!()
            if max_steps > 0 && step >= max_steps
                done = true; break
            end
        end
        # epoch mode: eval at every pass end
        max_steps == 0 && (isempty(val_steps) || val_steps[end] != step) && do_val!()
    end
    (isempty(val_steps) || val_steps[end] != step) && do_val!()

    if save_model && !isnothing(best_state)
        mkpath(joinpath(save_dir, "best"))
        jldsave(joinpath(save_dir, "best", "model_state.jld2"); model_state=best_state)
    end

    # test: final model, then best only if it differs from final
    Flux.testmode!(model)
    GC.gc(true); CUDA.reclaim()
    println("  test eval ($(streaming ? length(test_paths) : size(d.X_test, 2)) $(streaming ? "shards" : "samples"))"); flush(stdout)
    gm = (streaming && !is_regression) ? d.test_group_map : nothing
    unseen = (streaming && !is_regression) ? sc_pt_unseen_shards(config) : nothing
    no_agg = (well_acc = NaN, n_wells = 0, wellcl_acc = NaN, n_wellcl = 0)
    res = streaming ? _eval(model, fk, is_regression; paths=test_paths, shard_map=d.test_shard_map, label="test", group_map=gm, unseen=unseen) :
                      _eval(model, fk, is_regression; X=d.X_test, Y=d.y_test, label="test")
    tl, _, all_preds, all_trues = res[1:4]
    final_agg = length(res) == 5 ? res[5] : no_agg
    best_preds, best_trues, best_agg = all_preds, all_trues, final_agg
    memlog("after final test")
    if best_step != step
        # best_cpu = load_best_cpu(model, save_dir)
        best_cpu = isnothing(best_state) ? nothing : Flux.loadmodel!(cpu(model), best_state)
        if !isnothing(best_cpu)
            best_model = fix_gpu_dropout(cu(best_cpu))
            Flux.testmode!(best_model)
            println("  best-model test eval (step $best_step)"); flush(stdout)
            res = streaming ? _eval(best_model, fk, is_regression; paths=test_paths, shard_map=d.test_shard_map, label="test", group_map=gm, unseen=unseen) :
                              _eval(best_model, fk, is_regression; X=d.X_test, Y=d.y_test, label="test")
            best_preds, best_trues = res[3], res[4]
            best_agg = length(res) == 5 ? res[5] : no_agg
        end
    else
        println("  best step = final step ($step): reusing final test predictions")
    end
    jldsave(joinpath(save_dir, "eval_history.jld2"); val_steps, train_losses, val_losses, val_accs, best_step)
    # test = pretrain-unseen cells; *_full = all test cells
    all_preds_full, all_trues_full, best_preds_full, best_trues_full = all_preds, all_trues, best_preds, best_trues
    best_ids = hasproperty(best_agg, :pred_shard) ?
        (shard = basename.(test_paths)[best_agg.pred_shard], cell = best_agg.pred_cell, pt_unseen = best_agg.unseen_mask) : nothing
    final_agg_full, best_agg_full = final_agg, best_agg
    _u(a) = hasproperty(a, :unseen_mask) ? (well_acc = a.well_acc_u, n_wells = a.n_wells_u, wellcl_acc = a.wellcl_acc_u, n_wellcl = a.n_wellcl_u) : a
    if !isnothing(unseen) && hasproperty(final_agg, :unseen_mask)
        mf = final_agg.unseen_mask; mb = best_agg.unseen_mask
        all_preds, all_trues = all_preds[mf], all_trues[mf]
        best_preds, best_trues = best_preds[mb], best_trues[mb]
        final_agg, best_agg = _u(final_agg), _u(best_agg)
    end
    return (; train_losses, val_losses, val_accs, val_steps, test_losses=Float32[tl],
              all_preds, all_trues, best_preds, best_trues, best_step, best_val_loss, global_step=step,
              final_agg, best_agg, best_val_acc,
              all_preds_full, all_trues_full, best_preds_full, best_trues_full, final_agg_full, best_agg_full,
              split_tag=(isnothing(unseen) ? "sc_full" : "sc_ptunseen"), best_ids)
end


end  # module SCTrain
