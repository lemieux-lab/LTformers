module Args

using ArgParse

export load_pretrain_args, load_finetune_args, load_sc_finetune_args


function load_pretrain_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config", "-c"
            help = "path to TOML config file"
            arg_type = String
            default = "config/default.toml"
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
        "--modeltype", "-t"
            help = "model type: rtf or etf"
            arg_type = String
            required = true
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
        "--additional_notes", "-n"
            help = "run-specific notes"
            arg_type = String
        "--lr"
            help = "learning rate"
            arg_type = Float64
        "--drop_prob"
            help = "dropout probability"
            arg_type = Float64
        "--embed_dim"
            help = "embedding dimension"
            arg_type = Int
        "--hidden_dim"
            help = "hidden layer dimension"
            arg_type = Int
        "--n_heads"
            help = "number of attention heads"
            arg_type = Int
        "--n_layers"
            help = "number of transformer layers"
            arg_type = Int
        "--data_path"
            help = "path to JLD2 data file"
            arg_type = String
        "--data_format"
            help = "data format: tahoe or lincs"
            arg_type = String
        "--sorted_gene_path"
            help = "path to sorted_gene_indices_by_exp.jld2 for global-rank error plots"
            arg_type = String
        "--n_hvg"
            help = "number of highly variable genes to keep (0 = all)"
            arg_type = Int
        "--subset_ratio"
            help = "fraction of data to use (stratified by pert_type x cell line)"
            arg_type = Float64
        "--mask_ratio"
            help = "fraction of tokens to mask"
            arg_type = Float64
        "--max_steps"
            help = "max training steps (0 = use n_epochs)"
            arg_type = Int
        "--ema_decay"
            help = "EMA decay rate for teacher model"
            arg_type = Float64
        "--eval_every"
            help = "PB pretrain: val/checkpoint/log every N train steps (0 = once per pass over the data)"
            arg_type = Int
        "--max_val_samples"
            help = "PB pretrain: val on the first N val samples (0 = all)"
            arg_type = Int
        "--wandb_mode"
            help = "wandb mode: disabled, online, or offline"
            arg_type = String
        "--data_dir"
            help = "path to Tahoe-100M parquet shard directory"
            arg_type = String
        "--meta_dir"
            help = "path to Tahoe-100M metadata directory"
            arg_type = String
        "--coding_gene_path"
            help = "path to protein-coding gene list TSV"
            arg_type = String
        "--top_k"
            help = "number of top genes per cell for sequence input"
            arg_type = Int
        "--n_eval_shards"
            help = "number of test shards to evaluate per epoch"
            arg_type = Int
        "--n_val_shards"
            help = "SC pretrain: val shards (checkpoint + sweep metric; 0 = n_eval_shards)"
            arg_type = Int
        "--subset_shards"
            help = "max number of shards to use (0 = all); applied after train/test split"
            arg_type = Int
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
        "--hvg_n_shards"
            help = "number of shards to scan for HVG computation (compute_hvg.jl)"
            arg_type = Int
        "--hvg_out"
            help = "output path for HVG indices (compute_hvg.jl)"
            arg_type = String
    end
    return parse_args(s)
end

function load_finetune_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config", "-c"
            help = "path to TOML config file"
            arg_type = String
            default = "config/default.toml"
        "--mode", "-m"
            help = "ft mode: e2e or emb"
            arg_type = String
            required = true
        "--task"
            help = "pretrain objective: mlm, lrecon, or erecon"
            arg_type = String
        "--model_dir"
            help = "path to pretrained model directory (contains model_state.jld2)"
            arg_type = String
        "--level", "-l"
            help = "level of finetuning: lvl1, lvl2, or lvl3 (regression)"
            arg_type = String
            required = true
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
            required = true
        "--modeltype", "-t"
            help = "model type: rtf, etf, rmlp, emlp, rlog, or elog"
            arg_type = String
            required = true
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
        "--additional_notes", "-n"
            help = "run-specific notes"
            arg_type = String
        "--lr"
            help = "learning rate"
            arg_type = Float64
        "--drop_prob"
            help = "dropout probability"
            arg_type = Float64
        "--embed_dim"
            help = "embedding dimension"
            arg_type = Int
        "--hidden_dim"
            help = "hidden layer dimension"
            arg_type = Int
        "--n_heads"
            help = "number of attention heads"
            arg_type = Int
        "--n_layers"
            help = "number of transformer layers"
            arg_type = Int
        "--mlp_hidden_dim"
            help = "emlp/rmlp: constant hidden width (n_layers hidden layers); 0 = tapered n_genes -> n_classes"
            arg_type = Int
        "--mlp_shape"
            help = "emlp/rmlp with mlp_hidden_dim > 0: const (all hidden = mlp_hidden_dim) or funnel (halve each layer)"
            arg_type = String
        "--max_ft_steps"
            help = "max finetune steps (0 = use n_epochs)"
            arg_type = Int
        "--wandb_mode"
            help = "wandb mode: disabled, online, or offline"
            arg_type = String
        "--data_path"
            help = "path to JLD2 data file"
            arg_type = String
        "--data_format"
            help = "data format: tahoe or lincs"
            arg_type = String
        "--label_path"
            help = "path to label JLD2 file (LINCS only; has mfc + pert_id)"
            arg_type = String
        "--n_hvg"
            help = "number of highly variable genes to keep (0 = all)"
            arg_type = Int
        "--subset_ratio"
            help = "fraction of data to use (stratified by pert_type x cell line)"
            arg_type = Float64
        "--sorted_gene_path"
            help = "path to sorted_gene_indices_by_exp.jld2 for global-rank error plots"
            arg_type = String
        "--source_cell"
            help = "source cell line for lvl3 regression (default: MCF7)"
            arg_type = String
        "--target_cell"
            help = "target cell line for lvl3 regression (default: PC3)"
            arg_type = String
        "--dose"
            help = "dose filter for lvl3 regression (e.g. '5.0' for Tahoe, '10.0' for LINCS; empty = no filter)"
            arg_type = String
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
        "--rank_top_k"
            help = "rlog/rmlp only: keep each sample's top-k ranks, tie the rest at the bottom (0 = off; matches rtf input info)"
            arg_type = Int
        "--save_model"
            help = "PB emlp/rmlp: 1 = write model_state.jld2 (best + final), 0 = skip (sweeps)"
            arg_type = Int
        "--standardize"
            help = "PB elog/emlp: per-gene z-score inputs with train-split mean/sd (default 1; 0 = raw)"
            arg_type = Int
        "--clip"
            help = "PB elog/emlp: clamp z-scores to [-c, c] (default 10; 0 = off). tag _zc (clipped) / _z"
            arg_type = Float64
        "--weight_decay"
            help = "PB elog/rlog: AdamW weight decay λ (per-step shrink lr·λ, i.e. L2; default 0; saves as <tag>_wd<λ>)"
            arg_type = Float64
        "--group_split"
            help = "1 = group split: tahoe PB by (drug, dose) (_gdd); LINCS by detection plate, whole plates held out (_gpl)"
            arg_type = Int
        "--split_path"
            help = "pretrain split file (default: ProcessLabels.CANONICAL_SPLITS)"
            arg_type = String
        "--lincs_min_n"
            help = "LINCS lvl2: min profiles per trt_cp compound (default 500)"
            arg_type = Int
        "--input"
            help = "abs (default) = stored log expression; delta = minus the mean DMSO of the same cell line + plate (elog/emlp/etf; _delta)"
            arg_type = String
    end
    return parse_args(s)
end


function load_sc_finetune_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        # finetune args
        "--config", "-c"
            help = "path to TOML config file"
            arg_type = String
            default = "config/default.toml"
        "--mode", "-m"
            help = "ft mode: e2e or emb"
            arg_type = String
            required = true
        "--task"
            help = "pretrain objective: mlm, lrecon, or erecon"
            arg_type = String
        "--model_dir"
            help = "path to pretrained model directory (contains model_state.jld2)"
            arg_type = String
        "--level", "-l"
            help = "level of finetuning: lvl1, lvl2, or lvl3 (regression)"
            arg_type = String
            required = true
        "--n_epochs", "-e"
            help = "number of epochs total"
            arg_type = Int
            required = true
        "--modeltype", "-t"
            help = "model type: rtf, etf, rmlp, emlp, rlog, or elog"
            arg_type = String
            required = true
        "--batch_size", "-b"
            help = "batchsize"
            arg_type = Int
        "--additional_notes", "-n"
            help = "run-specific notes"
            arg_type = String
        "--lr"
            help = "learning rate"
            arg_type = Float64
        "--drop_prob"
            help = "dropout probability"
            arg_type = Float64
        "--embed_dim"
            help = "embedding dimension"
            arg_type = Int
        "--hidden_dim"
            help = "hidden layer dimension"
            arg_type = Int
        "--n_heads"
            help = "number of attention heads"
            arg_type = Int
        "--n_layers"
            help = "number of transformer layers"
            arg_type = Int
        "--mlp_hidden_dim"
            help = "emlp/rmlp: constant hidden width (n_layers hidden layers); 0 = tapered n_genes -> n_classes"
            arg_type = Int
        "--save_model"
            help = "1 = write model_state.jld2 (best + final), 0 = skip (sweeps)"
            arg_type = Int
        "--mlp_shape"
            help = "emlp/rmlp with mlp_hidden_dim > 0: const (all hidden = mlp_hidden_dim) or funnel (halve each layer)"
            arg_type = String
        "--max_ft_steps"
            help = "max finetune steps (0 = use n_epochs)"
            arg_type = Int
        "--wandb_mode"
            help = "wandb mode: disabled, online, or offline"
            arg_type = String
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
        "--ft_eval_shards"
            help = "SC finetune: val shards per step eval (fixed seeded subset; default 100, 0 = all); separate from pretrain n_eval_shards"
            arg_type = Int
        "--ft_test_shards"
            help = "SC finetune: test shards (default 0 = all)"
            arg_type = Int
        "--ft_eval_every"
            help = "SC finetune: val eval every N steps (default 4000; 0 = only at the end / epoch ends)"
            arg_type = Int
        "--group_shards"
            help = "SC finetune: shards pooled + shuffled together per train-batch group (default 64)"
            arg_type = Int
        "--standardize"
            help = "SC elog/emlp: z-score features with train mean/sd from 100 train shards (default 1; 0 = raw)"
            arg_type = Int
        "--clip"
            help = "SC elog/emlp: clamp z-scores to [-c, c] (default 10; 0 = off). tag _zc (clipped) / _z"
            arg_type = Float64
        "--cache_val"
            help = "SC finetune: 1 = read val shards once and keep features in memory (use for <=1024 features; ~19G at full genes)"
            arg_type = Int
        "--sc_split"
            help = "SC finetune split unit: drug_dose (default; all wells of a drug-dose together), well (_well), or well_cl ((well, cell line) units = random PB split; _wcl)"
            arg_type = String
        "--input"
            help = "abs (default) = stored log expression; delta = minus the mean DMSO_TF of the same cell line + plate (lvl2; data/tahoe/sc_dmso_means.jld2; _delta)"
            arg_type = String
        # sc args
        "--data_dir"
            help = "path to Tahoe-100M parquet shard directory"
            arg_type = String
        "--meta_dir"
            help = "path to Tahoe-100M metadata directory (gene_vocabulary.jsonl)"
            arg_type = String
        "--coding_gene_path"
            help = "path to protein-coding gene list TSV"
            arg_type = String
        "--top_k"
            help = "number of top genes per cell for RTF sequence input"
            arg_type = Int
        "--n_hvg"
            help = "number of highly variable genes to keep (ETF; 0 = all)"
            arg_type = Int
        "--hvg_path"
            help = "path to pre-computed HVG indices JLD2 (for ETF)"
            arg_type = String
        "--rank_top_k"
            help = "rlog/rmlp only: rank features for each cell's top-k genes (default: top_k; 0 = all genes -> <model>_full)"
            arg_type = Int
        "--subset_shards"
            help = "max number of shards to use (0 = all); for debugging"
            arg_type = Int
        "--pb_data_path"
            help = "path to PB JLD2 for determining valid lvl2 drugs"
            arg_type = String
        "--source_cell"
            help = "source cell line for lvl3 regression"
            arg_type = String
        "--target_cell"
            help = "target cell line for lvl3 regression"
            arg_type = String
        "--dose"
            help = "dose filter for lvl3 regression (e.g. '5.0'; empty = no filter)"
            arg_type = String
        "--sc_lvl3_percell"
            help = "no-op (per-cell is now the SC lvl3 default); kept so existing commands still parse"
            action = :store_true
        "--sc_lvl3_pseudobulk"
            help = "pseudo-bulk SC source cells for lvl3 instead of the default per-cell inputs"
            action = :store_true
        "--data_format"
            help = "data format (ignored for SC; kept for CLI compatibility with PB sweep launchers)"
            arg_type = String
    end
    return parse_args(s)
end


end  # module Args
