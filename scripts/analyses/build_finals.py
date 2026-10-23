# build notes/finals/<ds>_{pretrained,lvl1,lvl2,lvl1_ablation,lvl2_ablation}.csv from notes/baselines.csv (+ partial-seed index)
# finals rule (user, Oct 21): LINCS lvl2 = long runs (200k, patience 10); TPB = fixed 40k (lvl1 TF 10k); TSC = 40k lvl1 / 80k lvl2
# usage: python scripts/analyses/build_finals.py [partial_runs.csv] [pretrain_losses.tsv]
import csv, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT = os.path.join(REPO, "notes", "finals")
DS = {"lincs": "lincs", "tahoe_pb": "tpb", "tahoe_sc": "tsc"}
SPLIT = {("lincs", "lvl1"): "random", ("lincs", "lvl2"): None, ("tahoe_pb", "lvl1"): "random",
         ("tahoe_pb", "lvl2"): "drug_dose", ("tahoe_sc", "lvl1"): "well_cl", ("tahoe_sc", "lvl2"): "drug_dose"}
LVL_COLS = ["dataset", "level", "split", "category", "model", "input", "gene_set", "pretrain", "checkpoint", "seed",
            "accuracy", "accuracy_fullgroup", "wellcl_acc", "wellcl_acc_fullgroup", "final_accuracy", "best_epoch",
            "best_step", "total_steps", "max_ft_steps", "patience", "lr", "drop_prob", "mlp_hidden_dim", "note", "path"]

def fnum(x):
    try:
        v = float(x); return "" if v != v else v
    except (TypeError, ValueError):
        return ""

def steps(r): return int(float(r["max_ft_steps"])) if r.get("max_ft_steps") not in ("", None) else None

def parse_tag(r):
    ds, tag = r["dataset"], r["model_tag"]
    base = tag.split("_")[0]
    gene = ("978" if ds == "lincs" else ("all" if "_full" in tag else "1024"))
    zc = "_zc" in tag or "_z" in tag.split("_gdd")[-1] and tag.endswith("_z")
    inp = "zscored" if "_zc" in tag else ("rank" if base[0] == "r" else "raw")
    if ds == "lincs" and r["level"] == "lvl2":
        split = "plate" if (r["split_tag"] == "group_ptunseen" or "_gpl" in tag) else "random"
    else:
        split = SPLIT[(ds, r["level"])]
    return base, gene, inp, split

def lvl_row(r, category, note=""):
    base, gene, inp, split = parse_tag(r)
    return {"dataset": r["dataset"], "level": r["level"], "split": split, "category": category, "model": base,
            "input": inp, "gene_set": gene, "pretrain": "none", "checkpoint": "", "seed": r["seed"],
            "accuracy": fnum(r["best_accuracy"]), "accuracy_fullgroup": fnum(r["best_accuracy_fullgroup"]),
            "wellcl_acc": fnum(r.get("best_wellcl_acc")), "wellcl_acc_fullgroup": fnum(r.get("best_wellcl_acc_fullgroup")),
            "final_accuracy": fnum(r["final_accuracy"]), "best_epoch": fnum(r["best_epoch"]), "best_step": fnum(r["best_step"]),
            "total_steps": fnum(r["total_steps"]), "max_ft_steps": fnum(r["max_ft_steps"]), "patience": fnum(r["patience"]),
            "lr": fnum(r["lr"]), "drop_prob": fnum(r["drop_prob"]), "mlp_hidden_dim": fnum(r["mlp_hidden_dim"]),
            "note": note, "path": r["path"]}

def is_final(r):
    ds, lv, st, w, tag = r["dataset"], r["level"], steps(r), fnum(r["mlp_hidden_dim"]), r["model_tag"]
    if ds == "lincs":
        return lv == "lvl1" or st == 200000
    if ds == "tahoe_pb":
        if lv == "lvl1": return True
        if st != 40000: return False
        return not (tag == "rmlp_gdd" and w == 512)        # rmlp 1024 final = 4096x1 (wide-sweep winner)
    if ds == "tahoe_sc":
        return st == (40000 if lv == "lvl1" else 80000)
    return False

def ablation_note(r):
    ds, st, w, tag, pat = r["dataset"], steps(r), fnum(r["mlp_hidden_dim"]), r["model_tag"], fnum(r["patience"])
    if ds == "tahoe_pb" and w == 16384: return "width ablation: wide sweep (16384)"
    if ds == "tahoe_pb" and tag == "rmlp_gdd" and w == 512: return "superseded: old sweep pick 512x7"
    if re.search(r"_ps[01]$", tag): return f"head-scale ablation: pool_std={tag[-1]} on npt ({st} steps)"
    if "_delta" in tag: return "input ablation: plate-matched DMSO delta (pre-protocol, partial seeds)"
    if pat: return f"convergence check: up to {st} steps, patience {int(pat)}"
    return f"convergence check: fixed {st} steps"

def trivial_rows(ds_key):
    p = {"lincs": "results/lincs/analyses/plate/baselines.csv", "tahoe_pb": "results/tahoe/pb/analyses/plate/baselines.csv",
         "tahoe_sc": "results/tahoe/sc/analyses/plate/baselines.csv"}[ds_key]
    rows = list(csv.DictReader(open(os.path.join(REPO, p))))
    out = {}
    for r in rows:
        split = {"random": "random", "group_plate": "plate", "group_drugdose": "drug_dose", "well_cl": "well_cl",
                 "drug_dose": "drug_dose"}[r["split"]]
        if ds_key == "tahoe_pb" and r["level"] == "lvl2" and split == "random": continue   # TPB lvl2 reported on drug_dose only
        for pred in ("chance", "majority_class", "plate_lookup"):
            k = (r["level"], split, pred)
            o = out.setdefault(k, {c: "" for c in LVL_COLS})
            o.update({"dataset": ds_key, "level": r["level"], "split": split, "category": "trivial", "model": pred,
                      "input": "", "gene_set": "", "pretrain": "none", "seed": "",
                      "note": f"n_classes {r['n_classes']}, n_test {r['n_test']} ({r['test_set']})" if r["test_set"] == "headline" else o["note"],
                      "path": p})
            o["accuracy" if r["test_set"] == "headline" else "accuracy_fullgroup"] = fnum(r[pred])
    return list(out.values())

def write(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")

def toml_note(cfg_file, sec):
    L = open(os.path.join(REPO, "config", cfg_file)).read().split("\n")
    for i, l in enumerate(L):
        if l.strip() == f"[{sec}]" and i + 1 < len(L) and L[i + 1].startswith("#"):
            return L[i + 1].lstrip("# ").strip()
    return ""

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(REPO, "notes", "baselines.csv"))))
    part = list(csv.DictReader(open(sys.argv[1]))) if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else []
    ptl = {}
    if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
        for l in open(sys.argv[2]):
            p, t, vlast, n = l.rstrip("\n").split("\t"); ptl[p] = (fnum(t), fnum(vlast), n)
    reg, sec = {}, None   # minimal parser (python 3.9 has no tomllib): [ds.obj] then key = "path"
    for l in open(os.path.join(REPO, "config", "pretrained_models.toml")):
        l = l.split("#")[0].strip()
        m = re.match(r"\[(\w+)\.(\w+)\]", l)
        if m: sec = m.groups(); continue
        m = re.match(r'(\w+)\s*=\s*"(.*)"', l)
        if m and sec: reg.setdefault(sec[0], {}).setdefault(sec[1], {})[m.group(1)] = m.group(2)
    triv = {}
    for k, f in (("lincs", "results/lincs/analyses/pretrain_trivial.csv"), ("tahoe_pb", "results/tahoe/pb/analyses/pretrain_trivial.csv"),
                 ("tahoe_sc", "results/tahoe/sc/analyses/pretrain_trivial.csv")):
        for r in csv.DictReader(open(os.path.join(REPO, f))):
            key = (k, r["objective"], r["arch"])
            triv.setdefault(key, {})["test" if r.get("eval_set", "test") == "test" else "val"] = fnum(r["value"])
            triv[key]["predictor"] = r["predictor"]
    for ds_key, short in DS.items():
        print(ds_key)
        smoke = lambda r: (steps(r) or 0) < 1000
        ft = [r for r in rows if r["stage"] == "finetune" and r["dataset"] == ds_key and r["mode"] == "no_pretrain" and not smoke(r)]
        pf = [r for r in part if r["dataset"] == ds_key and r["stage"] == "finetune" and not smoke(r)]
        tv = trivial_rows(ds_key)
        for lv in ("lvl1", "lvl2"):
            fin = [lvl_row(r, "npt") for r in ft if r["level"] == lv and is_final(r)]
            ab = [lvl_row(r, "npt_ablation", ablation_note(r)) for r in ft if r["level"] == lv and not is_final(r)]
            ab += [lvl_row(r, "npt_ablation", ablation_note(r) + " (partial seeds)") for r in pf if r["level"] == lv]
            key = lambda x: (x["split"] or "", x["category"], x["model"], x["input"], x["gene_set"], str(x["seed"]))
            write(f"{short}_{lv}.csv", [t for t in tv if t["level"] == lv] + sorted(fin, key=key), LVL_COLS)
            write(f"{short}_{lv}_ablation.csv", sorted(ab, key=key), LVL_COLS)
        # pretrained models
        cfg = {"lincs": "lincs.toml", "tahoe_pb": "tpb.toml", "tahoe_sc": "ts.toml"}[ds_key]
        regk = {"lincs": "lincs", "tahoe_pb": "tahoe_pb", "tahoe_sc": "tahoe_sc"}[ds_key]
        pt = {(r["mode"], r["model"]): r for r in rows if r["stage"] == "pretrain" and r["dataset"] == ds_key}
        prow = []
        for obj in ("erecon", "mlm", "lrecon"):
            for mt in ("etf", "rtf"):
                r = pt.get((obj, mt)); t = triv.get((ds_key, obj, mt), {})
                d = {"dataset": ds_key, "objective": obj, "model": mt, "status": "final" if r else "pending",
                     "best_val_loss": fnum(r["best_val_loss"]) if r else "", "final_val_loss": ptl.get(r["path"], ("", "", ""))[1] if r else "",
                     "test_loss": ptl.get(r["path"], ("", "", ""))[0] if r else "",
                     "best_epoch": fnum(r["best_epoch"]) if r else "", "total_steps": fnum(r["total_steps"]) if r else "",
                     "loss": {"erecon": "mse (masked expression)", "mlm": "cross-entropy (masked token)", "lrecon": "latent mse / target var (~1 = constant)"}[obj],
                     "trivial_val": t.get("val", ""), "trivial_test": t.get("test", ""), "trivial_predictor": t.get("predictor", ""),
                     "lr": fnum(r["lr"]) if r else "", "drop_prob": fnum(r["drop_prob"]) if r else "",
                     "finetune_checkpoint": reg.get(regk, {}).get(obj, {}).get(mt, ""),
                     "hp_source": toml_note(cfg, f"pretrain.{obj}.{mt}"), "run_path": r["path"] if r else "",
                     "note": ("lrecon test loss scores the best student vs the final (drifted) teacher; use val" if obj == "lrecon" and r else "")}
                prow.append(d)
        write(f"{short}_pretrained.csv", prow, list(prow[0].keys()))

if __name__ == "__main__":
    main()
