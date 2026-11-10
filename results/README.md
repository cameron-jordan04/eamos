# results/

Small JSON summaries of the runs behind the write-up. The folder names are the GPU stage names (`runs/<stage>/` in the
pipeline). The per-sample records (`*.jsonl`, the model traces) are not included.

| Folder | What it is |
|---|---|
| `gap_target.json` | G\* = 0.379, the SDF final organism's val gap (4 samples × 128 tasks), and the earlier 1-sample value |
| `sdf_docs/{v1,v2,stage2}` | Document filter reports and the stage-2 mix metadata (sha256 of the training file) |
| `sdf_eval*`, `baseline_d13`, `probe_trigger_r3` | SDF val evaluations, the base-model baseline with starter annotations stripped, and the final cue-name probe |
| `ei_both_sides`, `ei_both_sides_r2` | Two-sided EI rounds 1 and 2: `ei_log.json`, per-temperature pass counts, val summaries, checkpoint gaps |
| `ei_eval_only*` | The eval-only EI variant (not carried forward) |
| `reeval_gstar` | 4-sample val re-evaluation of SDF final and SDF+EI checkpoints 10/20/25 (selection of SDF+EI) |
| `demo_data/{v6,v8}`, `demo_v8_match`, `demo_reeval` | Demo training-set stats, checkpoint sweep (1 sample), 4-sample re-evaluation of checkpoint-50 |
| `grid_*`, `demo_grid_*`, `grid_extra` | Test-grid summaries per organism: none, no_cot, deploy claims, eval_control, base model; `pass/` = BigCodeBench pass rates |
| `steer_*` | Steering calibration (`cal/calibration.json`) and test summaries; `_s04` = fixed scale 0.4 |
| `pass_steer_{1,2}` | Pass rates of steered outputs |
| `sdf_hf_baseline`, `recap_8192` | HF-vs-vLLM backend check for unsteered SDF; token-cap re-scoring |
| `judge*` | Judge pilot (κ per candidate) and hand labels, per-organism checks, verbalization summaries |
| `grid_summary` | Cross-organism test cells, criteria checks, full SDF steering (scale 0.6) |
| `analysis` | Between-organism comparisons (`53_compare_organisms.py`); `*_hintfrac` uses the fraction-annotated metric |
| `steering_vectors` | The organism-derived steering vectors (`sdf.pt`, `sdf_ei.pt`, `demo.pt`) |
