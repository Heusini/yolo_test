# Model registry

State of 2026-11-18. Weights on mars: `/home/sheusinger/repos/yolo_test/runs/detect/yolo/<run>/weights/best.pt`
(`best.pt` = epoch with the best mAP50-95, or the metric chosen with `--select` for newer runs).
Scores: SkyEV val at 1280x736, 33 ms event window, mAP50 / mAP50-95 at the best epoch, unless noted.
Rating 1-10 = clean accuracy + robustness table (`test/robustness_eval.py`) + fit for the Orin deployment.
Typst version: `typst_tables/model_registry.typ` (outer repo).

## Current candidates

| rating | model | run folder | mAP50 / 50-95 | note |
|---|---|---|---|---|
| 9 | DualStem, zoom recipe (deployment) | `eventrgb_dualstem_mildaug_shift_zoom1280_yolo26n` | .954 / .581 | best clean score and robustness table; TensorRT engine, 23 h on the Orin |
| 9 | DualStem, mixed SkyEV+FRED (deployment candidate) | `eventrgb_dualstem_mixed_zoom1280_yolo26n2` | .952 / .572 | 0.9 behind the zoom model on SkyEV, 1-2 ahead on misaligned/stale RGB; on FRED val .518 (specialist .528). One model for both domains, still rising at epoch 12 (best = ep 12). Not yet exported to TensorRT |
| 9 | DualStem, mixed data, center recipe (deployment candidate, presence first) | `eventrgb_dualstem_mixed_center1280_yolo26n` | .981 / .660 on 24 px grown labels | zoom recipe + `--min-box 24 --box 3 --cls 1.5 --dfl 0.5 --select center`, 12 epochs, best = ep 12. Labels grown to >= 24 px in train and val, so mAP is only comparable at equal label size (table below). cAP, which ignores box size, is 0.2-1.6 above the mixed model in every condition, most where a sensor is missing. Price: boxes are ~24 px, raw-label mAP50-95 .343. Pick this when presence and position matter, the mixed model when tight boxes matter |
| 8 | DualStem, shift recipe (previous deployment) | `eventrgb_dualstem_mildaug_shift1280_yolo26n` | .948 / .559 | recipe before zoom; 2 points behind on clean and on small drones |
| 7 | DualStem, mild aug | `eventrgb_dualstem_mildaug1280_yolo26n` | .942 / .556 | keeps .38 with RGB off; no shift training, 4 px misalignment costs 12 points |

## Tried and shelved

| rating | model | run folder | mAP50 / 50-95 | note |
|---|---|---|---|---|
| 6 | two-tower, zoom recipe | `eventrgb_twotower_mildaug_shift_zoom_tower1280_yolo26n` | .936 / .572 | every robustness row 0.6-1.6 below the zoom dual stem, 1.6x params, overfits from epoch 3 |
| 6 | DualStem, plain | `eventrgb_dualstem_plain1280_yolo26n` (unconfirmed) | .941 / .563 | .20 with RGB off, 7-11 points lost under light changes |
| 6 | DualStem, drop-rgb 0.3 | `eventrgb_dualstem_mildaug_shift_drop30_1280_yolo26n` (unconfirmed) | .936 / .549 | +2.6 with RGB off, -1 everywhere else |
| 5 | DualStem, two-phase | `eventrgb_dualstem_2phaseB_1280_yolo26n` | .951 / .556 | fragile: .10 with RGB off |
| 5 | ConvLSTM, frozen trunk, clip 12 | `eventrgb_dualstem_lstmfrz12_1280_yolo26n` | .940 / .554 | robustness identical to stateless; block stayed at identity |
| 5 | ConvLSTM, frozen trunk, clip 4 | `eventrgb_dualstem_lstmfrz1280_yolo26n` | .938 / .552 | same |
| 5 | ConvLSTM, full fine-tune | `eventrgb_dualstem_lstm1280_yolo26n2` | .935 / .544 | 1 point below stateless, extra state on the Orin |
| 5 | plain 13-channel YOLO26n | `eventrgb_baseline_plain1280_yolo26n` | .930 / .541 | reference the dual stem beats by 2.2 |

## References: single modality, other domain

| rating | model | run folder | mAP50 / 50-95 | note |
|---|---|---|---|---|
| 4 | event only, mild aug | `eventonly_mildaug1280_yolo26n` (unconfirmed) | .907 / .472 | recall .88, loose boxes; fallback if an RGB camera fails |
| 3 | RGB only, mild aug | `rgbonly_mildaug1280_yolo26n2` (suffix unconfirmed) | .837 / .509 | recall .80, misses the small drones |
| 7 on FRED / 1 on SkyEV | DualStem, FRED only | `eventrgb_dualstem_fred_zoom1280_yolo26n` | .917 / .528 on FRED val | beats the FRED paper by 4 mAP50; effectively event-only; zero-shot on SkyEV .125 / .028 |

## Old 640x384 runs (33 ms window, superseded by the 1280 export)

| rating | model | run folder | mAP50 / 50-95 | note |
|---|---|---|---|---|
| 3 | DualStem | `eventrgb_dualstem_yolo26n` | .930 / .532 | |
| 3 | DualStem, P2 head | `eventrgb_dualstem_p2_plain_yolo26n` | .930 / .531 | 2.7x slower epochs, no gain |
| 2 | plain 13-channel | `eventrgb_yolo26n` | .920 / .510 | |
| 2 | DualStem, full aug | `eventrgb_dualstem_aug_yolo26n` | .892 / .470 | 8 px RGB shift hurt at this resolution |

## Robustness reference rows (SkyEV val, mAP50-95)

| condition | zoom | mixed | mixed cAP |
|---|---|---|---|
| clean | .580 | .571 | .850 |
| rgb_zero | .384 | .406 | .576 |
| evt_zero | .481 | .482 | .698 |
| rgb_dark / bright / noise | .555 / .563 / .565 | .539 / .559 / .559 | .811 / .842 / .841 |
| rgb_shift2 / 4 / 8 | .545 / .476 / .331 | .540 / .479 / .344 | .839 / .812 / .774 |
| rgb_prev | .434 | .453 | .817 |
| evt_half / double / noise / drop | .557 / .579 / .584 / .578 | .536 / .569 / .564 / .569 | .801 / .858 / .835 / .842 |
| evt_shift1-5 | .581 .576 .569 .561 .553 | .571 .566 .558 .549 .539 | .851 .850 .844 .843 .836 |

cAP in this table was computed before commit 87a5b1a (closest center took the hit regardless of confidence, so junk duplicates stole hits): biased low and noisy. Corrected values for the mixed model in the next table.

## Center recipe vs mixed, SkyEV val, after the cAP fix (87a5b1a)

Each model wins mAP50-95 on its own label definition, so the box score only says which labels were used. cAP is almost identical on both label definitions and is the fair comparison.

| condition | mAP raw labels center / mixed | mAP 24 px labels center / mixed | cAP raw labels center / mixed | cAP 24 px labels center / mixed |
|---|---|---|---|---|
| clean | .343 / .571 | .659 / .402 | .985 / .981 | .985 / .981 |
| rgb_zero | .275 / .406 | .498 / .323 | .965 / .953 | .967 / .955 |
| evt_zero | .261 / .482 | .543 / .308 | .852 / .836 | .853 / .837 |
| rgb_shift8 | .231 / .344 | .385 / .274 | .977 / .975 | .986 / .981 |
| rgb_prev | .282 / .453 | .531 / .333 | .974 / .970 | .983 / .977 |
| evt_half | .312 / .536 | .626 / .359 | .962 / .949 | .962 / .949 |

Reading: with correct matching, the mixed model's cAP is .98 on clean and still .975 under an 8 px RGB shift, so the robustness drops in mAP were mostly box precision, not missed drones. The only condition that costs detections is events off (cAP .84-.85).

FRED val, mAP50-95 (cAP): FRED-only model clean .528, rgb_zero .505, evt_zero .149, evt_shift1-5 .322/.230/.168/.112/.039; mixed model clean .518 (.712), rgb_zero .499 (.660), evt_zero .137 (.090). The gate leans on events for FRED frames and on both for SkyEV frames in the same network.

## Recipes

- zoom recipe: `--flip 0.5 --photo 0.5 --blur 0 --exposure 0.7 1.4 --shift 2 1 --drop-rgb 0.1 --drop-evt 0.05 --zoom 0.5 1.5 --zoom-out 1.2`, imgsz 1280, from `yolo26n.pt`
- center recipe: zoom recipe + `--min-box 24 --box 3 --cls 1.5 --dfl 0.5 --select center` (labels grown to >= 24 px in train and val; score with `test/robustness_eval.py --min-box 24` and without)
- shift recipe: same without `--zoom --zoom-out`; mild aug: same without `--shift`
- mixed data: `utils/make_mixed_dataset.py` -> `conf/mixed_data.yaml` (val = SkyEV val); FRED: `conf/fred_data.yaml`; SkyEV 1280 yaml on mars: `conf/eventrgb_720_33ms_data.yaml`
