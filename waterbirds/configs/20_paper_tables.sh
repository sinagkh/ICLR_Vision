#!/usr/bin/env bash
# Writes paper_table10_11.csv (paper Tables 10 and 11) and paper_table13_exmap.csv (paper Table 13) to results/waterbirds/
# (or $RES) from the train_summary.json files of the runs of 02 (ERM), 03 (IS, manifest variant), 09 (GroupDRO),
# 11 (DINOv2 ERM and IS), 17, 18 and 19. A spec ending in :combo_checkpoint reads the run's
# steered_finetuned_best_combo.pt entry instead of the best-validation-WGA checkpoint.
set -euo pipefail
source "$(dirname "$0")/env.sh"
RES=${RES:-$PKG/../results/waterbirds}
C=$CKPT_ROOT

specs=()
add() { for s in 1 21 42; do specs+=("$1|$2|$3|$s=$C/$4/seed$s"); done; }
T=table10_11; R=ResNet-50; D="DINOv2 ViT-B/14"
add $T $R ERM erm_anchor_audit
add $T $R "Focal Loss" focal
add $T $R CVaR cvar
add $T $R "Spectral Decoupling" sd
add $T $R JTT jtt
specs+=("$T|$R|Interaction Steering|1=$C/is_manifest/seed1:combo_checkpoint")
for s in 21 42; do specs+=("$T|$R|Interaction Steering|$s=$C/is_manifest/seed$s"); done
add $T $R GroupDRO groupdro
add $T "$D" ERM dinov2_erm
add $T "$D" "Focal Loss" dinov2_focal
add $T "$D" CVaR dinov2_cvar
add $T "$D" "Spectral Decoupling" dinov2_sd
add $T "$D" JTT dinov2_jtt
add $T "$D" "Interaction Steering" dinov2_is
add $T "$D" GroupDRO dinov2_groupdro
T=table13_exmap
add $T $R ERM erm_anchor_audit
add $T $R ExMap exmap
specs+=("$T|$R|Interaction Steering|1=$C/is_manifest/seed1:combo_checkpoint")
for s in 21 42; do specs+=("$T|$R|Interaction Steering|$s=$C/is_manifest/seed$s"); done

$PY $PKG/eval/write_paper_tables.py "$RES" "${specs[@]}"
