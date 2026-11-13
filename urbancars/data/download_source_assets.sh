#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPERIMENT_ROOT/data/env.sh"

"$EXPERIMENT_ROOT/data/prepare_workspace.sh" >/dev/null

KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
URBANCARS_EXISTING_COCO_ROOT="${URBANCARS_EXISTING_COCO_ROOT:-}"
URBANCARS_EXISTING_PLACES_ROOT="${URBANCARS_EXISTING_PLACES_ROOT:-}"

download_file() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    printf 'reuse %s\n' "$out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  wget -c "$url" -O "$out"
}

extract_tgz() {
  local archive="$1"
  local marker="$2"
  local dest="$3"
  if [[ -e "$marker" ]]; then
    printf 'reuse %s\n' "$marker"
    return
  fi
  mkdir -p "$dest"
  tar -xzf "$archive" -C "$dest"
}

extract_tar() {
  local archive="$1"
  local marker="$2"
  local dest="$3"
  if [[ -e "$marker" ]]; then
    printf 'reuse %s\n' "$marker"
    return
  fi
  mkdir -p "$dest"
  tar -xf "$archive" -C "$dest"
}

extract_zip() {
  local archive="$1"
  local marker="$2"
  local dest="$3"
  if [[ -e "$marker" ]]; then
    printf 'reuse %s\n' "$marker"
    return
  fi
  mkdir -p "$dest"
  unzip -q -n "$archive" -d "$dest"
}

cleanup_archive() {
  local archive="$1"
  if [[ "$KEEP_ARCHIVES" != "1" && -f "$archive" ]]; then
    rm -f "$archive"
  fi
}

has_valid_coco_tree() {
  local root="$1"
  [[ -d "$root/train2017" && -d "$root/val2017" ]]
}

has_valid_places_tree() {
  local root="$1"
  [[ -d "$root/train" && -d "$root/val" ]]
}

link_tree_if_missing() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ -L "$dst" ]]; then
    local current_target
    current_target="$(readlink -f "$dst")"
    if [[ "$current_target" == "$(readlink -f "$src")" ]]; then
      printf 'reuse %s -> %s\n' "$label" "$src"
      return
    fi
    echo "error: $dst already points to $current_target, not $src" >&2
    exit 1
  fi
  if [[ -e "$dst" ]]; then
    printf 'reuse existing %s at %s\n' "$label" "$dst"
    return
  fi
  ln -s "$src" "$dst"
  printf 'link %s -> %s\n' "$dst" "$src"
}

extract_places_subset_stream() {
  local url="$1"
  local target_root="$2"
  local marker="$target_root/places/train/alley"
  if [[ -d "$marker" ]]; then
    printf 'reuse %s\n' "$marker"
    return
  fi

  local tmp_extract_root="$target_root"
  local places_top="$target_root/places365_standard"
  local patterns=()
  local split
  local meta_dir="$VENDOR_ROOT/create_datasets/urbancars/metadata/places/filter_objs"
  for split in train val; do
    while IFS= read -r meta_path; do
      local category
      category="$(basename "${meta_path%.txt}")"
      patterns+=("*/$split/$category/*")
    done < <(find "$meta_dir/$split" -maxdepth 1 -type f -name '*.txt' | sort)
  done

  mkdir -p "$tmp_extract_root"
  wget -qO- "$url" | tar -xf - -C "$tmp_extract_root" --wildcards --no-anchored "${patterns[@]}"

  if [[ ! -d "$target_root/places" && -d "$places_top" ]]; then
    mv "$places_top" "$target_root/places"
  fi

  if [[ ! -d "$target_root/places/train/alley" ]]; then
    echo "error: failed to extract required Places subset into $target_root/places" >&2
    exit 1
  fi
}

clone_maskformer() {
  if [[ -d "$URBANCARS_MASKFORMER_ROOT/.git" ]]; then
    printf 'reuse %s\n' "$URBANCARS_MASKFORMER_ROOT"
    return
  fi
  rm -rf "$URBANCARS_MASKFORMER_ROOT"
  git clone https://github.com/facebookresearch/MaskFormer.git "$URBANCARS_MASKFORMER_ROOT"
}

mkdir -p "$URBANCARS_RAW_ROOT/stanford_cars/mask/cars_train"
mkdir -p "$URBANCARS_RAW_ROOT/stanford_cars/mask/cars_test"
mkdir -p "$URBANCARS_RAW_ROOT/lvis"
mkdir -p "$URBANCARS_EXP_ROOT/weights"

"$PY" "$EXPERIMENT_ROOT/data/prepare_hf_stanford_cars.py" --output-root "$URBANCARS_RAW_ROOT/stanford_cars"

clone_maskformer
download_file \
  "https://dl.fbaipublicfiles.com/maskformer/panoptic-coco/maskformer_panoptic_swin_large_IN21k_384_bs64_554k/model_final_7505c4.pkl" \
  "$URBANCARS_EXP_ROOT/weights/maskformer_panoptic_swin_large.pkl"

download_file \
  "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip" \
  "$URBANCARS_RAW_ROOT/lvis/lvis_v1_train.json.zip"
download_file \
  "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" \
  "$URBANCARS_RAW_ROOT/lvis/lvis_v1_val.json.zip"
extract_zip \
  "$URBANCARS_RAW_ROOT/lvis/lvis_v1_train.json.zip" \
  "$URBANCARS_RAW_ROOT/lvis/lvis_v1_train.json" \
  "$URBANCARS_RAW_ROOT/lvis"
extract_zip \
  "$URBANCARS_RAW_ROOT/lvis/lvis_v1_val.json.zip" \
  "$URBANCARS_RAW_ROOT/lvis/lvis_v1_val.json" \
  "$URBANCARS_RAW_ROOT/lvis"
cleanup_archive "$URBANCARS_RAW_ROOT/lvis/lvis_v1_train.json.zip"
cleanup_archive "$URBANCARS_RAW_ROOT/lvis/lvis_v1_val.json.zip"

if has_valid_coco_tree "$URBANCARS_EXISTING_COCO_ROOT"; then
  link_tree_if_missing "$URBANCARS_EXISTING_COCO_ROOT" "$URBANCARS_RAW_ROOT/coco" "COCO"
else
  mkdir -p "$URBANCARS_RAW_ROOT/coco"
  download_file "http://images.cocodataset.org/zips/train2017.zip" \
    "$URBANCARS_RAW_ROOT/coco/train2017.zip"
  download_file "http://images.cocodataset.org/zips/val2017.zip" \
    "$URBANCARS_RAW_ROOT/coco/val2017.zip"
  extract_zip \
    "$URBANCARS_RAW_ROOT/coco/train2017.zip" \
    "$URBANCARS_RAW_ROOT/coco/train2017" \
    "$URBANCARS_RAW_ROOT/coco"
  extract_zip \
    "$URBANCARS_RAW_ROOT/coco/val2017.zip" \
    "$URBANCARS_RAW_ROOT/coco/val2017" \
    "$URBANCARS_RAW_ROOT/coco"
  cleanup_archive "$URBANCARS_RAW_ROOT/coco/train2017.zip"
  cleanup_archive "$URBANCARS_RAW_ROOT/coco/val2017.zip"
fi

if has_valid_places_tree "$URBANCARS_EXISTING_PLACES_ROOT"; then
  link_tree_if_missing "$URBANCARS_EXISTING_PLACES_ROOT" "$URBANCARS_RAW_ROOT/places" "Places"
elif [[ -d "$URBANCARS_RAW_ROOT/places/train" && -d "$URBANCARS_RAW_ROOT/places/val" ]]; then
  printf 'reuse existing Places subset at %s\n' "$URBANCARS_RAW_ROOT/places"
else
  extract_places_subset_stream \
    "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar" \
    "$URBANCARS_RAW_ROOT"
fi

"$PY" "$EXPERIMENT_ROOT/data/verify_setup.py" --workspace-only
