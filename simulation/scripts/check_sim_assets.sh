#!/bin/bash
# Verifica que esten los assets NO versionados que necesita la simulacion Restaurant.
# Uso:  simulation/scripts/check_sim_assets.sh [--copy-from <worktree_o_workspace_con_los_assets>]
# Salida 0 = todo lo obligatorio esta; 1 = falta algo (se imprime de donde copiarlo).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
MODELS="$WT/src/base_athome_perception/models"
HUMANS="$WT/simulation/base_athome_sim/models/humans"
SRC=""
if [ "${1:-}" = "--copy-from" ]; then SRC="${2:-}"; [ -d "$SRC" ] || { echo "--copy-from necesita una carpeta existente" >&2; exit 2; }; fi
missing=0; warn=0

copy_if_missing() {  # $1 = ruta relativa al repo
  [ -n "$SRC" ] && [ -s "$SRC/$1" ] && mkdir -p "$(dirname "$WT/$1")" && cp --update=none "$SRC/$1" "$WT/$1"
}
need() {  # archivo obligatorio
  local rel=${1#"$WT/"}
  [ -s "$1" ] || copy_if_missing "$rel"
  if [ -s "$1" ]; then echo "  OK    $rel"; else echo "  FALTA $rel"; missing=1; fi
}
want() {  # archivo opcional
  local rel=${1#"$WT/"}
  [ -s "$1" ] || copy_if_missing "$rel"
  if [ -s "$1" ]; then echo "  OK    $rel"; else echo "  (opc) $rel"; warn=1; fi
}

echo "[1/4] Herramientas"
for cmd in gz ros2 colcon python3; do
  if command -v "$cmd" >/dev/null 2>&1; then echo "  OK    $cmd"
  else echo "  FALTA $cmd (source /opt/ros/jazzy/setup.bash; sudo apt install ros-jazzy-ros-gz)"; missing=1; fi
done
[ -x "$HOME/ros2_vision_venv/bin/python3" ] && echo "  OK    ~/ros2_vision_venv" || { echo "  (opc) ~/ros2_vision_venv (solo para el benchmark de percepcion y el detector YOLO de muneca)"; warn=1; }

echo "[2/4] Modelos de vision (src/base_athome_perception/models)"
need "$MODELS/deploy.prototxt"
need "$MODELS/res10_300x300_ssd_iter_140000.caffemodel"
need "$MODELS/face_recognition_sface_2021dec.onnx"
need "$MODELS/face_detection_yunet_2023mar.onnx"
need "$MODELS/yolov8n.pt"
need "$MODELS/yolov8n-pose.pt"
want "$MODELS/yolov8n-seg.pt"

echo "[3/4] Humanos de la Restaurant (versionados en git)"
for m in male_visitor female_visitor nurse_female; do need "$HUMANS/$m/model.sdf"; done

echo "[4/4] Mallas Fuel de la Receptionist (no versionadas)"
for m in FemaleVisitor FemaleVisitorSit MaleVisitorOnPhone MaleVisitorSit; do want "$HUMANS/$m/model.sdf"; done

if [ "$missing" = 1 ]; then
  cat <<MSG

Faltan assets obligatorios. Opciones:
  - copiarlos desde un workspace que ya los tenga:  $0 --copy-from <ruta_al_workspace>
  - modelos de vision:  src/base_athome_perception/scripts/download_models.sh
  - yolov8n*.pt:  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt (y yolov8n-pose.pt)
MSG
  exit 1
fi
[ "$warn" = 1 ] && echo "
Opcionales ausentes: Receptionist con humanos texturizados -> python3 -m base_athome_sim.receptionist_humans --fetch"
echo "Assets obligatorios completos."
exit 0
