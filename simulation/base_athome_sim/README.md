# base_athome_sim

Simulacion de Base@Home en ROS 2 Jazzy + Gazebo Harmonic. Reemplaza a la Jetson (brazo, pan-tilt, microfono,
camaras) y al hardware de base con las **mismas interfaces ROS 2**, asi que el codigo del robot (`src/`) corre sin
cambios. Esta pensado para la prueba **Restaurant**, pero tambien sirve la Receptionist y la mision de la basura.
Guia general de la carpeta: `simulation/README.md`.

> Nunca uses el dominio del robot (`ROS_DOMAIN_ID=10`) ni `0`. La sim corre en `ROS_DOMAIN_ID=42`.

## 1. Requisitos

| Que | Detalle |
|---|---|
| SO / ROS | Ubuntu 24.04, ROS 2 Jazzy (`source /opt/ros/jazzy/setup.bash`) |
| Gazebo | Harmonic (`gz sim`), `ros-jazzy-ros-gz`, `ros-jazzy-nav2-bringup`, `ros-jazzy-moveit` |
| Dependencias | `rosdep install --from-paths src simulation --ignore-src -y` |
| Venv de vision | `~/ros2_vision_venv` (opencv + ultralytics): solo para el detector YOLO de muneca y el benchmark de percepcion. **No exportes modelos con ese venv**: ultralytics instala numpy 2 y rompe el cv2 del sistema. |
| Modelos de vision | **No estan en git.** `src/base_athome_perception/models/`: `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`, `face_recognition_sface_2021dec.onnx`, `face_detection_yunet_2023mar.onnx`, `yolov8n.pt`, `yolov8n-pose.pt` (opcional `yolov8n-seg.pt`) |
| Mallas de humanos | `models/humans/male_visitor`, `female_visitor`, `nurse_female` **si estan en git** (Restaurant). Las cuatro de la Receptionist (`FemaleVisitor`, `FemaleVisitorSit`, `MaleVisitorOnPhone`, `MaleVisitorSit`) **no** estan en git |

De donde sacar lo que no esta en git:

1. Copiarlo de un workspace que ya lo tenga (lo mas rapido):
   `simulation/scripts/check_sim_assets.sh --copy-from <ruta_al_workspace>` (solo copia lo que falta).
2. Modelos de vision: `src/base_athome_perception/scripts/download_models.sh` (caffemodel, SFace, pose); los
   `yolov8n*.pt` se bajan de `https://github.com/ultralytics/assets/releases/download/v8.3.0/`.
3. Mallas de la Receptionist (CC BY 4.0, Open Robotics / Fuel): `python3 -m base_athome_sim.receptionist_humans --fetch`.

Para verificar todo de una vez:

```bash
simulation/scripts/check_sim_assets.sh      # exit 0 = estan los obligatorios
```

## 2. Compilar

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to base_athome_sim base_athome_skills base_athome_perception base_athome_moveit_config
source install/setup.bash
```

`setup.py` instala `launch/`, `worlds/*.sdf`, `config/`, `rviz/` y **todas** las carpetas de `models/humans/`.

## 3. Lanzar la Restaurant

Camino recomendado, el de los companeros: la campana levanta actor, Gazebo, Nav2, SLAM y el nodo Restaurant,
despausa el mundo, graba evidencia y deja `summary.json`:

```bash
SIM_DOMAIN=42 python3 simulation/scripts/restaurant_campaign.py --mode real --runs 1 --tag mi_prueba
```

`--mode real` usa percepcion real (camara de Gazebo + YOLO/caras); `--mode projected` usa posiciones proyectadas.
Las corridas dejan sus logs y `summary.json` en `~/sim_lab/logs/` (`SIM_LAB_DIR` cambia la carpeta). La campana
pide el turno de Gazebo con `~/sim_lab/tools/gz_turno.sh` (una sola simulacion a la vez en el equipo). Con `GZ_GUI=0`
corre sin ventana.

A mano (dos terminales con `ROS_DOMAIN_ID=42` y el workspace sourceado):

```bash
python3 simulation/scripts/restaurant_actor.py &                     # clientes y barman con guion
SIM=1 AUTO=1 RVIZ=0 src/base_athome_skills/scripts/restaurant.sh     # Gazebo + Nav2 + SLAM + restaurant_node
```

Variables que importan:

| Variable | Efecto |
|---|---|
| `SIM_DOMAIN` | Dominio ROS aislado de la sim (por defecto 42; el validador rechaza 0 y 10) |
| `VISUAL_PICK` | `1` = el robot agarra el item con vision; `0` = el barman lo pone en la pinza con **doble confirmacion** (asi corre en el real). Por defecto `1` en sim, `0` en el robot real |
| `RESTAURANT_DOCK` | `1` = docking contra la barra con el scan trasero simulado; `0` apagado. En el robot real siempre queda apagado (no hay scan trasero) |
| `ARM` | `1` = brazo simulado con movimiento (por defecto en sim) |
| `ATHOME_SIM_ARENA=1` | Comedor de la casa completa (`worlds/arena.sdf`) en lugar de `restaurant.sdf`, SLAM sin mapa previo |
| `GZ_GUI`, `RVIZ` | Ventana de Gazebo / RViz (`0` apaga) |
| `RESTAURANT_LOG_DIR` | Carpeta de `states.jsonl`, `run.log` y evidencia |

### Modo "como el robot real": `ATHOME_SIM_REAL_GAP=1`

La sim por defecto es mas amable que el robot. Con `ATHOME_SIM_REAL_GAP=1` reproduce lo que paso en el real el 17/11:

- `VISUAL_PICK=0` y `RESTAURANT_DOCK=0` por defecto (un valor explicito los vuelve a prender);
- el actor responde con ASR ruidoso: "Road yes", "A reward? Yes.", "A Coke.", "Coke!", "goak", "Order men.", silencios,
  el eco de la propia voz del robot (con retardo de 0.3 a 1.0 s) y voces ajenas ("Who are you name?"); nunca cambia un si
  por un no. El barman calla las primeras 2 ventanas de confirmacion (`ATHOME_SIM_REAL_GAP_BARMAN_SILENT=<n>`);
- el brazo simulado arranca sin torque (`enabled=false`) y la capa de manipulacion debe habilitarlo por `/arm/set_enable`;
- no se reenvia el scan trasero (`/restaurant/rear_scan`): si se fuerza `RESTAURANT_DOCK=1`, el docking responde `no_scan`.

Pendiente de modelar: joint_states viejos y mano a media altura / dos manos arriba.

```bash
ATHOME_SIM_REAL_GAP=1 SIM_DOMAIN=42 python3 simulation/scripts/restaurant_campaign.py --mode real --runs 1 --tag real_gap_1
```

Ajustes finos: `ATHOME_SIM_REAL_GAP_ASR=0` apaga solo el ruido del ASR; `ATHOME_SIM_REAL_GAP_ARM=0` deja el brazo
habilitado desde el inicio; `ATHOME_SIM_REAL_GAP_SEED=<n>` fija la semilla del ruido para repetir una corrida.
Codigo: `base_athome_sim/real_gap.py`.

## 4. Estructura

```
simulation/base_athome_sim/
  base_athome_sim/        sim_robot.py (nodo que reemplaza a la Jetson), sim_core.py (brazo/pan-tilt/microfono),
                          gz_model.py (spawn/pose de modelos en Gazebo), sim_scene.py, counter_carry.py,
                          measured_arm_feedback.py, restaurant_humans.py + restaurant_actor_protocol.py (logica de
                          clientes y barman), receptionist_humans.py (humanos de la Receptionist), real_gap.py
  launch/                 sim.launch.py (robot + puentes), nav2_sim.launch.py
  worlds/                 restaurant.sdf, arena.sdf, receptionist_humans.sdf, casa_*.sdf, nav_eval.sdf
  config/                 mapas semanticos, restaurant_arm_poses.yaml, parametros Nav2 de evaluacion
  models/humans/          mallas de personas (ver requisitos)
  rviz/  test/  resource/
simulation/scripts/       restaurant_campaign.py, restaurant_actor.py, receptionist_campaign.py, sim_*.sh,
                          check_sim_assets.sh, hacer_mapa.py
simulation/config/        mapas (.pgm/.yaml) de las casas
```

Actores de la Restaurant (`scripts/restaurant_actor.py`): el **barman** (en el mostrador, entrega latas/items y confirma
por voz) y dos **comensales** en mesas; contestan en `/sim/speech` cuando el robot abre `/voice/listen_request`.
Mundos: `restaurant.sdf` (paso fisico de 10 ms; no valida contacto fino de manipulacion), `arena.sdf` (casa completa),
`receptionist_humans.sdf`, `casa_basura.sdf`, `casa_pick_place.sdf`, `casa_simple.sdf`, `nav_eval.sdf`.

## 5. Tests

Sin Gazebo, rapido (usa un dominio de laboratorio, nunca 0 ni 10):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ROS_DOMAIN_ID=42 python3 -m pytest -q simulation/base_athome_sim/test
ROS_DOMAIN_ID=42 python3 -m pytest -q src/base_athome_skills/test -k restaurant
```

## 6. Errores frecuentes

| Sintoma | Causa / arreglo |
|---|---|
| `ImportError: cannot import name 'real_gap'` o cambios que no se ven | El `PYTHONPATH` apunta a otro workspace (p. ej. el del robot). Abri una terminal limpia, `source` solo este `install/` |
| La campana sale con codigo 4 / falta un modelo | Faltan assets no versionados: `simulation/scripts/check_sim_assets.sh` |
| `numpy`/`cv2` rotos tras exportar un modelo | Se uso `~/ros2_vision_venv` para exportar. No lo hagas; reinstala numpy<2 en el venv |
| Gazebo arranca pausado y nada se mueve | La campana despausa a los 10 s; a mano: `gz service -s /world/restaurant/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 2000 --req 'pause: false'` |
| `validate_lab_domain` rechaza el dominio | Usaste 0 o 10 (robot real). Usa 42 o 77 |
| Dos simulaciones se pisan | Una sola sim por equipo; usa un `GZ_PARTITION` distinto por corrida (la campana ya lo hace) |
| El brazo no se mueve | Falta `ARM=1` / `ATHOME_SIM_ARM_MOTION=1`, o MoveIt no arranco (la campana cae al camino de bandeja) |
| La Receptionist no muestra humanos texturizados | Faltan las mallas Fuel: `python3 -m base_athome_sim.receptionist_humans --fetch` |
