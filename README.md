# Robbie@Home — Simulación (UruBots)

Simulación en Gazebo Harmonic + ROS 2 Jazzy del robot Robbie (Base@Home, RoboCup@Home), enfocada en la prueba **Restaurant**.

- `simulation/base_athome_sim/` — paquete ROS 2: mundos, modelos, actores (barman/clientes), puente del robot simulado. Guía completa: [`simulation/base_athome_sim/README.md`](simulation/base_athome_sim/README.md).
- `simulation/scripts/` — campañas (`restaurant_campaign.py`), chequeo de assets (`check_sim_assets.sh`) y utilidades.
- `simulation/README.md` — instalación y ejecución general.

## Requisitos

La simulación ejecuta el código real del robot (misión, percepción, brazo), que vive en el workspace `base_athome` (paquetes `base_athome_skills`, `base_athome_perception`, `base_athome_moveit_config`, ...). Clonar este repo dentro de ese workspace (`base_athome_ws/simulation`) o pedir acceso al repo del robot.

Assets que NO están en git (copiarlos o generarlos): modelos YOLO `yolov8n*.pt` y mallas de humanos de Gazebo Fuel. Ver `simulation/scripts/check_sim_assets.sh --copy-from <ruta>`.

## Modo "como el robot real"

`ATHOME_SIM_REAL_GAP=1` reproduce las condiciones observadas en el robot real: sin agarre por visión, sin docking, brazo que puede arrancar deshabilitado y ASR con ruido/eco. Detalles en el README del paquete.
