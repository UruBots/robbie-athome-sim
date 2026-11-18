"""Modo "como el robot real" (ATHOME_SIM_REAL_GAP=1) de la simulacion Restaurant.

La simulacion por defecto es mas amable que el robot: docking y agarre por vision
encendidos, ASR perfecto, brazo siempre habilitado y un LiDAR trasero que el robot
no tiene. Con ``ATHOME_SIM_REAL_GAP=1`` se reproduce lo que paso en el robot real el
17/11 (ver ``simulation/base_athome_sim/README.md``):

* ``VISUAL_PICK=0`` y ``RESTAURANT_DOCK=0`` por defecto (el barman pone el item en la pinza);
* el actor responde con ASR ruidoso: "Road yes", "A reward? Yes.", "A Coke.", silencios
  y el eco de la propia voz del robot;
* el brazo simulado arranca deshabilitado (``enabled=false``) hasta ``/arm/set_enable``;
* no se reenvia el scan trasero (``/restaurant/rear_scan``).

Funciones puras y sin ROS para poder probarlas sin Gazebo.
"""
import random

FLAG = "ATHOME_SIM_REAL_GAP"

# Probabilidades por defecto; ATHOME_SIM_REAL_GAP_ASR=0 apaga solo el ruido del ASR.
SILENCE_P = 0.10
ECHO_P = 0.08
MISHEAR_P = 0.35
PARTIAL_ORDER_P = 0.15
FOREIGN_P = 0.06
# Cuantas ventanas seguidas calla el barman antes de confirmar (19:49 del 17/11: 6 ventanas sin voz).
BARMAN_SILENT_WINDOWS = 2
ECHO_DELAY_S = (0.3, 1.0)

# Transcripciones ajenas oidas por el robot real en la ventana de escucha (logs 17/11).
FOREIGN_PHRASES = ("Who are you name?", "I want to tell you.", "Next stop.", "Okay.", "Nice.",
                   "The table.", "I need help.", "Are you ready to go?")

_YES_VARIANTS = ("road yes", "a reward? yes.", "yes", "robot yes.", "roberto yes")
_NO_VARIANTS = ("road no", "a robot, no.", "no", "robot no.", "roberto no")


def real_gap_enabled(environ):
    """True solo con ATHOME_SIM_REAL_GAP=1 (cualquier otro valor deja la sim como antes)."""
    return str(environ.get(FLAG, "0")).strip() == "1"


def asr_noise_enabled(environ):
    return real_gap_enabled(environ) and str(environ.get(FLAG + "_ASR", "1")).strip() != "0"


def arm_starts_disabled(environ):
    return real_gap_enabled(environ) and str(environ.get(FLAG + "_ARM", "1")).strip() != "0"


def rear_scan_blocked(environ):
    return real_gap_enabled(environ)


def manipulation_defaults(environ):
    """Valores por defecto de VISUAL_PICK / RESTAURANT_DOCK ("0" o "1"), respetando lo explicito."""
    gap = real_gap_enabled(environ)
    default = "0" if gap else "1"
    return {
        "VISUAL_PICK": environ.get("VISUAL_PICK", default),
        "RESTAURANT_DOCK": environ.get("RESTAURANT_DOCK", default),
    }


def noisy_reply(answer, rng=None, *, allow_silence=True, last_robot_say=""):
    """Devuelve lo que el ASR real le entregaria al robot, o None si hubo silencio.

    ``answer`` es la respuesta limpia del actor ("robot yes", "my order is coke and water").
    Nunca convierte un si en un no ni al reves: el ruido solo cambia la forma de la frase,
    la pierde (silencio) o la reemplaza por el eco de la voz del robot.
    """
    rng = rng or random
    roll = rng.random()
    if allow_silence and roll < SILENCE_P:
        return None
    if allow_silence and last_robot_say and roll < SILENCE_P + ECHO_P:
        return last_robot_say
    if allow_silence and roll < SILENCE_P + ECHO_P + FOREIGN_P:
        return rng.choice(FOREIGN_PHRASES)
    text = (answer or "").strip()
    low = text.lower()
    if low in ("robot yes", "yes"):
        return rng.choice(_YES_VARIANTS) if rng.random() < MISHEAR_P else text
    if low in ("robot no", "no"):
        return rng.choice(_NO_VARIANTS) if rng.random() < MISHEAR_P else text
    if low.startswith("my order is "):
        items = low[len("my order is "):]
        first = items.split(" and ")[0].strip()
        if rng.random() < PARTIAL_ORDER_P and " and " in items:
            # Formas reales: "A Coke.", "Coke!", "My order is a goak.", "Order men."
            heard = first if first != "coke" else rng.choice(("coke", "goak"))
            return rng.choice(("a " + first + ".", first.capitalize() + "!",
                               "my order is a " + heard + ".", "order men."))
        if rng.random() < MISHEAR_P:
            return items.replace(" and ", ". And ", 1).capitalize()
    return text


def echo_delay_s(rng=None):
    """Retardo con que el eco del TTS llega al ASR."""
    return (rng or random).uniform(*ECHO_DELAY_S)


def barman_silent_windows(environ):
    """Ventanas de confirmacion en las que el barman calla (0 fuera del modo real-gap)."""
    if not asr_noise_enabled(environ):
        return 0
    try:
        return max(0, int(environ.get(FLAG + "_BARMAN_SILENT", BARMAN_SILENT_WINDOWS)))
    except ValueError:
        return BARMAN_SILENT_WINDOWS
