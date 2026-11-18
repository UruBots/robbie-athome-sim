"""Unit tests for Smart Seat Detector with Virtual Chairs (inspired by schenker332)."""
import math
from types import SimpleNamespace
import pytest

from base_athome_perception.seat_occupancy import (
    VirtualChair,
    is_sitting_posture,
    smart_seat_detector,
)


@pytest.fixture
def physical_seats():
    return {
        "sofa_host": {
            "name": "Host sofa",
            "x": 3.5,
            "y": 0.0,
            "capacity": 1,
            "polygon": [[3.2, -0.4], [3.8, -0.4], [3.8, 0.4], [3.2, 0.4]],
        },
        "chair_left": {
            "name": "Chair north / left",
            "x": 3.5,
            "y": 1.5,
            "capacity": 1,
            "polygon": [[3.25, 1.25], [3.75, 1.25], [3.75, 1.75], [3.25, 1.75]],
        },
        "chair_right": {
            "name": "Chair south / right",
            "x": 3.5,
            "y": -1.5,
            "capacity": 1,
            "polygon": [[3.25, -1.75], [3.75, -1.75], [3.75, -1.25], [3.25, -1.25]],
        },
    }


def test_empty_chair_is_free(physical_seats):
    # Only guest chairs considered for guests, host default at (3.5, 0.0)
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    res = smart_seat_detector(guest_seats, detected_persons=[], host_xy=(3.5, 0.0))

    assert "chair_left" in res["free_seats"]
    assert "chair_right" in res["free_seats"]
    assert res["occupied_seats"]["chair_left"] == 0
    assert res["occupied_seats"]["chair_right"] == 0
    assert res["chosen_seat"] in ("chair_left", "chair_right")


def test_chair_occupied_by_sitting_person_is_suppressed(physical_seats):
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    # Person sitting on chair_left
    persons = [
        {"xy": (3.5, 1.5), "posture": "SENTADO", "name": "John"}
    ]
    res = smart_seat_detector(guest_seats, detected_persons=persons, host_xy=(3.5, 0.0))

    assert "chair_left" not in res["free_seats"]
    assert "chair_right" in res["free_seats"]
    assert res["occupied_seats"]["chair_left"] == 1
    assert res["occupied_seats"]["chair_right"] == 0
    assert res["occupants"].get("chair_left") == "John"
    assert res["chosen_seat"] == "chair_right"


def test_person_standing_next_to_chair_does_not_occupy_it(physical_seats):
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    # Person standing right next to chair_left (x=3.5, y=1.4)
    persons = [
        {"xy": (3.5, 1.4), "posture": "DE_PIE", "name": "VisitorStanding"}
    ]
    res = smart_seat_detector(guest_seats, detected_persons=persons, host_xy=(3.5, 0.0))

    assert "chair_left" in res["free_seats"]
    assert "chair_right" in res["free_seats"]
    assert res["occupied_seats"]["chair_left"] == 0
    assert res["occupied_seats"]["chair_right"] == 0


def test_two_chairs_with_one_guest_already_seated(physical_seats):
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    # Guest 1 seated on chair_right
    persons = [
        {"xy": (3.5, -1.5), "posture": "SITTING", "name": "John"}
    ]
    res = smart_seat_detector(guest_seats, detected_persons=persons, host_xy=(3.5, 0.0))

    assert res["free_seats"] == ["chair_left"]
    assert res["chosen_seat"] == "chair_left"
    assert res["occupants"]["chair_right"] == "John"


def test_seat_swap_rulebook_falla_b(physical_seats):
    """RuleBook Falla B: Charlie and John swap seats.

    Biometric recognition identifies Charlie at chair_right and John at sofa_host.
    The detector must assign occupants according to face biometrics, not nominal coordinates.
    """
    detected_persons = [
        {"xy": (3.5, -1.5), "posture": "SENTADO", "name": "Charlie"},
        {"xy": (3.5, 0.0), "posture": "SENTADO", "name": "John"},
    ]
    res = smart_seat_detector(physical_seats, detected_persons=detected_persons, host_xy=(3.5, 0.0))

    assert res["occupants"].get("chair_right") == "Charlie"
    assert res["occupants"].get("sofa_host") == "John"
    assert "chair_left" in res["free_seats"]
    assert res["chosen_seat"] == "chair_left"


def test_is_sitting_posture_helper():
    assert is_sitting_posture("SENTADO")
    assert is_sitting_posture("seated")
    assert is_sitting_posture("sitting")
    assert is_sitting_posture("SENTADA")
    assert not is_sitting_posture("DE_PIE")
    assert not is_sitting_posture("standing")
    assert not is_sitting_posture(None)


def test_smart_seat_detector_fail_closed_missing_chair_xy(physical_seats):
    """Fail-closed: las sillas que carecen de coordenadas espaciales (x, y o polígono) no se ofrecen jamás."""
    from base_athome_perception.seat_occupancy import seat_has_xy, choose_free_seat

    seats_with_unlocated = {
        "chair_left": physical_seats["chair_left"],
        "chair_phantom": {"name": "Phantom chair", "capacity": 1},  # Sin x, y ni polígono
        "chair_null_coords": {"name": "Null chair", "x": None, "y": None, "capacity": 1},
    }

    assert seat_has_xy(seats_with_unlocated["chair_left"]) is True
    assert seat_has_xy(seats_with_unlocated["chair_phantom"]) is False
    assert seat_has_xy(seats_with_unlocated["chair_null_coords"]) is False

    # Ocupamos chair_left; solo quedan sillas sin xy
    persons = [{"xy": (3.5, 1.5), "posture": "SENTADO", "name": "John"}]
    res = smart_seat_detector(seats_with_unlocated, detected_persons=persons, host_xy=(3.5, 0.0))

    # Ninguna silla sin coordenadas puede estar en free_seats ni ser chosen_seat
    assert "chair_phantom" not in res["free_seats"]
    assert "chair_null_coords" not in res["free_seats"]
    assert res["free_seats"] == []
    assert res["chosen_seat"] is None
    # choose_free_seat tampoco debe elegir sillas sin coordenadas
    assert choose_free_seat(seats_with_unlocated, {"chair_left": 1}) is None


def test_smart_seat_detector_respects_reservations(physical_seats):
    """Verifica que el detector respete reservas de invitados previos (fail-closed)."""
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    # chair_left reservada para Guest 1
    res = smart_seat_detector(
        guest_seats,
        detected_persons=[],
        host_xy=(3.5, 0.0),
        reserved={"chair_left": 1},
    )

    assert "chair_left" not in res["free_seats"]
    assert res["free_seats"] == ["chair_right"]
    assert res["chosen_seat"] == "chair_right"


def test_smart_seat_detector_person_missing_xy_with_assigned_seat(physical_seats):
    """Si una persona sentada carece de profundidad 3D (xy=None) pero tiene asiento asociado, se marca ocupada."""
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    persons = [
        {"xy": None, "posture": "SENTADO", "name": "John", "seat": "chair_left"}
    ]
    res = smart_seat_detector(guest_seats, detected_persons=persons, host_xy=(3.5, 0.0))

    assert "chair_left" not in res["free_seats"]
    assert res["occupied_seats"]["chair_left"] == 1
    assert res["occupants"].get("chair_left") == "John"
    assert res["chosen_seat"] == "chair_right"


def test_smart_seat_detector_all_seats_occupied_returns_none(physical_seats):
    """Cuando todos los asientos estan ocupados por personas sentadas, no hay asientos libres."""
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    persons = [
        {"xy": (3.5, 1.5), "posture": "SENTADO", "name": "John"},
        {"xy": (3.5, -1.5), "posture": "SENTADO", "name": "Mary"},
    ]
    res = smart_seat_detector(guest_seats, detected_persons=persons, host_xy=(3.5, 0.0))

    assert res["free_seats"] == []
    assert res["chosen_seat"] is None
    assert res["occupied_seats"]["chair_left"] == 1
    assert res["occupied_seats"]["chair_right"] == 1


def test_smart_seat_detector_all_seats_reserved_returns_none(physical_seats):
    """Cuando todos los asientos estan reservados para invitados previos, free_seats queda vacio y chosen_seat es None."""
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    res = smart_seat_detector(
        guest_seats,
        detected_persons=[],
        host_xy=(3.5, 0.0),
        reserved={"chair_left": 1, "chair_right": 1},
    )

    assert res["free_seats"] == []
    assert res["chosen_seat"] is None

def test_smart_seat_detector_incorporates_scanned_occupancy(physical_seats):
    """Verifica que smart_seat_detector integre el diccionario de ocupacion previo (ej. de scan_seats)."""
    guest_seats = {"chair_left": physical_seats["chair_left"], "chair_right": physical_seats["chair_right"]}
    res = smart_seat_detector(
        guest_seats,
        detected_persons=[],
        host_xy=(3.5, 0.0),
        occupancy={"chair_left": 1, "chair_right": 0},
    )

    assert "chair_left" not in res["free_seats"]
    assert res["free_seats"] == ["chair_right"]
    assert res["chosen_seat"] == "chair_right"


def test_smart_seat_detector_metadata_flags_occupied_and_reserved(physical_seats):
    """Verifica que asientos marcados en metadatos como occupied o reserved sean descartados (fail-closed)."""
    guest_seats = {
        "chair_left": {**physical_seats["chair_left"], "occupied": True},
        "chair_right": {**physical_seats["chair_right"], "reserved": True},
    }
    res = smart_seat_detector(guest_seats, detected_persons=[], host_xy=(3.5, 0.0))

    assert res["free_seats"] == []
    assert res["chosen_seat"] is None


def test_choose_free_seat_respects_virtual_chairs_and_rejects_host_sofa(physical_seats):
    """choose_free_seat debe recibir y respetar VirtualChairs ocupadas para no ofrecer sofa_host al invitado."""
    from base_athome_perception.seat_occupancy import VirtualChair, choose_free_seat

    # physical_seats contiene:
    # "sofa_host": (3.5, 0.0), capacity 1
    # "chair_left": (3.5, 1.5), capacity 1
    # "chair_right": (3.5, -1.5), capacity 1
    # Simulamos que ambas sillas de invitados estan ocupadas fisicamente o por invitados sentados
    virtual_chairs = [
        VirtualChair(x=3.5, y=0.0, occupied=True, occupant_name="Charlie", source="host_default"),
        VirtualChair(x=3.5, y=1.5, occupied=True, occupant_name="John", source="seated_person"),
    ]
    # sofa_host NO fue detectado por camara de profundidad (occ=0)
    occ = {"sofa_host": 0, "chair_left": 1, "chair_right": 1}

    # Si choose_free_seat ignorara virtual_chairs, sofa_host tendria occ=0 y seria ofrecido erroneamente!
    chosen = choose_free_seat(physical_seats, occ, reserved={}, virtual_chairs=virtual_chairs)

    # Con respeto a VirtualChair: sofa_host esta ocupado por Charlie -> resultado None (fail-closed)
    assert chosen is None


def test_choose_free_seat_chooses_unoccupied_guest_chair_when_host_sofa_has_virtual_chair(physical_seats):
    """choose_free_seat protege sofa_host (ocupado virtualmente por Charlie) y elige la silla libre para el invitado."""
    from base_athome_perception.seat_occupancy import VirtualChair, choose_free_seat

    virtual_chairs = [
        VirtualChair(x=3.5, y=0.0, occupied=True, occupant_name="Charlie", source="host_default"),
    ]
    # sofa_host no tiene lecturas de profundidad (0), chair_left ocupada (1), chair_right libre (0)
    occ = {"sofa_host": 0, "chair_left": 1, "chair_right": 0}

    chosen = choose_free_seat(physical_seats, occ, reserved={}, virtual_chairs=virtual_chairs)
    # Debe elegir chair_right y nunca sofa_host
    assert chosen == "chair_right"


def test_is_seat_free_unified_filter(physical_seats):
    """Verifica que is_seat_free unifique las comprobaciones de coordenadas, metadatos, real, virtual y reservas."""
    from base_athome_perception.seat_occupancy import VirtualChair, is_seat_free

    sofa = physical_seats["sofa_host"]
    charlie_vc = [VirtualChair(x=3.5, y=0.0, occupied=True, occupant_name="Charlie")]

    # 1. Libre si no hay ocupacion ni reservas ni virtual chair
    assert is_seat_free("sofa_host", sofa) is True

    # 2. Ocupado si solapa con VirtualChair (anfitrión)
    assert is_seat_free("sofa_host", sofa, virtual_chairs=charlie_vc) is False

    # 3. Ocupado si tiene ocupacion fisica
    assert is_seat_free("sofa_host", sofa, occupancy={"sofa_host": 1}) is False

    # 4. Ocupado si esta reservado
    assert is_seat_free("sofa_host", sofa, reserved={"sofa_host": 1}) is False

    # 5. Ocupado si metadatos lo marcan
    assert is_seat_free("sofa_host", {**sofa, "occupied": True}) is False
    assert is_seat_free("sofa_host", {**sofa, "reserved": True}) is False

    # 6. Descartado si no tiene coordenadas xy validas (fail-closed)
    assert is_seat_free("sofa_host", {"capacity": 1}) is False
