"""Modo ATHOME_SIM_REAL_GAP: defaults, ruido ASR acotado y compatibilidad con el parser del robot."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/base_athome_skills'))

from base_athome_sim import real_gap  # noqa: E402
from base_athome_skills.mission.order_asr import parse_yes_no  # noqa: E402


def test_flag_off_keeps_previous_behaviour():
    env = {}
    assert not real_gap.real_gap_enabled(env)
    assert real_gap.manipulation_defaults(env) == {'VISUAL_PICK': '1', 'RESTAURANT_DOCK': '1'}
    assert not real_gap.arm_starts_disabled(env)
    assert not real_gap.rear_scan_blocked(env)
    assert not real_gap.asr_noise_enabled(env)


def test_flag_on_turns_off_dock_and_visual_pick_but_respects_explicit_values():
    env = {'ATHOME_SIM_REAL_GAP': '1'}
    assert real_gap.manipulation_defaults(env) == {'VISUAL_PICK': '0', 'RESTAURANT_DOCK': '0'}
    env['VISUAL_PICK'] = '1'
    assert real_gap.manipulation_defaults(env)['VISUAL_PICK'] == '1'
    assert real_gap.arm_starts_disabled(env) and real_gap.rear_scan_blocked(env)
    assert not real_gap.asr_noise_enabled(dict(env, ATHOME_SIM_REAL_GAP_ASR='0'))


def test_noise_never_flips_yes_no_and_parser_understands_it():
    rng = random.Random(7)
    heard = {'yes': set(), 'no': set()}
    silences = 0
    for _ in range(2000):
        for clean, key, expected in (('robot yes', 'yes', True), ('robot no', 'no', False)):
            out = real_gap.noisy_reply(clean, rng, allow_silence=False)
            heard[key].add(out)
            assert parse_yes_no(out) is expected, out
        if real_gap.noisy_reply('robot yes', rng) is None:
            silences += 1
    assert {'road yes', 'a reward? yes.'} <= heard['yes']
    assert 0 < silences < 600


def test_echo_is_the_robot_voice_and_is_not_a_confirmation():
    rng = random.Random(3)
    say = 'Please answer robot yes or robot no.'
    echoes = [o for o in (real_gap.noisy_reply('robot yes', rng, last_robot_say=say) for _ in range(500)) if o == say]
    assert echoes and parse_yes_no(say) is None


def test_order_can_be_partial_but_never_invents_items():
    rng = random.Random(5)
    outs = {real_gap.noisy_reply('my order is coke and water', rng, allow_silence=False) for _ in range(400)}
    assert 'a coke.' in outs
    for out in outs:
        assert 'juice' not in out and 'tea' not in out


def test_real_log_forms_and_foreign_voices_appear():
    rng = random.Random(11)
    outs = {real_gap.noisy_reply('my order is coke and water', rng, allow_silence=False) for _ in range(800)}
    assert {'Coke!', 'order men.'} & outs and any('goak' in o for o in outs)
    foreign = {real_gap.noisy_reply('robot yes', rng) for _ in range(3000)} & set(real_gap.FOREIGN_PHRASES)
    assert foreign
    for phrase in real_gap.FOREIGN_PHRASES:
        assert parse_yes_no(phrase) is None


def test_barman_silence_budget_and_echo_delay():
    assert real_gap.barman_silent_windows({}) == 0
    assert real_gap.barman_silent_windows({'ATHOME_SIM_REAL_GAP': '1'}) == 2
    assert real_gap.barman_silent_windows({'ATHOME_SIM_REAL_GAP': '1', 'ATHOME_SIM_REAL_GAP_BARMAN_SILENT': '6'}) == 6
    assert 0.3 <= real_gap.echo_delay_s(random.Random(1)) <= 1.0


def test_barman_silent_windows_zero_without_gap_two_with_gap():
    assert real_gap.barman_silent_windows({}) == 0
    assert real_gap.barman_silent_windows({'ATHOME_SIM_REAL_GAP': '0'}) == 0
    assert real_gap.barman_silent_windows({'ATHOME_SIM_REAL_GAP': '1'}) == 2
