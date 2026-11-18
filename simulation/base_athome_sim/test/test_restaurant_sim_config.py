from pathlib import Path

from base_athome_perception.pose_store import load_named_poses
from base_athome_perception.restaurant_arm import (
    RESTAURANT_REQUIRED_POSES,
    merge_restaurant_poses,
    poses_calibrated,
)


ROOT = Path(__file__).resolve().parents[3]


def test_simulator_uses_checkout_configs_and_models():
    sim_robot = (ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py").read_text()
    arm_manipulation = (
        ROOT / "src/base_athome_perception/base_athome_perception/arm_manipulation.py"
    ).read_text()

    assert 'os.path.expanduser("~/base_athome_ws")' not in sim_robot
    assert "os.path.join(WS," not in sim_robot
    assert 'os.path.expanduser("~/base_athome_ws/src/base_athome_perception/models")' not in arm_manipulation
    assert 'os.environ.get("ATHOME_SIM_POSES") or CONFIG' in arm_manipulation
    assert 'os.path.dirname(os.path.realpath(__file__)), "..", "models"' in arm_manipulation


def test_restaurant_startup_logs_base_pose_and_measured_arm_home():
    sim_robot = (ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py").read_text()
    launch = (ROOT / "simulation/base_athome_sim/launch/sim.launch.py").read_text()
    restaurant = (ROOT / "src/base_athome_skills/scripts/restaurant.sh").read_text()
    assert "SIM_INITIAL_SPAWN gz=" in launch
    assert "SIM_INITIAL_POSE expected_gz=" in sim_robot
    assert "SIM_INITIAL_ARM_JOINTS measured=" in sim_robot
    assert "export ATHOME_SIM_ARM_START_POSE=home" in restaurant


def test_all_required_restaurant_poses_are_calibrated_in_simulation():
    arm_config = ROOT / "src/base_athome_perception/config/arm_poses_and_limits.yaml"
    restaurant_config = ROOT / "simulation/base_athome_sim/config/restaurant_arm_poses.yaml"
    merged = merge_restaurant_poses(load_named_poses(arm_config), restaurant_config)

    assert poses_calibrated(merged)
    assert all(merged[name]["calibrated"] is True for name in RESTAURANT_REQUIRED_POSES)
    assert merged["restaurant_carry"]["calibrated"] is True
    assert merged["restaurant_pregrasp"]["joints_deg"] == merged["restaurant_postgrasp"]["joints_deg"]
    assert merged["restaurant_postgrasp"]["joints_deg"] == [21.2, 170.93, -72.96, 59.79, -78.0, -89.05]


def test_simulation_carry_pose_is_folded_home_with_gripper_closed():
    poses = load_named_poses(ROOT / "src/base_athome_perception/config/arm_poses_and_limits.yaml")
    carry = merge_restaurant_poses(
        poses, ROOT / "simulation/base_athome_sim/config/restaurant_arm_poses.yaml"
    )["restaurant_carry"]

    assert carry["joints_deg"] == [84.74, 172.79, -160.31, -20.99, -10.47, 1.49]
    assert carry["gripper_mm"] == 50.0
    assert carry["calibrated"] is True


def test_debug_subprocess_template_does_not_reference_parent_args():
    import ast
    from types import SimpleNamespace

    source = (ROOT / "simulation/scripts/sim_counter_debug.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    template = next(node for node in ast.walk(main)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "test_script" for target in node.targets))
    namespace = {
        "REPO": ROOT,
        "WORLD_NAME": "restaurant",
        "args": SimpleNamespace(can_y=0.0, hold=5, start_x=-0.90, start_yaw=180.0,
                                timeout=240, retreat_m=0.20, tag="test_tag"),
        "env": {"GZ_PARTITION": "test"},
    }
    child_script = eval(compile(ast.Expression(template.value), "sim_counter_debug.py", "eval"), namespace)
    ast.parse(child_script)
    assert "args.hold" not in child_script


def test_debug_docks_after_safe_camera_look_and_refreshes_scene():
    source = (ROOT / "simulation/scripts/sim_counter_debug.py").read_text()
    assert 'default=0.0, help="Coordenada Y' in source
    assert 'default=-0.90' in source
    assert 'default=180.0' in source
    pregrasp = source.index("while arm_thread.is_alive():")
    wait_for_pregrasp = source.index("if arm_thread.is_alive() or not arm_ok:")
    visual_reference = source.index("PRE_DOCK_CAN_REFERENCE:")
    look_pose = source.index("node.go_to_blocking('restaurant_look')")
    docking = source.index("if not docking.run():")
    search = source.index("Camara en rumbo actual")
    assert pregrasp < wait_for_pregrasp < visual_reference < look_pose < docking < search
    assert "counter_look_pose_active = True" in source
    assert "REDETECT_MATCH" in source and "match_error > 0.03" in source
    assert "PICK_TARGET_MATCH" in source and "det_ok = match_error <= 0.03" in source
    assert "scene.world.collision_objects = [counter, can]" in source
    assert "scene_after_dock.world.collision_objects = [counter, can]" in source
    assert "world_xy_in_base(can_pose0[0], can_pose0[1])" in source
    assert "offsets = [0] + list(range(5, 41, 5)) + list(range(35, -41, -5))" in source
    assert "quick_can_visible()" in source
    assert "diagnose_can_framing()" in source
    interrupted_sweep = source.split("if node._visual_interrupt:", 1)[1].split("print(f'Barrido detenido", 1)[0]
    assert "bearing = math.atan2(float(target['obj'][1]), float(target['obj'][0]))" in interrupted_sweep
    assert "turn_to(docking.pose[2] + bearing, interrupt_on_detection=False)" in interrupted_sweep
    assert "node.abort = True" not in source
    assert "create_subscription(Image, '/sim/gz/wrist'" in source
    assert "save_wrist_snapshot('pre_barrido')" in source
    assert "save_wrist_snapshot('antes_docking_pose_look')" in source
    assert "save_wrist_snapshot('pregrasp_fallido', failed_frame)" in source
    assert "rclpy.time.Time())" in source
    assert "FRAMING_TF" in source
    assert "FRAMING_TF latest fallback age={{age:.3f}}s; arm/base stationary" in source
    assert "if age > 0.25:" in source
    assert "'/restaurant/wrist_detection/image'" in source
    assert "agarre_camaras.rviz" in source
    assert "can_distance_error(bbox, q_deg)" in source
    assert "def wrist_can_expected_size_px():" in source
    assert "_detect_sim_can(frame, wrist_can_expected_size_px())" in source
    assert "arm.detect_item('can', timeout_s=15.0)" in source
    assert "arm.pick_item('can')" in source
    assert "align_can_with_collision_checked_plans" not in source
    assert "distancia gate no disponible" in source
    state_hold = source.index("state_pub.publish(String(data='{{\"state\":\"PICKING_ITEM\"}}'))")
    detect = source.index("arm.detect_item('can', timeout_s=15.0)")
    pick = source.index("arm.pick_item('can')")
    carry = source.index("carry = CounterCarry(", pick)
    assert state_hold < detect < pick < carry


def test_counter_debug_aligns_base_before_camera_search():
    source = (ROOT / "simulation/scripts/sim_counter_debug.py").read_text()
    assert "while arm_thread.is_alive():" in source
    align = "if not turn_to(math.pi):"
    assert align in source
    assert source.index("while arm_thread.is_alive():") < source.index(align)
    assert source.index(align) < source.index("Camara en rumbo actual")


def test_sim_gazebo_joint_states_are_measured_and_fail_closed_when_stale():
    source = (ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py").read_text()
    assert "measured_goal_deviation_deg(measured_feedback, measured_age, goal_deg)" in source
    assert 'arm_names = [f"joint{i}" for i in range(1, 9)]' in source
    assert 'arm_names, arm_positions, arm_stamp = [], [], stamp' in source
    assert "arm_status_from_measured_sample(" in source
    assert "SIM_GZ_JOINT_STATE_PERIOD" in source
    assert "os.path.join(REPO_ROOT, \"src/base_athome_perception/config/arm_poses_and_limits.yaml\")" in source
    assert 'if pub_detach:' in source


def test_gazebo_joint_state_publisher_runs_at_50_hz():
    from base_athome_sim import gz_model

    generated = gz_model.gazebo_blocks()
    assert "<topic>/sim/gz/joint_states</topic><update_rate>50</update_rate>" in generated


def test_sim_arm_status_timer_has_independent_group_and_period_diagnostics():
    source = (ROOT / "simulation/base_athome_sim/base_athome_sim/sim_robot.py").read_text()
    assert "self.arm_status_cb = MutuallyExclusiveCallbackGroup()" in source
    assert "self.create_timer(0.05, self._arm_status, callback_group=self.arm_status_cb)" in source
    assert "SIM_ARM_STATUS_PERIOD" in source
