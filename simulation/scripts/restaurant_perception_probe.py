#!/usr/bin/env python3
"""Collect real Restaurant perception rates without saving camera/person imagery.

Run with the simulation environment sourced, an isolated lab ROS domain, and an output
JSONL path. Subscribing debug_image enables bag_finder's periodic inference.
"""
import json
import argparse
import os
from pathlib import Path
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sim_domain import validate_lab_domain


def main():
    try:
        validate_lab_domain(os.environ.get('ROS_DOMAIN_ID'))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics')
    parser.add_argument('--states')
    args = parser.parse_args()
    path = Path(args.metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node('restaurant_perception_probe')
    with path.open('a', buffering=1) as out:
        def metric(msg):
            row = json.loads(msg.data)
            out.write(json.dumps(row) + '\n')
        node.create_subscription(String, '/bag_finder/sim_metrics', metric, 10)
        states = None if args.states is None else Path(args.states).open('a', buffering=1)
        if states is not None:
            node.create_subscription(String, '/restaurant/state', lambda msg: states.write(msg.data + '\n'),
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_subscription(Image, '/bag_finder/debug_image', lambda msg: None, qos_profile_sensor_data)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            if states is not None:
                states.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
