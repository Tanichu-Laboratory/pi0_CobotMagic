#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Noetic side bridge (Python 3.8).
- Collects observations from ROS topics
- Sends compact multipart messages to openpi-server via ZeroMQ (REQ)
- Receives actions (binary float32 arrays) and publishes to ROS
- Designed for minimal overhead between separate conda environments
"""

import argparse
import json
import threading
import time

import cv2
import numpy as np
import yaml
import zmq

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool

bridge = CvBridge()

# shared buffers
buf = {
    'front': None,
    'left': None,
    'right': None,
    'jl': None,
    'jr': None,
    'odom': None,
}
lock = threading.Lock()
enable_state = True  # /enable_flag があればこれに従う


def encode_jpeg(img, quality=80):
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return enc.tobytes()


def img_cb(which, mode='raw', quality=80):
    use_compressed = (mode == 'compressed')

    def f(msg):
        try:
            if use_compressed:
                data = bytes(msg.data)
                # best effort verification (optional)
                if not data:
                    return
                payload = data
            else:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                payload = encode_jpeg(img, quality=quality)
                if payload is None:
                    return
            with lock:
                buf[which] = payload
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(f"image cb error {which}: {exc}")

    return f


def jl_cb(msg: JointState):
    with lock:
        buf['jl'] = list(msg.position)


def jr_cb(msg: JointState):
    with lock:
        buf['jr'] = list(msg.position)


def odom_cb(msg: Odometry):
    with lock:
        buf['odom'] = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]


def enable_cb(msg: Bool):
    global enable_state
    enable_state = bool(msg.data)


def have_obs(use_base=False):
    with lock:
        ok = all(buf[key] is not None for key in ('front', 'left', 'right', 'jl', 'jr'))
        if use_base:
            ok = ok and (buf['odom'] is not None)
        return ok


def snapshot(task_prompt: str, use_base=False):
    with lock:
        if not all(buf[key] is not None for key in ('front', 'left', 'right', 'jl', 'jr')):
            return None
        pkt = {
            'task_prompt': task_prompt,
            'front': bytes(buf['front']),
            'left': bytes(buf['left']),
            'right': bytes(buf['right']),
            'jleft': list(buf['jl']),
            'jright': list(buf['jr']),
        }
        if use_base and buf['odom'] is not None:
            pkt['odom'] = list(buf['odom'])
        else:
            pkt['odom'] = None
        return pkt


def wait_for_joint_arrays(timeout_sec=10.0):
    deadline = time.monotonic() + max(timeout_sec, 0.1)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        with lock:
            jl = buf['jl']
            jr = buf['jr']
        if jl is not None and jr is not None:
            return np.array(jl, dtype=np.float32), np.array(jr, dtype=np.float32)
        if time.monotonic() > deadline:
            break
        rate.sleep()
    return None, None


def step_towards(current, target, step_lengths):
    next_pos = current.copy()
    for idx, step_len in enumerate(step_lengths):
        diff = target[idx] - current[idx]
        if abs(diff) <= step_len:
            next_pos[idx] = target[idx]
        else:
            next_pos[idx] = current[idx] + np.sign(diff) * step_len
    return next_pos


def move_to_home(pub_l, pub_r, name_list, target_left, target_right, step_lengths, rate_hz):
    jl, jr = wait_for_joint_arrays()
    if jl is None or jr is None:
        rospy.logwarn("home move skipped: joint state not available")
        return

    rate = rospy.Rate(max(rate_hz, 1))
    cur_left = jl.copy()
    cur_right = jr.copy()

    done = False
    while not rospy.is_shutdown() and not done:
        cur_left = step_towards(cur_left, target_left, step_lengths)
        cur_right = step_towards(cur_right, target_right, step_lengths)

        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = name_list
        msg_left.position = cur_left.tolist()
        pub_l.publish(msg_left)

        msg_right = JointState()
        msg_right.header.stamp = msg_left.header.stamp
        msg_right.name = name_list
        msg_right.position = cur_right.tolist()
        pub_r.publish(msg_right)

        if np.allclose(cur_left, target_left, atol=1e-4) and np.allclose(cur_right, target_right, atol=1e-4):
            done = True
        rate.sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='path to config.yaml')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    rospy.init_node('openpi_bridge_node')

    image_type = cfg['ros'].get('image_type', 'raw').lower()
    jpeg_quality = int(cfg['ros'].get('jpeg_quality', 80))
    rate_hz = int(cfg['ros'].get('rate_hz', 20))
    rate_hz = max(rate_hz, 1)
    topics = cfg['ros']['topics']
    task_prompt = cfg.get('task_prompt', 'demo-task')

    use_base = bool(cfg['ros'].get('use_robot_base', False))
    clip = cfg['ros'].get('clip', {'v_max': 0.2, 'w_max': 0.6})
    v_max = float(clip.get('v_max', 0.2))
    w_max = float(clip.get('w_max', 0.6))
    rb_topics = cfg['ros'].get('robot_base_topics', {'odom': '/odom', 'cmd_vel': '/cmd_vel'})

    # Subscribers (images)
    if image_type == 'compressed':
        rospy.Subscriber(topics['img_front'] + '/compressed', CompressedImage, img_cb('front', 'compressed'), queue_size=10)
        rospy.Subscriber(topics['img_left'] + '/compressed', CompressedImage, img_cb('left', 'compressed'), queue_size=10)
        rospy.Subscriber(topics['img_right'] + '/compressed', CompressedImage, img_cb('right', 'compressed'), queue_size=10)
    else:
        rospy.Subscriber(topics['img_front'], Image, img_cb('front', 'raw', jpeg_quality), queue_size=10)
        rospy.Subscriber(topics['img_left'], Image, img_cb('left', 'raw', jpeg_quality), queue_size=10)
        rospy.Subscriber(topics['img_right'], Image, img_cb('right', 'raw', jpeg_quality), queue_size=10)

    # Subscribers (joints/base/enable)
    rospy.Subscriber(topics['puppet_arm_left'], JointState, jl_cb, queue_size=50)
    rospy.Subscriber(topics['puppet_arm_right'], JointState, jr_cb, queue_size=50)
    if use_base and 'robot_base_topics' in cfg['ros']:
        rospy.Subscriber(rb_topics['odom'], Odometry, odom_cb, queue_size=50)

    if 'enable_flag' in topics and topics['enable_flag']:
        try:
            rospy.Subscriber(topics['enable_flag'], Bool, enable_cb, queue_size=10)
        except Exception:
            pass

    # Publishers
    pub_l = rospy.Publisher(topics['cmd_joint_left'], JointState, queue_size=10)
    pub_r = rospy.Publisher(topics['cmd_joint_right'], JointState, queue_size=10)
    pub_v = None
    if use_base:
        pub_v = rospy.Publisher(rb_topics['cmd_vel'], Twist, queue_size=10)

    # Home positioning before policy loop
    name_list = cfg['ros'].get('joint_names', [f'joint{i}' for i in range(7)])
    home_cfg = cfg['ros'].get('home_position')
    step_lengths = cfg['ros'].get('arm_steps_length', [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])
    if home_cfg:
        try:
            target_left = np.array(home_cfg['left'], dtype=np.float32)
            target_right = np.array(home_cfg['right'], dtype=np.float32)
            if target_left.shape[0] == len(name_list) and target_right.shape[0] == len(name_list):
                rospy.loginfo("Moving arms to home position before inference.")
                step_arr = np.asarray(step_lengths, dtype=np.float32)
                if step_arr.shape[0] != len(name_list):
                    rospy.logwarn("arm_steps_length size mismatch; using uniform step length 0.01.")
                    step_arr = np.full(len(name_list), 0.01, dtype=np.float32)
                move_to_home(pub_l, pub_r, name_list, target_left, target_right, step_arr, rate_hz)
            else:
                rospy.logwarn("home_position size mismatch; skipping home move.")
        except (KeyError, ValueError, TypeError):
            rospy.logwarn("Invalid home_position configuration; skipping home move.")

    # ZeroMQ client
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    connect_addr = cfg['zmq'].get('client_connect', 'tcp://127.0.0.1:5557')
    sock.connect(connect_addr)
    sock.setsockopt(zmq.LINGER, 0)
    recv_timeout_ms = max(int(1000 / rate_hz), 1)
    sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, recv_timeout_ms)
    rospy.loginfo(f"Connected to OpenPI server at {connect_addr}")

    rate = rospy.Rate(rate_hz)
    dt = 1.0 / rate_hz / 4.0
    integrated_left = None
    integrated_right = None

    try:
        while not rospy.is_shutdown():
            if not enable_state:
                rate.sleep()
                continue

            if not have_obs(use_base=use_base):
                rate.sleep()
                continue

            # ZeroMQ handshake
            pkt = snapshot(task_prompt, use_base=use_base)
            if pkt is None:
                rate.sleep()
                continue

            header = {
                'task_prompt': pkt['task_prompt'],
                'jleft': pkt['jleft'],
                'jright': pkt['jright'],
            }
            if use_base and pkt['odom'] is not None:
                header['odom'] = pkt['odom']

            current_left = np.array(pkt['jleft'], dtype=np.float32)
            current_right = np.array(pkt['jright'], dtype=np.float32)
            integrated_left = current_left.copy()
            integrated_right = current_right.copy()

            frames = [
                json.dumps(header, separators=(',', ':')).encode('utf-8'),
                pkt['front'],
                pkt['left'],
                pkt['right'],
            ]

            try:
                sock.send_multipart(frames)
            except zmq.error.Again:
                rospy.logwarn("ZeroMQ send timeout; skipping cycle.")
                rate.sleep()
                continue

            rep_frames = None
            while not rospy.is_shutdown():
                try:
                    rep_frames = sock.recv_multipart()
                    break
                except zmq.error.Again:
                    continue

            if rospy.is_shutdown() or rep_frames is None:
                break

            if not rep_frames:
                rate.sleep()
                continue

            rep_header = json.loads(rep_frames[0].decode('utf-8'))
            chunk_size = int(rep_header.get('chunk_size', 0))
            if chunk_size == 0 or len(rep_frames) < 3:
                rate.sleep()
                continue

            left_stride = int(rep_header.get('left_stride', 7))
            right_stride = int(rep_header.get('right_stride', 7))

            left_arr = np.frombuffer(rep_frames[1], dtype=np.float32, count=chunk_size * left_stride)
            right_arr = np.frombuffer(rep_frames[2], dtype=np.float32, count=chunk_size * right_stride)
            try:
                left_mat = left_arr.reshape((chunk_size, left_stride))
                right_mat = right_arr.reshape((chunk_size, right_stride))
            except ValueError:
                rospy.logwarn("Invalid action matrix shape.")
                rate.sleep()
                continue

            vel_mat = None
            if rep_header.get('has_vel', False) and len(rep_frames) >= 4:
                vel_arr = np.frombuffer(rep_frames[3], dtype=np.float32, count=chunk_size * 2)
                try:
                    vel_mat = vel_arr.reshape((chunk_size, 2))
                except ValueError:
                    vel_mat = None

            for tau in range(chunk_size):
                # velocity -> position
                integrated_left = integrated_left + left_mat[tau] * dt
                integrated_right = integrated_right + right_mat[tau] * dt

                integrated_left[-1] = left_mat[tau, -1]
                integrated_right[-1] = right_mat[tau, -1]

                now = rospy.Time.now()
                js = JointState()
                js.header.stamp = now
                js.name = name_list
                js.position = integrated_left.tolist()
                pub_l.publish(js)
                js.position = integrated_right.tolist()
                pub_r.publish(js)

                if use_base and vel_mat is not None and pub_v is not None:
                    v = Twist()
                    v.linear.x = float(np.clip(vel_mat[tau, 0], -v_max, v_max))
                    v.angular.z = float(np.clip(vel_mat[tau, 1], -w_max, w_max))
                    pub_v.publish(v)

                rate.sleep()

    except (KeyboardInterrupt, rospy.ROSInterruptException):
        rospy.loginfo("openpi_bridge_node interrupted, shutting down.")
    finally:
        sock.close(0)

if __name__ == '__main__':
    main()
