"""
scene.py
--------
Populate the workcell: table, backdrop wall, tabletop clutter for the depth
camera to see, the UR5e, and a ChArUco calibration board at a hardcoded pose.

    python scripts/scene.py            open the viewer
    python scripts/scene.py --check    compile only, print stats

VIEWER KEY:  press 'E' to toggle "environment only" rendering (table, wall,
monitor, floor, and the robot itself -- everything except the movable
tabletop clutter) on and off. The same toggle is available programmatically
via apply_environment_only(), for the depth-camera ROS node to hide clutter
when it wants a clean background with the arm still visible.

THE TRAP THIS FILE HAD TO SOLVE:  build_scene_xml() returns a plain string,
which you'd naturally compile with mujoco.MjModel.from_xml_string(...). But
relative filenames inside an XML *string* have no file location to resolve
against, and the vendored assets/robots/ur5e/ur5e.xml has its own top-level
<compiler angle="radian" meshdir="assets" .../> that its 20 .obj meshes
depend on. Giving the PARENT scene its own <compiler meshdir=...> (needed, in
theory, for the ChArUco texture) does not "merge" with the child's the way
you'd hope -- it was tried empirically here and produced
"Error opening file .../ur5e/base_0.obj: No such file or directory": the
child's relative meshdir got resolved against the wrong base entirely.

The fix actually used: never give the parent scene a <compiler> element at
all. Every path this file writes into the XML (the ChArUco texture, the
<include> target) is either absolute or, when the robot is included, made
correct by colocation -- see _compile_model() below, which writes the
generated XML into assets/robots/ur5e/ (the same directory as ur5e.xml) and
compiles with from_xml_path instead of from_xml_string. That gives ur5e.xml's
own relative meshdir="assets" a real directory to resolve against, exactly as
it has when MuJoCo Menagerie's own scene.xml is used standalone. This was
verified by actually compiling both ways before picking one -- see the
module's git history / the task report for the failed attempts.

A parent <option> element, by contrast, was tested and found to merge fine
with ur5e.xml's own <option integrator="implicitfast"/> (different attributes,
no clash), so this file does add one, for gravity.

A THIRD trap, discovered only by actually compiling with the clutter's
<freejoint/>s in place: ur5e.xml carries its own <keyframe><key name="home"
qpos="six numbers".../></keyframe>. A keyframe's qpos length is validated
against the WHOLE compiled model's nq, not the included file's own nq, so the
moment this scene adds 5 free bodies (35 more DOF) the vendored "home"
keyframe becomes invalid ("expected length 41, got 6") and the model refuses
to compile -- whether or not anything ever reads that keyframe. MuJoCo also
refuses two keys with the same name, so you cannot just add a corrected
"home" alongside it. The fix: _compile_model() strips the vendored
<keyframe> out of ur5e.xml's text before writing the colocated scratch file,
and build_scene_xml() emits its own full-length "home" keyframe instead --
the same 6 robot angles plus each clutter body's resting (pos, identity
quat) as its 7-number freejoint entry, in the exact order those bodies are
declared (verified empirically: <include>'d content occupies the lower qpos
addresses, ahead of this file's own <worldbody> children). The replacement
key also needs its OWN "ctrl" attribute matching the robot's home angles --
omitting it (as a first pass here did) leaves data.ctrl at its zero default
while qpos starts at home, so the position actuators immediately drag the
arm from home back toward qpos=0, sweeping it through the tabletop clutter
during the very stability check meant to prove the clutter is undisturbed.

ROBOT PLACEMENT: the UR5e sits at ROBOT_BASE_XY (the +x end of the table, far
from the human at the -x end), not at the world origin. The vendored
ur5e.xml's <body name="base" quat="0 0 0 -1"> has no pos, so _compile_model()
injects one into the same scratch copy it already writes for the keyframe
strip. Anything that treats "base_link" as the robot frame must subtract
ROBOT_BASE_POS from world coordinates first (mujoco_ros_cell does, for the
camera TF).
"""

import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ----------------------------------------------------------------------
# Paths. Computed at runtime, like scripts/fetch_robot.py does -- never
# hardcoded, so this works from any checkout.
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UR5E_DIR = PROJECT_ROOT / "assets" / "robots" / "ur5e"
UR5E_XML = UR5E_DIR / "ur5e.xml"
CHARUCO_PNG = PROJECT_ROOT / "assets" / "charuco" / "charuco_6x9_sq32mm_mk24mm_DICT6X6_250.png"

# Name of the scratch copy of ur5e.xml (keyframe stripped, see module
# docstring) and of the generated scene, both written into UR5E_DIR so
# ur5e.xml's own relative meshdir="assets" resolves correctly.
UR5E_NOKEY_NAME = "_scene_ur5e_nokeyframe.xml"
SCENE_SCRATCH_NAME = "_scene_scratch.xml"

HOME_QPOS_ROBOT = "-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"


# ----------------------------------------------------------------------
# Geom groups. This is the mechanism behind the environment-only toggle:
# MuJoCo's MjvOption.geomgroup is a per-group visibility mask that both the
# interactive viewer and offscreen mujoco.Renderer respect, so hiding
# "everything but the environment" is one array write, not per-geom logic.
#
#   group 0  ENVIRONMENT   table, wall, monitor, floor -- never move
#   group 1  CLUTTER       glass, bottle, blocks, tray, screwdriver, car, human,
#                          ChArUco board -- anything that is NOT static background
#   group 2  robot visual  (from the vendored ur5e.xml, untouched)
#   group 3  robot collision (from the vendored ur5e.xml, untouched)
#   group 4  robot sites     (from the vendored ur5e.xml, untouched)
# ----------------------------------------------------------------------

GROUP_ENV = 0
GROUP_CLUTTER = 1
GROUP_ROBOT_VISUAL = 2

ENV_ONLY_TOGGLE_KEY = "E"


# Robot mount: the +x end of the table, where block_flat used to sit next to
# the ChArUco board (board spans x 0.25..0.54; the base pedestal is ~0.09 m
# radius, and the table edge is at x = 0.80). The human is at the -x end.
ROBOT_BASE_XY = (0.68, 0.0)
ROBOT_BASE_POS = np.array([ROBOT_BASE_XY[0], ROBOT_BASE_XY[1], 0.0])


# ----------------------------------------------------------------------
# Table. Same z=0-at-the-top convention as Lesson 3, sized up so the robot's
# reach and every clutter object below fit on it without overhang.
# ----------------------------------------------------------------------

TABLE_LENGTH = 1.60      # along x
TABLE_WIDTH = 1.10       # along y
TABLE_HEIGHT = 0.75      # floor to top surface
SLAB_THICKNESS = 0.05
LEG_THICKNESS = 0.05
LEG_INSET = 0.08

# ----------------------------------------------------------------------
# Backdrop wall. The depth camera below sits at y = -0.75 looking toward the
# table at y = 0, so "behind the table" from the camera's viewpoint is +y --
# past the table's far edge at y = +TABLE_WIDTH/2. Sized generously (rather
# than trigonometrically exact for one fixed fovy) so it fills the camera's
# background regardless of small pose changes later.
# ----------------------------------------------------------------------

WALL_GAP = 0.05          # clearance behind the table's far edge
WALL_THICKNESS = 0.05
WALL_HEIGHT = 2.0        # measured up from the FLOOR, not the table top
WALL_MARGIN_X = 1.0      # extra width past each end of the table

# ----------------------------------------------------------------------
# The depth camera, carried over from Lesson 3's bench.
# ----------------------------------------------------------------------

CAMERA_POS = (0.75, -0.75, 0.65)
CAMERA_FOVY = 58

# ----------------------------------------------------------------------
# Tabletop clutter. Every object's SMALLEST dimension must stay >= 0.04 m --
# these get voxelized at 5 mm for the distance field, and anything thinner
# than that voxel-and-a-bit disappears instead of surviving as a recognizable
# cluster of points. Each object rests with its bottom face exactly on
# z = 0: a box/cylinder of half-height h therefore has its centre at z = h.
# ----------------------------------------------------------------------

GLASS_RADIUS = 0.040
GLASS_HALF_HEIGHT = 0.060
GLASS_POS_XY = (-0.35, -0.35)
GLASS_RGBA = "0.75 0.90 1.00 0.55"

BOTTLE_BODY_RADIUS = 0.040
BOTTLE_BODY_HALF_HEIGHT = 0.080
BOTTLE_NECK_RADIUS = 0.025
BOTTLE_NECK_HALF_HEIGHT = 0.030
BOTTLE_POS_XY = (-0.55, 0.15)
BOTTLE_RGBA = "0.10 0.45 0.20 1"

MONITOR_HALF_X = 0.175
MONITOR_HALF_Y = 0.025    # thin, but 0.05 m thick overall -- clears the 0.04 m floor
MONITOR_HALF_Z = 0.125
MONITOR_POS_XY = (0.0, 0.45)
MONITOR_RGBA = "0.08 0.08 0.09 1"

# Blocks of differing sizes (a third, block_flat, used to sit at (0.65, 0.0);
# the robot took its place) -- (half_x, half_y, half_z), xy position, colour.
BLOCK_SPECS = [
    dict(name="block_small", half=(0.060, 0.060, 0.060),
         pos_xy=(0.55, -0.35), rgba="0.85 0.25 0.20 1"),
    dict(name="block_medium", half=(0.080, 0.080, 0.080),
         pos_xy=(0.60, 0.35), rgba="0.95 0.55 0.10 1"),
]

# ----------------------------------------------------------------------
# Composite objects: each is ONE free body made of several geoms (so it moves
# and collides as a rigid whole). All rest with their lowest point at z = 0.
# The screwdriver's shaft is thinner than the 0.04 m rule above -- that rule
# came from the removed distance-field code; the camera-side voxel filter is
# now 5 mm, so it still survives as a recognisable strip of points.
# ----------------------------------------------------------------------

# Open-top tray: floor plate + four walls.
TRAY_HALF = (0.090, 0.065)     # outer half-extents in x, y
TRAY_HALF_HEIGHT = 0.030
TRAY_WALL = 0.008
TRAY_FLOOR = 0.008
TRAY_POS_XY = (0.0, -0.20)
TRAY_RGBA = "0.20 0.55 0.80 1"

# Screwdriver lying on the table along x: octagonal handle (two boxes turned
# 45 deg so it cannot roll), a shaft, a tip, and a flat blade.
SCREWDRIVER_POS_XY = (-0.30, 0.30)
SCREWDRIVER_HANDLE_HALF_LEN = 0.055
SCREWDRIVER_HANDLE_HALF_W = 0.024    # across flats = 0.048
SCREWDRIVER_SHAFT_LEN = 0.110
SCREWDRIVER_SHAFT_RADIUS = 0.006
SCREWDRIVER_HANDLE_RGBA = "0.85 0.15 0.15 1"
SCREWDRIVER_SHAFT_RGBA = "0.72 0.72 0.75 1"

# Toy car, nose toward +x: chassis, cabin, four wheels with axles along y.
CAR_POS_XY = (0.15, -0.32)
CAR_YAW_DEG = 20.0
CAR_WHEEL_RADIUS = 0.022
CAR_WHEEL_HALF_W = 0.008
CAR_CHASSIS_HALF = (0.065, 0.030, 0.012)
CAR_CABIN_HALF = (0.030, 0.026, 0.017)
CAR_BODY_RGBA = "0.95 0.80 0.10 1"
CAR_WHEEL_RGBA = "0.10 0.10 0.10 1"

# Free bodies in declaration order -- the home keyframe's qpos block must
# follow exactly this order (see build_scene_xml).
CLUTTER_BODIES = ["glass", "bottle"] + [b["name"] for b in BLOCK_SPECS] + \
                 ["tray", "screwdriver", "toy_car"]

# ----------------------------------------------------------------------
# Scene variants. "default" is the cluttered workcell above. "two_cylinders"
# replaces ALL clutter (human figure and ChArUco board included) with two
# tall static cylinders in front of the robot, for testing per-class
# collision behaviour: the robot can be driven straight into either one.
# "moving_cylinder" is two_cylinders plus a third, mocap-driven cylinder.
# The environment (table, wall, monitor, floor) is identical in every
# variant, so one environment capture serves all of them.
SCENES = ("default", "two_cylinders", "moving_cylinder")

# Positions are given in the ROBOT's frame (base_link: x forward from the
# base, y to its left), then converted below -- the base sits at
# ROBOT_BASE_XY, yawed 180 deg (ur5e.xml's base quat="0 0 0 -1"), so
# world = (ROBOT_BASE_XY[0] - x, ROBOT_BASE_XY[1] - y).
# Keep in sync with class_regions in rmp2_ros' sim_startup.launch, which
# labels each cylinder by these same base_link positions.
CYLINDER_RADIUS = 0.05
CYLINDER_HALF_HEIGHT = 0.25
CYLINDER_SPECS = [
    dict(name="cylinder_left",  base_xy=(0.50, 0.25),  rgba="0.85 0.35 0.30 1"),
    dict(name="cylinder_right", base_xy=(0.50, -0.25), rgba="0.30 0.55 0.85 1"),
]
# "moving_cylinder" adds a third cylinder, a MOCAP body: the ROS node sets its
# pose before every step from moving_cylinder_world_pos(sim time), so physics
# never moves it. It sweeps in base_link y at fixed base_link x.
MOVING_CYLINDER_X = 0.65            # base_link x (forward from the base), m
MOVING_CYLINDER_AMPLITUDE = 0.35    # base_link y swing, +/- m
MOVING_CYLINDER_PERIOD = 12.0       # sim-time seconds per full sweep


def _cylinder_world_xy(base_xy):
    return ROBOT_BASE_XY[0] - base_xy[0], ROBOT_BASE_XY[1] - base_xy[1]


def moving_cylinder_world_pos(t):
    """World-frame (x, y, z) of the moving cylinder at sim time t (seconds)."""
    base_y = MOVING_CYLINDER_AMPLITUDE * np.sin(2.0 * np.pi * t / MOVING_CYLINDER_PERIOD)
    x, y = _cylinder_world_xy((MOVING_CYLINDER_X, base_y))
    return np.array([x, y, CYLINDER_HALF_HEIGHT], dtype=float)


def _cylinders_xml(group):
    # Static (no freejoint) and contype/conaffinity 0: the arm passes
    # through instead of knocking them over, so a collision shows up as
    # penetration and the cylinders (and their labels) never move.
    out = []
    for spec in CYLINDER_SPECS:
        x, y = _cylinder_world_xy(spec["base_xy"])
        out.append(f"""
    <body name="{spec['name']}" pos="{x:.4f} {y:.4f} {CYLINDER_HALF_HEIGHT:.4f}">
      <geom name="{spec['name']}_geom" type="cylinder" group="{group}"
            contype="0" conaffinity="0"
            size="{CYLINDER_RADIUS:.4f} {CYLINDER_HALF_HEIGHT:.4f}" rgba="{spec['rgba']}"/>
    </body>""")
    return "".join(out)


def _moving_cylinder_xml(group):
    # Mocap body: must be a direct child of worldbody and carry no joint.
    # Its pose at compile time is the t=0 pose; the node overwrites
    # data.mocap_pos every step.
    x, y, z = moving_cylinder_world_pos(0.0)
    return f"""
    <body name="cylinder_moving" mocap="true" pos="{x:.4f} {y:.4f} {z:.4f}">
      <geom name="cylinder_moving_geom" type="cylinder" group="{group}"
            contype="0" conaffinity="0"
            size="{CYLINDER_RADIUS:.4f} {CYLINDER_HALF_HEIGHT:.4f}" rgba="0.30 0.80 0.35 1"/>
    </body>"""


# ----------------------------------------------------------------------
# Human figure, reaching for the bottle without touching it. Built as a
# nested body chain (base -> pelvis -> shoulder -> elbow) with NO joints: it
# never moves (so it can never tip or drift into the bottle) but it is a
# person, i.e. a dynamic obstacle: GROUP_CLUTTER, NOT GROUP_ENV, so env_only
# hides it and the captured environment cloud never contains it. Each body carries its own rotation, so posing
# it is just editing the angles below.
#
# TO POSE IT, edit only these HUMAN_* numbers (angles in degrees, all 0 =
# standing straight with arms hanging straight down):
#   placement  HUMAN_BASE_XY, HUMAN_YAW_DEG
#   body       HUMAN_TORSO_LEAN_DEG
#   arms       HUMAN_{REACH,SIDE}_SHOULDER_PITCH_DEG / _SWING_DEG / _ELBOW_DEG
# then run `python scripts/scene.py --check`, which prints how far each
# body part is from every object, and fails if the hand touches anything.
#
# HUMAN_AUTO_PLACE = True treats HUMAN_BASE_XY as a starting point only: the
# whole figure is slid horizontally so the reaching hand's surface ends
# exactly HUMAN_REACH_CLEARANCE from the bottle's surface, whatever pose you
# choose. Set it False to put the feet exactly at HUMAN_BASE_XY.
# ----------------------------------------------------------------------

HUMAN_AUTO_PLACE = True
HUMAN_BASE_XY = (-1.15, 0.20)      # between the feet, on the floor
HUMAN_YAW_DEG = 15.0               # 0 = facing +x (toward the table)
HUMAN_TORSO_LEAN_DEG = 30.0        # + bends forward at the hips

# pitch: + raises the arm forward (90 = pointing straight ahead)
# swing: + moves the hand across toward the body's centre line
# elbow: + bends the forearm forward/up
HUMAN_REACH_SHOULDER_PITCH_DEG = 40.0     # right arm
HUMAN_REACH_SHOULDER_SWING_DEG = 0.0
HUMAN_REACH_ELBOW_DEG = 60.0
HUMAN_SIDE_SHOULDER_PITCH_DEG = 20.0      # left arm, relaxed
HUMAN_SIDE_SHOULDER_SWING_DEG = 0.0
HUMAN_SIDE_ELBOW_DEG = 15.0

HUMAN_REACH_CLEARANCE = 0.05

HUMAN_HIP_HEIGHT = 0.90
HUMAN_TORSO_LEN = 0.45             # hip to shoulder line
HUMAN_SHOULDER_HALF_WIDTH = 0.20
HUMAN_UPPER_ARM_LEN = 0.30
HUMAN_FOREARM_LEN = 0.27
HUMAN_HAND_RADIUS = 0.04
HUMAN_HEAD_RADIUS = 0.11

HUMAN_SKIN_RGBA = "0.88 0.68 0.55 1"
HUMAN_SHIRT_RGBA = "0.25 0.35 0.55 1"
HUMAN_PANTS_RGBA = "0.15 0.15 0.20 1"


# Rotations are emitted as quats rather than euler angles because the
# included ur5e.xml switches the compiler to radians while the no-robot
# scene stays in degrees -- a quat means the same thing in both.
def _quat(axis, deg):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), np.radians(deg))
    return q


def _quat_mul(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def _rotate(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), q)
    return out


def _arm_quats(side, pitch, swing, elbow):
    """side = -1 for the right arm (body -y), +1 for the left."""
    shoulder = _quat_mul(_quat((0, 1, 0), -pitch), _quat((1, 0, 0), -side * swing))
    return shoulder, _quat((0, 1, 0), -elbow)


def _human_frames():
    q_shoulder, q_elbow = _arm_quats(-1, HUMAN_REACH_SHOULDER_PITCH_DEG,
                                     HUMAN_REACH_SHOULDER_SWING_DEG, HUMAN_REACH_ELBOW_DEG)
    return dict(base=_quat((0, 0, 1), HUMAN_YAW_DEG),
                pelvis=_quat((0, 1, 0), HUMAN_TORSO_LEAN_DEG),
                reach_shoulder=q_shoulder, reach_elbow=q_elbow)


def _human_hand_world(base_xy):
    """Forward kinematics of the reaching hand's centre -- must mirror the
    body chain _human_xml() emits."""
    f = _human_frames()
    p = np.array([0.0, 0.0, -HUMAN_FOREARM_LEN])
    p = np.array([0.0, 0.0, -HUMAN_UPPER_ARM_LEN]) + _rotate(f["reach_elbow"], p)
    p = np.array([0.0, -HUMAN_SHOULDER_HALF_WIDTH, HUMAN_TORSO_LEN]) + _rotate(f["reach_shoulder"], p)
    p = np.array([0.0, 0.0, HUMAN_HIP_HEIGHT]) + _rotate(f["pelvis"], p)
    return np.array([base_xy[0], base_xy[1], -TABLE_HEIGHT]) + _rotate(f["base"], p)


def human_base_xy():
    if not HUMAN_AUTO_PLACE:
        return tuple(HUMAN_BASE_XY)
    hand = _human_hand_world(HUMAN_BASE_XY)
    if not 0.0 < hand[2] < 2 * BOTTLE_BODY_HALF_HEIGHT:
        raise ValueError(
            f"human hand ends at z={hand[2]:.3f}, outside the bottle body's height "
            f"(0 .. {2 * BOTTLE_BODY_HALF_HEIGHT:.2f}), so HUMAN_AUTO_PLACE can't aim it "
            "at the bottle's side -- change the lean/shoulder/elbow angles, or set "
            "HUMAN_AUTO_PLACE = False")
    offset = hand[:2] - np.asarray(BOTTLE_POS_XY)
    dist = np.linalg.norm(offset)
    want = BOTTLE_BODY_RADIUS + HUMAN_HAND_RADIUS + HUMAN_REACH_CLEARANCE
    return tuple(np.asarray(HUMAN_BASE_XY) + offset / dist * (want - dist))


def _human_xml(group):
    bx, by = human_base_xy()
    f = _human_frames()
    L1, L2, W, T = (HUMAN_UPPER_ARM_LEN, HUMAN_FOREARM_LEN,
                    HUMAN_SHOULDER_HALF_WIDTH, HUMAN_TORSO_LEN)

    def q(quat):
        return " ".join(f"{v:.6f}" for v in quat)

    def leg(name, y):
        return f"""
      <geom name="human_leg_{name}" type="capsule" group="{group}"
            fromto="0 {y:.4f} {HUMAN_HIP_HEIGHT:.4f}  0 {y:.4f} 0.0700" size="0.07"
            rgba="{HUMAN_PANTS_RGBA}"/>
      <geom name="human_foot_{name}" type="box" group="{group}" pos="0.06 {y:.4f} 0.03"
            size="0.12 0.05 0.03" rgba="{HUMAN_PANTS_RGBA}"/>"""

    def arm(name, side, q_shoulder, q_elbow):
        return f"""
        <body name="human_{name}_shoulder" pos="0 {side * W:.4f} {T:.4f}" quat="{q(q_shoulder)}">
          <geom name="human_upper_arm_{name}" type="capsule" group="{group}"
                fromto="0 0 0  0 0 {-L1:.4f}" size="0.045" rgba="{HUMAN_SHIRT_RGBA}"/>
          <body name="human_{name}_elbow" pos="0 0 {-L1:.4f}" quat="{q(q_elbow)}">
            <geom name="human_forearm_{name}" type="capsule" group="{group}"
                  fromto="0 0 0  0 0 {-L2:.4f}" size="0.035" rgba="{HUMAN_SKIN_RGBA}"/>
            <geom name="human_hand_{name}" type="sphere" group="{group}"
                  pos="0 0 {-L2:.4f}" size="{HUMAN_HAND_RADIUS:.4f}" rgba="{HUMAN_SKIN_RGBA}"/>
          </body>
        </body>"""

    side_shoulder, side_elbow = _arm_quats(+1, HUMAN_SIDE_SHOULDER_PITCH_DEG,
                                           HUMAN_SIDE_SHOULDER_SWING_DEG, HUMAN_SIDE_ELBOW_DEG)
    return f"""
    <body name="human" pos="{bx:.4f} {by:.4f} {-TABLE_HEIGHT:.4f}" quat="{q(f['base'])}">
{leg("r", -0.10)}{leg("l", 0.10)}
      <body name="human_pelvis" pos="0 0 {HUMAN_HIP_HEIGHT:.4f}" quat="{q(f['pelvis'])}">
        <geom name="human_torso" type="box" group="{group}" pos="0 0 {T / 2:.4f}"
              size="0.10 0.17 {T / 2:.4f}" rgba="{HUMAN_SHIRT_RGBA}"/>
        <geom name="human_neck" type="capsule" group="{group}"
              fromto="0 0 {T:.4f}  0 0 {T + 0.08:.4f}" size="0.05" rgba="{HUMAN_SKIN_RGBA}"/>
        <geom name="human_head" type="sphere" group="{group}"
              pos="0 0 {T + 0.08 + HUMAN_HEAD_RADIUS:.4f}" size="{HUMAN_HEAD_RADIUS:.4f}"
              rgba="{HUMAN_SKIN_RGBA}"/>
{arm("reach", -1, f["reach_shoulder"], f["reach_elbow"])}{arm("side", +1, side_shoulder, side_elbow)}
      </body>
    </body>"""

# ----------------------------------------------------------------------
# ChArUco calibration board. Pose is COPIED VERBATIM from the real rig's
# poses.yaml (marker "aruco_1", parent frame base_link == our world origin /
# table top). Do not "fix" this transform -- it is deliberately hardcoded.
#
# Filename decodes the physical board: 6x9 squares, 32 mm squares ->
# 0.192 m x 0.288 m. MuJoCo quat order is (w, x, y, z).
# ----------------------------------------------------------------------

BOARD_SQUARES_X = 6
BOARD_SQUARES_Y = 9
BOARD_SQUARE_SIZE = 0.032
BOARD_WIDTH = BOARD_SQUARES_X * BOARD_SQUARE_SIZE     # 0.192 m, local x
BOARD_DEPTH = BOARD_SQUARES_Y * BOARD_SQUARE_SIZE     # 0.288 m, local y
BOARD_THICKNESS = 0.003

BOARD_POS = (0.395, 0.0, 0.047)
BOARD_QUAT_WXYZ = (0.0, 0.7071067811865476, 0.7071067811865475, 0.0)

# This quat is R = [[0,1,0],[1,0,0],[0,0,-1]]: the board's local +z axis
# points DOWN in world (maps to world -z). See the module report / --check
# output for whether the printed face still ends up visible from above --
# short answer: yes, because it is a BOX, not a plane, so both its top and
# bottom faces carry the texture; only a single-sided plane geom would go
# dark from above with this quat.


def _legs_xml(table_length, table_width, table_height, slab_thickness,
              leg_thickness, leg_inset, group):
    leg_length = table_height - slab_thickness
    leg_half_z = leg_length / 2
    leg_center_z = -slab_thickness - leg_half_z
    leg_x = table_length / 2 - leg_inset
    leg_y = table_width / 2 - leg_inset
    leg_half = leg_thickness / 2

    legs = []
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        legs.append(
            f'      <geom name="table_leg{i}" type="box" group="{group}" '
            f'pos="{sx * leg_x:.4f} {sy * leg_y:.4f} {leg_center_z:.4f}" '
            f'size="{leg_half:.4f} {leg_half:.4f} {leg_half_z:.4f}" '
            f'rgba="0.35 0.35 0.38 1"/>'
        )
    return "\n".join(legs)


def _blocks_xml(group):
    blocks = []
    for spec in BLOCK_SPECS:
        hx, hy, hz = spec["half"]
        x, y = spec["pos_xy"]
        blocks.append(f"""
    <body name="{spec['name']}" pos="{x:.4f} {y:.4f} {hz:.4f}">
      <freejoint/>
      <geom name="{spec['name']}_geom" type="box" group="{group}"
            size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{spec['rgba']}"/>
    </body>""")
    return "".join(blocks)


def _tray_xml(group):
    hx, hy = TRAY_HALF
    H, w, F = TRAY_HALF_HEIGHT, TRAY_WALL, TRAY_FLOOR
    x, y = TRAY_POS_XY
    g = f'group="{group}" rgba="{TRAY_RGBA}"'
    return f"""
    <!-- Open tray: floor plate + four walls, one rigid body. -->
    <body name="tray" pos="{x:.4f} {y:.4f} {H:.4f}">
      <freejoint/>
      <geom name="tray_floor" type="box" {g} pos="0 0 {-H + F / 2:.4f}" size="{hx:.4f} {hy:.4f} {F / 2:.4f}"/>
      <geom name="tray_wall_px" type="box" {g} pos="{hx - w / 2:.4f} 0 0" size="{w / 2:.4f} {hy:.4f} {H:.4f}"/>
      <geom name="tray_wall_nx" type="box" {g} pos="{-(hx - w / 2):.4f} 0 0" size="{w / 2:.4f} {hy:.4f} {H:.4f}"/>
      <geom name="tray_wall_py" type="box" {g} pos="0 {hy - w / 2:.4f} 0" size="{hx - w:.4f} {w / 2:.4f} {H:.4f}"/>
      <geom name="tray_wall_ny" type="box" {g} pos="0 {-(hy - w / 2):.4f} 0" size="{hx - w:.4f} {w / 2:.4f} {H:.4f}"/>
    </body>"""


def _screwdriver_xml(group):
    x, y = SCREWDRIVER_POS_XY
    L, a = SCREWDRIVER_HANDLE_HALF_LEN, SCREWDRIVER_HANDLE_HALF_W
    t = a * np.tan(np.radians(22.5))
    r = SCREWDRIVER_SHAFT_RADIUS
    shaft_end = L + SCREWDRIVER_SHAFT_LEN
    hq = lambda deg: " ".join(f"{v:.6f}" for v in _quat((1, 0, 0), deg))
    hg = f'type="box" group="{group}" rgba="{SCREWDRIVER_HANDLE_RGBA}"'
    sg = f'group="{group}" rgba="{SCREWDRIVER_SHAFT_RGBA}"'
    # Regular octagon (apothem a) = union of four rectangles turned 0/45/90/135
    # deg about the handle axis; resting on a flat face, so it cannot roll.
    return f"""
    <!-- Screwdriver lying along +x: octagonal handle, collar, shaft, flat blade. -->
    <body name="screwdriver" pos="{x:.4f} {y:.4f} {a:.4f}">
      <freejoint/>
      <geom name="screwdriver_handle_0" {hg} size="{L:.4f} {a:.4f} {t:.4f}"/>
      <geom name="screwdriver_handle_90" {hg} size="{L:.4f} {t:.4f} {a:.4f}"/>
      <geom name="screwdriver_handle_45" {hg} quat="{hq(45)}" size="{L:.4f} {a:.4f} {t:.4f}"/>
      <geom name="screwdriver_handle_135" {hg} quat="{hq(-45)}" size="{L:.4f} {a:.4f} {t:.4f}"/>
      <geom name="screwdriver_collar" type="cylinder" {sg} fromto="{L:.4f} 0 0  {L + 0.012:.4f} 0 0" size="{r * 2:.4f}"/>
      <geom name="screwdriver_shaft" type="cylinder" {sg} fromto="{L:.4f} 0 0  {shaft_end:.4f} 0 0" size="{r:.4f}"/>
      <geom name="screwdriver_blade" type="box" {sg} pos="{shaft_end + 0.006:.4f} 0 0" size="0.008 {r * 1.5:.4f} {r * 0.8:.4f}"/>
    </body>"""


def _car_quat():
    return _quat((0, 0, 1), CAR_YAW_DEG)


def _car_xml(group):
    x, y = CAR_POS_XY
    R, W = CAR_WHEEL_RADIUS, CAR_WHEEL_HALF_W
    ch, cb = CAR_CHASSIS_HALF, CAR_CABIN_HALF
    chassis_z = 0.006
    cabin_z = chassis_z + ch[2] + cb[2]
    wheel_x, wheel_y = 0.042, ch[1] + W - 0.002
    wq = "0.70710678 0.70710678 0 0"        # cylinder axis z -> y
    wheels = "".join(f"""
      <geom name="car_wheel_{n}" type="cylinder" group="{group}" rgba="{CAR_WHEEL_RGBA}"
            pos="{sx * wheel_x:.4f} {sy * wheel_y:.4f} 0" quat="{wq}" size="{R:.4f} {W:.4f}"/>"""
                     for n, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)))
    return f"""
    <!-- Toy car, nose toward +x (then yawed): chassis, cabin, 4 wheels. -->
    <body name="toy_car" pos="{x:.4f} {y:.4f} {R:.4f}" quat="{' '.join(f'{v:.6f}' for v in _car_quat())}">
      <freejoint/>
      <geom name="car_chassis" type="box" group="{group}" rgba="{CAR_BODY_RGBA}"
            pos="0 0 {chassis_z:.4f}" size="{ch[0]:.4f} {ch[1]:.4f} {ch[2]:.4f}"/>
      <geom name="car_cabin" type="box" group="{group}" rgba="{CAR_BODY_RGBA}"
            pos="-0.010 0 {cabin_z:.4f}" size="{cb[0]:.4f} {cb[1]:.4f} {cb[2]:.4f}"/>{wheels}
    </body>"""


def build_scene_xml(include_robot: bool = True, camera_fovy=CAMERA_FOVY,
                     offwidth=None, offheight=None, scene: str = "default") -> str:
    """Return the full workcell MJCF as a string.

    See the module docstring for why this deliberately emits NO <compiler>
    element: every path here is either absolute (ChArUco texture) or, for the
    robot include, made to work by how the caller compiles the string (see
    _compile_model), not by anything declared inside this XML.

    camera_fovy overrides the depth_cam vertical field of view (degrees).
    offwidth/offheight, when both given, declare the offscreen framebuffer
    size on the <global> visual tag -- required whenever a caller will build
    a mujoco.Renderer larger than MuJoCo's 640x480 default (see
    mujoco_rgbd_node.build_scene for the same override and why).
    scene picks the variant from SCENES; only the clutter differs.
    """
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}, expected one of {SCENES}")
    slab_half_z = SLAB_THICKNESS / 2
    slab_center_z = -slab_half_z
    legs_xml = _legs_xml(TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT,
                         SLAB_THICKNESS, LEG_THICKNESS, LEG_INSET, GROUP_ENV)

    wall_half_x = (TABLE_LENGTH + 2 * WALL_MARGIN_X) / 2
    wall_half_y = WALL_THICKNESS / 2
    wall_half_z = WALL_HEIGHT / 2
    wall_pos_y = TABLE_WIDTH / 2 + WALL_GAP + wall_half_y
    wall_pos_z = -TABLE_HEIGHT + wall_half_z

    bottle_x, bottle_y = BOTTLE_POS_XY
    bottle_body_top_z = 2 * BOTTLE_BODY_HALF_HEIGHT
    bottle_neck_center_z = bottle_body_top_z + BOTTLE_NECK_HALF_HEIGHT

    blocks_xml = _blocks_xml(GROUP_CLUTTER)
    composites_xml = (_tray_xml(GROUP_CLUTTER) + _screwdriver_xml(GROUP_CLUTTER)
                      + _car_xml(GROUP_CLUTTER))
    human_xml = _human_xml(GROUP_CLUTTER)

    include_xml = f'  <include file="{UR5E_NOKEY_NAME}"/>\n' if include_robot else ""

    if offwidth is not None and offheight is not None:
        global_xml = ('<global azimuth="120" elevation="-20" '
                      f'offwidth="{int(offwidth)}" offheight="{int(offheight)}"/>')
    else:
        global_xml = '<global azimuth="120" elevation="-20"/>'

    # Full-length "home" keyframe: 6 robot angles + one (pos, identity quat)
    # 7-vector per freejoint body, in worldbody declaration order -- see the
    # module docstring for why the vendored keyframe cannot simply be reused.
    if scene in ("two_cylinders", "moving_cylinder"):
        human_slot_xml = ""
        clutter_xml = _cylinders_xml(GROUP_CLUTTER)
        if scene == "moving_cylinder":
            clutter_xml += _moving_cylinder_xml(GROUP_CLUTTER)
    else:
        human_slot_xml = f"""
    <!-- Human figure reaching for the bottle; pose via the HUMAN_* constants. -->
{human_xml}"""
        clutter_xml = f"""
    <!-- Drinking glass: one cylinder, free to be nudged by the arm. -->
    <body name="glass" pos="{GLASS_POS_XY[0]:.4f} {GLASS_POS_XY[1]:.4f} {GLASS_HALF_HEIGHT:.4f}">
      <freejoint/>
      <geom name="glass_geom" type="cylinder" group="{GROUP_CLUTTER}"
            size="{GLASS_RADIUS:.4f} {GLASS_HALF_HEIGHT:.4f}" rgba="{GLASS_RGBA}"/>
    </body>

    <!-- Bottle: body cylinder + a narrower neck cylinder, one rigid body. -->
    <body name="bottle" pos="{bottle_x:.4f} {bottle_y:.4f} {BOTTLE_BODY_HALF_HEIGHT:.4f}">
      <freejoint/>
      <geom name="bottle_body_geom" type="cylinder" group="{GROUP_CLUTTER}"
            size="{BOTTLE_BODY_RADIUS:.4f} {BOTTLE_BODY_HALF_HEIGHT:.4f}" rgba="{BOTTLE_RGBA}"/>
      <geom name="bottle_neck_geom" type="cylinder" group="{GROUP_CLUTTER}"
            pos="0 0 {bottle_neck_center_z - BOTTLE_BODY_HALF_HEIGHT:.4f}"
            size="{BOTTLE_NECK_RADIUS:.4f} {BOTTLE_NECK_HALF_HEIGHT:.4f}" rgba="{BOTTLE_RGBA}"/>
    </body>
{blocks_xml}
{composites_xml}

    <!-- ChArUco board: hardcoded pose from the real rig's poses.yaml
         (aruco_1, parent base_link == this world origin). NOT a freejoint --
         a calibration target is bolted at a known, fixed transform. -->
    <body name="charuco_board" pos="{BOARD_POS[0]} {BOARD_POS[1]} {BOARD_POS[2]}"
          quat="{BOARD_QUAT_WXYZ[0]} {BOARD_QUAT_WXYZ[1]:.16f} {BOARD_QUAT_WXYZ[2]:.16f} {BOARD_QUAT_WXYZ[3]}">
      <geom name="charuco_geom" type="box" group="{GROUP_CLUTTER}"
            size="{BOARD_WIDTH / 2:.4f} {BOARD_DEPTH / 2:.4f} {BOARD_THICKNESS / 2:.4f}"
            material="charuco_mat"/>
    </body>"""

    keyframe_xml = ""
    if include_robot:
        identity = (1, 0, 0, 0)
        free_body_home = [
            (GLASS_POS_XY[0], GLASS_POS_XY[1], GLASS_HALF_HEIGHT, identity),
            (bottle_x, bottle_y, BOTTLE_BODY_HALF_HEIGHT, identity),
        ] + [(spec["pos_xy"][0], spec["pos_xy"][1], spec["half"][2], identity)
             for spec in BLOCK_SPECS] + [
            (*TRAY_POS_XY, TRAY_HALF_HEIGHT, identity),
            (*SCREWDRIVER_POS_XY, SCREWDRIVER_HANDLE_HALF_W, identity),
            (*CAR_POS_XY, CAR_WHEEL_RADIUS, tuple(_car_quat())),
        ]
        if scene != "default":
            free_body_home = []   # the other variants have no freejoint bodies
        free_qpos = " ".join(
            f"{x:.4f} {y:.4f} {z:.4f} " + " ".join(f"{v:.6f}" for v in q)
            for x, y, z, q in free_body_home
        )
        keyframe_xml = f"""
  <keyframe>
    <key name="home" qpos="{HOME_QPOS_ROBOT} {free_qpos}" ctrl="{HOME_QPOS_ROBOT}"/>
  </keyframe>"""

    return f"""
<mujoco model="scene">

  <option gravity="0 0 -9.81"/>
{include_xml}
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    {global_xml}
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
             rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.1"/>

    <texture type="2d" name="charuco_tex" file="{CHARUCO_PNG}"/>
    <material name="charuco_mat" texture="charuco_tex" specular="0" shininess="0"/>
  </asset>

  <worldbody>

    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>

    <geom name="floor" type="plane" group="{GROUP_ENV}" pos="0 0 {-TABLE_HEIGHT}"
          size="4 4 0.05" material="groundplane"/>

    <!-- The table -- static (no joint), top surface at z = 0. -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" group="{GROUP_ENV}"
            pos="0 0 {slab_center_z:.4f}"
            size="{TABLE_LENGTH / 2:.4f} {TABLE_WIDTH / 2:.4f} {slab_half_z:.4f}"
            rgba="0.72 0.60 0.44 1"/>
{legs_xml}
    </body>

    <!-- Backdrop wall, past the table's far (+y) edge from the camera's
         viewpoint, so the distance field gets a real background instead of
         empty space. Static, part of the environment. -->
    <body name="wall" pos="0 {wall_pos_y:.4f} {wall_pos_z:.4f}">
      <geom name="wall_geom" type="box" group="{GROUP_ENV}"
            size="{wall_half_x:.4f} {wall_half_y:.4f} {wall_half_z:.4f}"
            rgba="0.55 0.55 0.58 1"/>
    </body>

    <!-- Monitor: listed with the "clutter" in the task write-up, but it
         never moves, so by the environment/non-environment split it belongs
         with the table and wall -- group GROUP_ENV, no freejoint. -->
    <body name="monitor" pos="{MONITOR_POS_XY[0]:.4f} {MONITOR_POS_XY[1]:.4f} {MONITOR_HALF_Z:.4f}">
      <geom name="monitor_geom" type="box" group="{GROUP_ENV}"
            size="{MONITOR_HALF_X:.4f} {MONITOR_HALF_Y:.4f} {MONITOR_HALF_Z:.4f}"
            rgba="{MONITOR_RGBA}"/>
    </body>

{human_slot_xml}

    <site name="robot_mount" pos="{ROBOT_BASE_XY[0]} {ROBOT_BASE_XY[1]} 0" size="0.035" rgba="0.9 0.2 0.2 1"/>

    <camera name="depth_cam" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}"
            fovy="{camera_fovy}" mode="targetbody" target="table"/>

{clutter_xml}
  </worldbody>
{keyframe_xml}
</mujoco>
"""


def _compile_model(xml: str, include_robot: bool) -> mujoco.MjModel:
    """Compile the generated XML, working around the meshdir and keyframe traps.

    Without the robot there is nothing relative to resolve (the ChArUco
    texture path is absolute) and no keyframe-length conflict, so
    from_xml_string is fine. With the robot: write a copy of ur5e.xml with
    its incompatible <keyframe> stripped out, and the generated scene, both
    into ur5e.xml's own directory, and compile the scene from that path --
    see the module docstring for why both steps are necessary.
    """
    if not include_robot:
        return mujoco.MjModel.from_xml_string(xml)

    nokey_path = UR5E_DIR / UR5E_NOKEY_NAME
    scratch = UR5E_DIR / SCENE_SCRATCH_NAME
    ur5e_text = UR5E_XML.read_text()
    ur5e_nokey = re.sub(r"<keyframe>.*?</keyframe>\s*", "", ur5e_text, flags=re.S)
    assert "<keyframe" not in ur5e_nokey, "failed to strip ur5e.xml's keyframe"
    base_tag = '<body name="base" quat="0 0 0 -1"'
    assert ur5e_nokey.count(base_tag) == 1, "ur5e.xml's base body tag changed"
    ur5e_nokey = ur5e_nokey.replace(
        base_tag, f'<body name="base" pos="{ROBOT_BASE_XY[0]} {ROBOT_BASE_XY[1]} 0" quat="0 0 0 -1"')

    nokey_path.write_text(ur5e_nokey)
    scratch.write_text(xml)
    try:
        return mujoco.MjModel.from_xml_path(str(scratch))
    finally:
        scratch.unlink(missing_ok=True)
        nokey_path.unlink(missing_ok=True)


def compile_scene_model(include_robot: bool = True, camera_fovy=CAMERA_FOVY,
                         offwidth=None, offheight=None, scene: str = "default") -> mujoco.MjModel:
    """Build AND compile the workcell in one call -- the entry point other
    scripts should use.

    Do NOT do from_xml_string(build_scene_xml()) instead: the generated MJCF
    <include>s the vendored ur5e.xml, whose meshdir only resolves relative to
    a real file on disk, so compiling from a bare string cannot find the
    meshes. _compile_model handles that.
    """
    xml = build_scene_xml(include_robot=include_robot, camera_fovy=camera_fovy,
                           offwidth=offwidth, offheight=offheight, scene=scene)
    return _compile_model(xml, include_robot)


def apply_environment_only(scene_option, env_only: bool) -> None:
    """Mask geom groups so only the environment + robot render (clutter
    hidden), or all groups do.

    Works on any mujoco.MjvOption -- the viewer's live `viewer.opt`, or one
    handed to mujoco.Renderer for offscreen rendering (what the depth-camera
    ROS node will use).
    """
    if env_only:
        for g in range(mujoco.mjNGROUP):
            scene_option.geomgroup[g] = 1 if g in (GROUP_ENV, GROUP_ROBOT_VISUAL) else 0
    else:
        for g in range(mujoco.mjNGROUP):
            scene_option.geomgroup[g] = 1


def _geom_world_z_range(model, data, gid):
    """World z-range of one geom, from its local AABB rotated into the world
    (exact for boxes and axis-aligned cylinders/capsules, a tight bound
    otherwise). Reads data.geom_xpos/xmat (post mj_forward), never
    model.geom_pos, which is relative to the parent body."""
    center = data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ model.geom_aabb[gid][:3]
    half_z = np.abs(data.geom_xmat[gid].reshape(3, 3)[2]) @ model.geom_aabb[gid][3:]
    return center[2] - half_z, center[2] + half_z


def describe(model, data):
    """Print each geom's WORLD z-range."""
    mujoco.mj_forward(model, data)

    print(f"  bodies : {model.nbody}    geoms : {model.ngeom}    DOF : {model.nv}")
    print()
    print("  geom                world z-range (metres)")
    print("  " + "-" * 48)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"<unnamed:{i}>"
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            print(f"  {name:<19} z = {data.geom_xpos[i][2]:+.3f} (infinite plane)")
            continue
        lo, hi = _geom_world_z_range(model, data, i)
        print(f"  {name:<19} {lo:+.3f} .. {hi:+.3f}")


def _body_geoms(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def _clutter_table_gaps(model, data):
    """{body: gap (m) between its lowest geom and the table top (z = 0)}.
    0 = resting on the table."""
    mujoco.mj_forward(model, data)
    return {name: min(_geom_world_z_range(model, data, g)[0] for g in _body_geoms(model, name))
            for name in CLUTTER_BODIES}


def _clutter_heights(model, data):
    return {n: float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)][2])
            for n in CLUTTER_BODIES}


def _check_human(model, data):
    """Print each human part's nearest non-human geom; fail if the reaching
    hand is closer than HUMAN_REACH_CLEARANCE or any part pokes into something."""
    print("\n  -- human checks (robot, if present, at its home keyframe) --")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    human = [i for i, n in enumerate(names) if n.startswith("human_")]
    others = [i for i, n in enumerate(names) if not n.startswith("human_") and n != "floor"]
    bx, by = human_base_xy()
    print(f"  base (between feet) at x={bx:+.3f} y={by:+.3f}")
    for h in human:
        dist, nearest = min((mujoco.mj_geomDistance(model, data, h, o, 1.0, None), o) for o in others)
        print(f"  {names[h]:<22} nearest {names[nearest] or f'<unnamed:{nearest}>':<20} {dist * 100:6.1f} cm")
        if names[h] == "human_hand_reach":
            assert dist > HUMAN_REACH_CLEARANCE - 0.002, (
                f"reaching hand is only {dist * 100:.1f} cm from {names[nearest]}")
        else:
            assert dist > -0.005, f"{names[h]} penetrates {names[nearest]} by {-dist * 100:.1f} cm"


def _check_robot(model, data):
    print("\n  -- robot checks --")
    expected_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    found = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    missing = [j for j in expected_joints if j not in found]
    print(f"  joints found ({len(found)}): {found}")
    assert not missing, f"missing expected joints: {missing}"
    assert model.nu == 6, f"expected 6 actuators, found {model.nu}"
    print(f"  actuators: {model.nu}  (OK)")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "no 'home' keyframe found"
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    ee_pos = data.site_xpos[site_id].copy()
    print(f"  attachment_site world pos @ home keyframe: "
          f"({ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f})")
    assert ee_pos[2] > 0, (
        "end-effector z <= 0 at home -- the arm is buried in the table, "
        "the mounting convention came out wrong"
    )
    print("  end-effector z > 0  (OK -- arm is above the table, not buried in it)")

    print("\n  -- stability check: stepping 300 steps --")
    before = _clutter_heights(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    after = _clutter_heights(model, data)
    gaps = _clutter_table_gaps(model, data)
    for name in CLUTTER_BODIES:
        drift = after[name] - before[name]
        print(f"  {name:<18} z before={before[name]:+.4f}  after={after[name]:+.4f}  "
              f"(drift {drift:+.4f})")
        assert abs(drift) < 0.01 and gaps[name] > -0.01, (
            f"{name} moved or sank through the table (drift {drift:+.4f}, gap {gaps[name]:+.4f})")


def main():
    include_robot = "--no-robot" not in sys.argv

    xml = build_scene_xml(include_robot=include_robot)
    model = _compile_model(xml, include_robot)
    data = mujoco.MjData(model)

    print(f"Scene compiled OK (include_robot={include_robot}).\n")
    describe(model, data)

    print()
    gaps = _clutter_table_gaps(model, data)
    print("  clutter resting check (gap to table top must start at +0.000):")
    all_zero = True
    for name, gap in gaps.items():
        ok = abs(gap) < 1e-6
        all_zero &= ok
        print(f"    {name:<18} {gap:+.6f}  {'OK' if ok else 'NOT ON TABLE'}")
    assert all_zero, "at least one clutter object is not resting exactly at z=0"

    _check_human(model, data)
    if include_robot:
        _check_robot(model, data)

    print("\n  -- build_scene_xml(include_robot=False) also compiles --")
    xml_no_robot = build_scene_xml(include_robot=False)
    model_no_robot = _compile_model(xml_no_robot, include_robot=False)
    print(f"  OK: {model_no_robot.nbody} bodies, {model_no_robot.ngeom} geoms, "
          f"{model_no_robot.nv} DOF (no robot)")

    print(f"\n  Table top surface : z = 0.000")
    print(f"  Floor             : z = {-TABLE_HEIGHT:.3f}")
    print(f"\n  Viewer key '{ENV_ONLY_TOGGLE_KEY}' toggles environment-only rendering.")

    if "--check" in sys.argv:
        return

    print("\nOpening viewer -- close the window to exit.")
    print(f"Press '{ENV_ONLY_TOGGLE_KEY}' to toggle environment-only rendering.")

    state = {"env_only": False}
    viewer_ref = {}

    def key_callback(keycode):
        if keycode == ord(ENV_ONLY_TOGGLE_KEY):
            state["env_only"] = not state["env_only"]
            v = viewer_ref.get("viewer")
            if v is not None:
                apply_environment_only(v.opt, state["env_only"])
                print(f"[toggle] environment_only = {state['env_only']}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer_ref["viewer"] = viewer
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
