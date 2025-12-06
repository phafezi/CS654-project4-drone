"""
Script for testing the Navigation Aviary.

Runs a single drone in your Navigation environment, records the simulation,
and commands the drone to fly in a straight line so you can visually confirm
that obstacles and the target sphere are spawning correctly.

Example
-------
In a terminal, run as:

    $ python test_navigation_aviary.py --gui True --record_video True
"""

import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

# 🔧 TODO: adjust this import to wherever your NavigationAviary class lives
from gym_pybullet_drones.envs.NavigationAviary import NavigationAviary

################################################################################
# Defaults
################################################################################

DEFAULT_DRONE = DroneModel.CF2X
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def run(
    drone=DEFAULT_DRONE,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VIDEO,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
    plot=True,
):
    """Run a straight-line flight to visually test the Navigation/Navigation Aviary."""

    ###########################################################################
    # Initialize the environment
    ###########################################################################
    INIT_XYZS = np.array([[0.0, 0.0, 1.0]])  # start at origin, 1m high
    INIT_RPYS = np.zeros((1, 3))

    env = NavigationAviary(
        drone_model=drone,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=Physics.PYB,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obs=ObservationType.JOINT,
        act=ActionType.RPM,
    )

    # Gymnasium-style reset to make sure obstacles / target are spawned
    obs, _ = env.reset()

    ###########################################################################
    # Straight-line trajectory along +X
    ###########################################################################
    # One waypoint per control step, simple linear ramp in X
    num_steps = int(duration_sec * env.CTRL_FREQ)
    start_pos = INIT_XYZS[0].copy()
    end_pos = start_pos + np.array([50.0, 0.0, 0.0])  # fly 50 m straight in +X

    waypoints = np.zeros((num_steps, 3))
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        waypoints[i, :] = (1 - alpha) * start_pos + alpha * end_pos

    ###########################################################################
    # Logger & controller
    ###########################################################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=1,
        duration_sec=duration_sec,
        output_folder=output_folder,
        colab=colab,
    )

    ctrl = DSLPIDControl(drone_model=drone)
    action = np.zeros((1, 4))

    ###########################################################################
    # Simulation loop
    ###########################################################################
    START = time.time()
    for i in range(num_steps):
        # Current waypoint (straight line)
        target_pos = waypoints[i, :]

        # Step the env with previous action
        obs, reward, terminated, truncated, info = env.step(action)

        print("obs", obs)
        # Compute low-level RPM action to move toward target_pos
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=env._getDroneStateVector(0),
            target_pos=target_pos,
        )

        # Log data
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=env._getDroneStateVector(0),
            control=np.hstack([target_pos, np.zeros(9)]),
        )

        # Render & sync so you can see obstacles/target
        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

        # Optional: break if episode ends early
        if bool(terminated or truncated):
            break

    ###########################################################################
    # Cleanup & save
    ###########################################################################
    env.close()

    logger.save()
    logger.save_as_csv("navigation_test")  # Optional CSV

    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Navigation/Navigation Aviary straight-line test script"
    )
    parser.add_argument(
        "--drone",
        default=DEFAULT_DRONE,
        type=DroneModel,
        help="Drone model (default: CF2X)",
        metavar="",
        choices=DroneModel,
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
        help="Simulation frequency in Hz (default: 240)",
        metavar="",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
        help="Control frequency in Hz (default: 48)",
        metavar="",
    )
    parser.add_argument(
        "--duration_sec",
        default=DEFAULT_DURATION_SEC,
        type=int,
        help="Duration of the simulation in seconds (default: 15)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=str2bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )

    ARGS = parser.parse_args()
    run(**vars(ARGS))