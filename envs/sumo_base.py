"""Shared machinery for Gymnasium environments backed by a SUMO simulation.

``SumoEnvironment`` starts and stops SUMO through TraCI and handles seeding.
Subclasses define the observation space, action space, how actions are sent
to SUMO and how rewards are calculated.
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "scenarios"

_connection_ids = itertools.count()


def import_traci():
    """Import TraCI from the Python environment or from ``$SUMO_HOME/tools``.

    TraCI is imported lazily so that the environment classes (and the unit
    tests that replace SUMO with a stand-in) can be imported without SUMO.
    """
    try:
        import traci
        return traci
    except ImportError:
        pass

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
        try:
            import traci
            return traci
        except ImportError:
            pass

    raise ImportError(
        "Could not import TraCI, SUMO's Python interface.\n"
        "Either install it with `pip install traci sumolib` (or `pip install eclipse-sumo traci`),\n"
        "or set SUMO_HOME to your SUMO installation so that $SUMO_HOME/tools can be used.\n"
        f"SUMO_HOME is currently {'set to ' + repr(sumo_home) if sumo_home else 'not set'}."
    )


def find_sumo_tool(name: str) -> str:
    """Return the full path of a SUMO executable such as ``sumo`` or ``netconvert``.

    The system PATH is searched first, then ``$SUMO_HOME/bin``.
    """
    found = shutil.which(name)
    if found:
        return found
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        found = shutil.which(name, path=os.path.join(sumo_home, "bin"))
        if found:
            return found
    raise FileNotFoundError(
        f"Could not find the SUMO program '{name}'. Install SUMO and either add its 'bin' "
        "folder to PATH or set the SUMO_HOME environment variable. See the README's "
        "Installation section."
    )


def network_file_from_config(config_file: Path) -> Path:
    """Read the ``net-file`` entry of a ``.sumocfg`` file and resolve its path."""
    root = ET.parse(config_file).getroot()
    entry = root.find("./input/net-file")
    if entry is None:
        raise ValueError(f"{config_file} does not define <input><net-file value=.../></input>.")
    return (config_file.parent / entry.get("value")).resolve()


class SumoEnvironment(gym.Env):
    """Base class: one SUMO process per environment instance.

    Every call to ``reset()`` closes the previous simulation (if any) and
    starts a new one from the scenario's ``.sumocfg`` file. SUMO receives a
    fresh random seed drawn from the environment's own random generator, so
    a run seeded with ``reset(seed=...)`` is reproducible while successive
    episodes still see different traffic.

    Subclasses implement ``on_reset()``, ``get_observation()``, ``get_info()``
    and ``step()``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_file: str | Path, gui: bool = False, sumo_options: list[str] | None = None):
        super().__init__()
        self.config_file = Path(config_file).resolve()
        if not self.config_file.is_file():
            raise FileNotFoundError(f"SUMO configuration file not found: {self.config_file}")
        self.network_file = network_file_from_config(self.config_file)
        if not self.network_file.is_file():
            raise FileNotFoundError(
                f"SUMO network file not found: {self.network_file}\n"
                "The networks are generated from plain XML sources. Run:\n"
                "    python scenarios/build_networks.py"
            )
        self.gui = gui
        self.sumo_options = list(sumo_options or [])
        self.sumo = None            # TraCI connection while a simulation is running
        self.step_length = 1.0      # seconds per SUMO step, read from SUMO at start

    # --------------------------------------------------------------- lifecycle

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.close_simulation()
        self.start_simulation(sumo_seed=int(self.np_random.integers(0, 2**31 - 1)))
        self.on_reset()
        return self.get_observation(), self.get_info()

    def sumo_command(self, sumo_seed: int) -> list[str]:
        """Build the command line used to launch SUMO."""
        command = [
            find_sumo_tool("sumo-gui" if self.gui else "sumo"),
            "-c", str(self.config_file),
            "--seed", str(sumo_seed),
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]
        if self.gui:
            # Start the simulation immediately and close the window at the end.
            command += ["--start", "true", "--quit-on-end", "true"]
        return command + self.sumo_options

    def start_simulation(self, sumo_seed: int) -> None:
        traci = import_traci()
        command = self.sumo_command(sumo_seed)
        # Each simulation gets its own TraCI label, so several environments
        # (or successive episodes) never share a connection.
        label = f"{type(self).__name__}-{next(_connection_ids)}"
        try:
            traci.start(command, label=label)
        except Exception as error:
            raise RuntimeError(
                "SUMO failed to start. Command was:\n  " + " ".join(command) +
                "\nRun that command in a terminal to see SUMO's own error message."
            ) from error
        self.sumo = traci.getConnection(label)
        self.step_length = float(self.sumo.simulation.getDeltaT())

    def close_simulation(self) -> None:
        if self.sumo is not None:
            try:
                self.sumo.close()
            except Exception:
                # SUMO may already have exited (e.g. the GUI window was closed).
                pass
            self.sumo = None

    def close(self) -> None:
        self.close_simulation()

    # ----------------------------------------------------------------- helpers

    @property
    def simulation_time(self) -> float:
        return float(self.sumo.simulation.getTime())

    def steps_for(self, seconds: float) -> int:
        """Number of SUMO steps that make up ``seconds`` of simulated time."""
        return max(1, int(round(seconds / self.step_length)))

    def check_action(self, action) -> int:
        if self.sumo is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; the action space is {self.action_space}.")
        return int(action)

    # ------------------------------------------------------ subclass interface

    def on_reset(self) -> None:
        """Prepare episode state after SUMO has started."""
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_info(self) -> dict:
        raise NotImplementedError
