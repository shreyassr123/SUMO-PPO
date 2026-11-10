# SUMO and TraCI in this repository

This note explains the SUMO files in `scenarios/` and how the environments in
`envs/` drive the simulator. The official documentation is at
<https://sumo.dlr.de/docs/>.

## 1. The pieces

| Piece | What it is | In this repository |
|---|---|---|
| `sumo` | the simulator, without a window | started by the environments for training |
| `sumo-gui` | the same simulator with a graphical interface | started with `--gui`, or by hand |
| `netconvert` | builds a network file from simpler descriptions | called by `scenarios/build_networks.py` |
| `netedit` | graphical network editor | optional, for drawing new networks |
| TraCI | Python library to control a running simulation | `import_traci()` in `envs/sumo_base.py` |

## 2. Network files

SUMO models roads as a directed graph. **Nodes** are junctions or road ends;
**edges** are roads between two nodes, in one direction, with one or more
**lanes**. A two-way street is two edges.

The repository keeps the short source files:

* `*.nod.xml`: node IDs, coordinates in metres and junction types
  (`traffic_light` for the signalised junction, `priority` otherwise);
* `*.edg.xml`: edge IDs, start and end node, number of lanes and speed limit
  in m/s.

`python scenarios/build_networks.py` runs `netconvert` on them and writes
`network.net.xml`. The generated file adds what SUMO needs to simulate:
lane shapes and lengths, **internal lanes** that carry vehicles across a
junction, **connections** (which incoming lane may continue to which outgoing
lane), right-of-way rules, and a default **traffic-light program** (`tlLogic`)
for each signalised junction. Each controlled connection has a `linkIndex`,
its position in the light's signal state string.

The traffic-signal environment reads two things from `network.net.xml`
(`read_signal_layout()`): the green phases of the light's default program, and
its incoming edges with their lane lengths.

## 3. Route files

`traffic.rou.xml` describes the demand:

```xml
<vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" .../>
<route id="north_south" edges="north_in south_out"/>
<flow id="flow_north_south" route="north_south" type="car"
      begin="0" end="3600" probability="0.10" departSpeed="max"/>
```

* A **vType** sets vehicle properties. `accel`/`decel` are the maximum
  acceleration and comfortable deceleration (m/s²), `sigma` (0–1) is driver
  imperfection in the default Krauss car-following model, `minGap` is the gap
  kept to the vehicle ahead when stopped, and `speedFactor` scales the speed
  limit per vehicle (here drawn from a normal distribution, so drivers differ).
* A **route** is a sequence of connected edges.
* A **flow** inserts many vehicles. With `probability="0.10"` a vehicle is
  inserted with probability 0.1 in every second, i.e. on average one every
  10 s, at random times that depend on SUMO's seed. (`period=` would insert at
  fixed intervals instead.)
* A **vehicle** is a single vehicle, like `ego` in the vehicle-control
  scenario.

Vehicles and flows must be listed in order of departure time.

## 4. Configuration files and time steps

`simulation.sumocfg` ties a scenario together:

```xml
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="traffic.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1"/>
    </time>
</configuration>
```

Every command-line option of SUMO can also be written here. The
**step length** is the simulated time per call of `simulationStep()`: here
1 s. The environments read it from SUMO (`getDeltaT()`) and convert durations
such as the 3 s yellow time into a number of steps (`steps_for()`). The
configuration files have no end time: the environments decide when an
episode ends.

## 5. How an environment runs SUMO

`SumoEnvironment.reset()` (in `envs/sumo_base.py`):

1. closes the previous SUMO process, if any;
2. draws a seed for SUMO from the environment's random generator;
3. starts SUMO with `traci.start([sumo, "-c", config, "--seed", seed, ...])`,
   which launches the program, picks a free network port and connects to it;
4. calls `on_reset()` of the concrete environment and returns the first
   observation.

SUMO then waits. Each `simulationStep()` call advances the simulation by one
step and returns control to Python, so the environment can read the state and
change it between steps. `close()` ends the SUMO process.

Every environment instance uses its own TraCI connection object
(`self.sumo`), so several environments can run at once, for example in
tests.

## 6. Traffic-light control

A traffic light's state is a string with one character per controlled
connection (link), in `linkIndex` order:

| Character | Meaning |
|---|---|
| `G` | green, priority |
| `g` | green, must yield (e.g. a left turn across oncoming traffic) |
| `y` | yellow |
| `r` | red |

Normally SUMO runs the light's program from the network file by itself. The
environment instead calls `trafficlight.setRedYellowGreenState(tls_id, state)`,
which replaces the program by the given state until it is changed again. This
gives the environment complete control of timing, and it builds the yellow
state itself (`yellow_state()`): every link that is green now but will not be
green in the next phase shows `y`, followed by an all-red state.

## 7. Vehicle control

`vehicle.setSpeed(vehicle_id, v)` sets the speed a vehicle should drive at,
replacing the speed its car-following model would choose. How strictly SUMO
obeys is controlled by the vehicle's **speed mode**, a set of bit flags. The
default, 31, keeps all safety checks: SUMO never drives faster than the safe
following speed, respects the vehicle's acceleration and deceleration limits,
and respects right of way and red lights. The vehicle environment sets speed
mode 31 explicitly. Speed mode 0 would make the vehicle follow commands even
into a crash; do not use it unless that is what you want to study.

Other TraCI calls used by the vehicle environment: `getSpeed`,
`getAllowedSpeed` (speed limit × the vehicle's speed factor), `getAcceleration`,
`getLeader(id, distance)` (the next vehicle ahead and the gap to it, or
nothing), and `getDrivingDistance` (distance along the route to a position).

A vehicle that has reached its destination is removed from the simulation;
calling TraCI functions with its ID then raises an error. The environment
therefore checks `simulation.getArrivedIDList()` after each step before
querying the vehicle.

## 8. Viewing the scenarios in the GUI

After building the networks:

```bash
sumo-gui -c scenarios/simple_intersection/simulation.sumocfg
sumo-gui -c scenarios/vehicle_control/simulation.sumocfg
```

Press the green *Run* triangle to start. Useful controls:

* **Delay (ms)** in the toolbar slows the simulation down.
* Right-click a vehicle and choose *Start Tracking* to follow it (the
  vehicle-control environment does this automatically for `ego` when run
  with `--gui`).
* Right-click a vehicle, lane or traffic light and choose *Show Parameter* to
  see its current values.
* *Edit → Edit Visualisation* changes how vehicles and lanes are coloured,
  e.g. by speed or waiting time.

When you open the configurations by hand, the traffic light runs the default
fixed-time program that `netconvert` generated, and `ego` is driven by SUMO.
To watch the environments themselves, use the scripts with `--gui`, for
example `python scripts/run_baseline.py --env traffic_signal --controller fixed_time --gui --episodes 1`.

`netedit` (`netedit scenarios/simple_intersection/network.net.xml`) shows the
network's structure, including the link indices of the traffic light. If you
edit a network in netedit, save the `.net.xml` and remember that
`build_networks.py` would overwrite it from the node and edge files.
