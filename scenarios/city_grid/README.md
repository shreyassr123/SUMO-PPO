# City-grid scenario

A 3.8 km × 3.6 km grid city with 200 signalised junctions, taken from the
original Mobility_PPO research project. It is used by the
`city_traffic_signal` environment, in which PPO controls junction **517**
while the other 199 signals run their own fixed programs.

| File | Contents | Origin |
|---|---|---|
| `network.net.xml` | road network: two traffic lanes plus a pavement per direction, pedestrian crossings, 200 static signal programs | `random.net.xml` of the original project (netconvert 1.7.0) |
| `activity_demand.rou.xml` | 20,000 people with daily activity chains; about 15,000 private cars, motorbikes and commercial vehicles that leave when their owner does and park between activities | `rou/saga.merged.rou.xml`, generated with SUMOActivityGen (https://github.com/lcodeca/SUMOActivityGen) |
| `buses.rou.xml`, `bus_stops.add.xml` | 6 bus lines, every 5 minutes | original project |
| `metro.rou.xml`, `metro_stops.add.xml` | 4 metro lines | original project |
| `parking_areas.add.xml`, `parking_rerouters.add.xml` | 536 roadside parking areas and the rerouters that send drivers elsewhere when one is full | original project |
| `vehicle_types.add.xml` | vehicle and pedestrian types used by the demand | `basic.vType.xml` of the original project |
| `simulation.sumocfg` | configuration; the day starts at 06:30 | adapted from `random.saga.sumocfg` |

The event-specific files of the original project (event travellers, the venue
and its parking) are not included.

**Junction 517** is the busiest four-way junction in the morning according to
the demand's routes (about 1,270 car passes between 06:30 and 08:00). Its
approaches are `-1259` (from the south), `-1263` (from the west), `-1267`
(from the north) and `518` (from the east), each with two traffic lanes of
about 180 m. A bus stop lies on `-1267`. Its program alternates an
east–west and a north–south green, each followed by a short phase in which
only the pedestrian signals turn red, and a 3 s yellow.

**Simulation cost.** Every person, parked car, bus and metro is simulated,
vehicles reroute around congestion every 5 minutes, and the sublane model
(`lateral-resolution`) lets motorbikes filter between cars. This is far
slower than the teaching intersection. To speed it up at the cost of
realism, remove the `device.rerouting.*` and `person-device.rerouting.*`
lines and the `lateral-resolution` line from `simulation.sumocfg`.

To control a different junction, pass its ID: `make_env("city_traffic_signal", tls_id="783")`.
Any junction with at least two green stages works; T-junctions have three
approaches, so the observation is one value shorter.
