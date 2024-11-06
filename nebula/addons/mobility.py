import asyncio
import logging
import math
import random
import time
from typing import TYPE_CHECKING

from nebula.addons.functions import print_msg_box

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Mobility:
    def __init__(self, config, cm: "CommunicationsManager"):
        """
        Initializes the mobility module with specified configuration and communication manager.

        This method sets up the mobility parameters required for the module, including grace time,
        geographical change interval, mobility type, and other network conditions based on distance.
        It also logs the initialized settings for the mobility system.

        Args:
            config (Config): Configuration object containing mobility parameters and settings.
            cm (CommunicationsManager): An instance of the CommunicationsManager class used for handling
                                         communication-related tasks within the mobility module.

        Attributes:
            grace_time (float): Time allocated for mobility processes to stabilize.
            period (float): Interval at which geographic changes are made.
            mobility (bool): Flag indicating whether mobility is enabled.
            mobility_type (str): Type of mobility strategy to be used (e.g., random, nearest).
            radius_federation (float): Radius for federation in meters.
            scheme_mobility (str): Scheme to be used for managing mobility.
            round_frequency (int): Number of rounds after which mobility changes are applied.
            max_distance_with_direct_connections (float): Maximum distance for direct connections in meters.
            max_movement_random_strategy (float): Maximum movement distance for the random strategy in meters.
            max_movement_nearest_strategy (float): Maximum movement distance for the nearest strategy in meters.
            max_initiate_approximation (float): Maximum distance for initiating approximation calculations.
            network_conditions (dict): A dictionary containing network conditions (bandwidth and delay)
                                       based on distance.
            current_network_conditions (dict): A dictionary mapping addresses to their current network conditions.

        Logs:
            Mobility information upon initialization to provide insights into the current setup.

        Raises:
            KeyError: If the expected mobility configuration keys are not found in the provided config.
        """
        logging.info("Starting mobility module...")
        self.config = config
        self.cm = cm
        self.grace_time = self.config.participant["mobility_args"]["grace_time_mobility"]
        self.period = self.config.participant["mobility_args"]["change_geo_interval"]
        self.mobility = self.config.participant["mobility_args"]["mobility"]
        self.mobility_type = self.config.participant["mobility_args"]["mobility_type"]
        self.radius_federation = float(self.config.participant["mobility_args"]["radius_federation"])
        self.scheme_mobility = self.config.participant["mobility_args"]["scheme_mobility"]
        self.round_frequency = int(self.config.participant["mobility_args"]["round_frequency"])
        # Protocol to change connections based on distance
        self.max_distance_with_direct_connections = 300  # meters
        self.max_movement_random_strategy = 100  # meters
        self.max_movement_nearest_strategy = 100  # meters
        self.max_initiate_approximation = self.max_distance_with_direct_connections * 1.2
        # Network conditions based on distance
        self.network_conditions = {
            100: {"bandwidth": "5Gbps", "delay": "5ms"},
            200: {"bandwidth": "2Gbps", "delay": "50ms"},
            300: {"bandwidth": "100Mbps", "delay": "200ms"},
            float("inf"): {"bandwidth": "10Mbps", "delay": "1000ms"},
        }
        # Current network conditions of each connection {addr: {bandwidth: "5Gbps", delay: "0ms"}}
        self.current_network_conditions = {}
        # Logging box with mobility information
        mobility_msg = f"Mobility: {self.mobility}\nMobility type: {self.mobility_type}\nRadius federation: {self.radius_federation}\nScheme mobility: {self.scheme_mobility}\nEach {self.round_frequency} rounds"
        print_msg_box(msg=mobility_msg, indent=2, title="Mobility information")

    @property
    def round(self):
        """
        Gets the current round number from the Communications Manager.

        This property retrieves the current round number that is being managed by the
        CommunicationsManager instance associated with this module. It provides an
        interface to access the ongoing round of the communication process without
        directly exposing the underlying method in the CommunicationsManager.

        Returns:
            int: The current round number managed by the CommunicationsManager.
        """
        return self.cm.get_round()

    async def start(self):
        """
        Initiates the mobility process by starting the associated task.

        This method creates and schedules an asynchronous task to run the
        `run_mobility` coroutine, which handles the mobility operations
        for the module. It allows the mobility operations to run concurrently
        without blocking the execution of other tasks.

        Returns:
            asyncio.Task: An asyncio Task object representing the scheduled
                           `run_mobility` operation.
        """
        task = asyncio.create_task(self.run_mobility())
        return task

    async def run_mobility(self):
        """
        Executes the mobility operations in a continuous loop.

        This coroutine manages the mobility behavior of the module. It first
        checks whether mobility is enabled. If mobility is not enabled, the
        function returns immediately.

        If mobility is enabled, the function will wait for the specified
        grace time before entering an infinite loop where it performs the
        following operations:

        1. Changes the geographical location by calling the `change_geo_location` method.
        2. Adjusts connections based on the current distance by calling
           the `change_connections_based_on_distance` method.
        3. Sleeps for a specified period (`self.period`) before repeating the operations.

        This allows for periodic updates to the module's geographical location
        and network connections as per the defined mobility strategy.

        Raises:
            Exception: May raise exceptions if `change_geo_location` or
                        `change_connections_based_on_distance` encounters errors.
        """
        if not self.mobility:
            return
        await asyncio.sleep(self.grace_time)
        while True:
            await self.change_geo_location()
            await self.change_connections_based_on_distance()
            await asyncio.sleep(self.period)

    async def change_geo_location_random_strategy(self, latitude, longitude):
        """
        Changes the geographical location of the entity using a random strategy.

        This coroutine modifies the current geographical location by randomly
        selecting a new position within a specified radius around the given
        latitude and longitude. The new location is determined using polar
        coordinates, where a random distance (radius) and angle are calculated.

        Args:
            latitude (float): The current latitude of the entity.
            longitude (float): The current longitude of the entity.

        Raises:
            Exception: May raise exceptions if the `set_geo_location` method encounters errors.

        Notes:
            - The maximum movement distance is determined by `self.max_movement_random_strategy`.
            - The calculated radius is converted from meters to degrees based on an approximate
              conversion factor (1 degree is approximately 111 kilometers).
        """
        logging.info("📍  Changing geo location randomly")
        # radius_in_degrees = self.radius_federation / 111000
        max_radius_in_degrees = self.max_movement_random_strategy / 111000
        radius = random.uniform(0, max_radius_in_degrees)  # noqa: S311
        angle = random.uniform(0, 2 * math.pi)  # noqa: S311
        latitude += radius * math.cos(angle)
        longitude += radius * math.sin(angle)
        await self.set_geo_location(latitude, longitude)

    async def change_geo_location_nearest_neighbor_strategy(
        self, distance, latitude, longitude, neighbor_latitude, neighbor_longitude
    ):
        """
        Changes the geographical location of the entity towards the nearest neighbor.

        This coroutine updates the current geographical location by calculating the direction
        and distance to the nearest neighbor's coordinates. The movement towards the neighbor
        is scaled based on the distance and the maximum movement allowed.

        Args:
            distance (float): The distance to the nearest neighbor.
            latitude (float): The current latitude of the entity.
            longitude (float): The current longitude of the entity.
            neighbor_latitude (float): The latitude of the nearest neighbor.
            neighbor_longitude (float): The longitude of the nearest neighbor.

        Raises:
            Exception: May raise exceptions if the `set_geo_location` method encounters errors.

        Notes:
            - The movement is scaled based on the maximum allowed distance defined by
              `self.max_movement_nearest_strategy`.
            - The angle to the neighbor is calculated using the arctangent of the difference in
              coordinates to determine the direction of movement.
            - The conversion from meters to degrees is based on approximate geographical conversion factors.
        """
        logging.info("📍  Changing geo location towards the nearest neighbor")
        scale_factor = min(1, self.max_movement_nearest_strategy / distance)
        # Calcular el ángulo hacia el vecino
        angle = math.atan2(neighbor_longitude - longitude, neighbor_latitude - latitude)
        # Conversión de movimiento máximo a grados
        max_lat_change = self.max_movement_nearest_strategy / 111000  # Cambio en grados para latitud
        max_lon_change = self.max_movement_nearest_strategy / (
            111000 * math.cos(math.radians(latitude))
        )  # Cambio en grados para longitud
        # Aplicar escala y dirección
        delta_lat = max_lat_change * math.cos(angle) * scale_factor
        delta_lon = max_lon_change * math.sin(angle) * scale_factor
        # Actualizar latitud y longitud
        new_latitude = latitude + delta_lat
        new_longitude = longitude + delta_lon
        await self.set_geo_location(new_latitude, new_longitude)

    async def set_geo_location(self, latitude, longitude):
        """
        Sets the geographical location of the entity to the specified latitude and longitude.

        This coroutine updates the latitude and longitude values in the configuration. If the
        provided coordinates are out of bounds (latitude must be between -90 and 90, and
        longitude must be between -180 and 180), the previous location is retained.

        Args:
            latitude (float): The new latitude to set.
            longitude (float): The new longitude to set.

        Raises:
            None: This function does not raise any exceptions but retains the previous coordinates
                  if the new ones are invalid.

        Notes:
            - The new location is logged for tracking purposes.
            - The coordinates are expected to be in decimal degrees format.
        """

        if latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180:
            # If the new location is out of bounds, we keep the old location
            latitude = self.config.participant["mobility_args"]["latitude"]
            longitude = self.config.participant["mobility_args"]["longitude"]

        self.config.participant["mobility_args"]["latitude"] = latitude
        self.config.participant["mobility_args"]["longitude"] = longitude
        logging.info(f"📍  New geo location: {latitude}, {longitude}")

    async def change_geo_location(self):
        """
        Changes the geographical location of the entity based on the current mobility strategy.

        This coroutine checks the mobility type and decides whether to move towards the nearest neighbor
        or change the geo location randomly. It uses the communications manager to obtain the current
        connections and their distances.

        If the number of undirected connections is greater than directed connections, the method will
        attempt to find the nearest neighbor and move towards it if the distance exceeds a certain threshold.
        Otherwise, it will randomly change the geo location.

        Args:
            None: This function does not take any arguments.

        Raises:
            Exception: If the neighbor's location or distance cannot be found.

        Notes:
            - The method expects the mobility type to be either "topology" or "both".
            - It logs actions taken during the execution for tracking and debugging purposes.
        """
        if self.mobility and (self.mobility_type == "topology" or self.mobility_type == "both"):
            random.seed(time.time() + self.config.participant["device_args"]["idx"])
            latitude = float(self.config.participant["mobility_args"]["latitude"])
            longitude = float(self.config.participant["mobility_args"]["longitude"])

            direct_connections = await self.cm.get_direct_connections()
            undirect_connection = await self.cm.get_undirect_connections()
            if len(undirect_connection) > len(direct_connections):
                logging.info("📍  Undirect Connections is higher than Direct Connections")
                # Get neighbor closer to me
                selected_neighbor = await self.cm.get_nearest_connections(top=1)
                logging.info(f"📍  Selected neighbor: {selected_neighbor}")
                try:
                    (
                        neighbor_latitude,
                        neighbor_longitude,
                    ) = selected_neighbor.get_geolocation()
                    distance = selected_neighbor.get_neighbor_distance()
                    if distance > self.max_initiate_approximation:
                        # If the distance is too big, we move towards the neighbor
                        await self.change_geo_location_nearest_neighbor_strategy(
                            distance,
                            latitude,
                            longitude,
                            neighbor_latitude,
                            neighbor_longitude,
                        )
                    else:
                        await self.change_geo_location_random_strategy(latitude, longitude)
                except Exception as e:
                    logging.info(f"📍  Neighbor location/distance not found for {selected_neighbor.get_addr()}: {e}")
                    await self.change_geo_location_random_strategy(latitude, longitude)
            else:
                await self.change_geo_location_random_strategy(latitude, longitude)
        else:
            logging.error(f"📍  Mobility type {self.mobility_type} not implemented")
            return

    async def change_connections_based_on_distance(self):
        """
        Changes the connections of the entity based on the distance to neighboring nodes.

        This coroutine evaluates the current connections in the topology and adjusts their status to
        either direct or undirected based on their distance from the entity. If a neighboring node is
        within a certain distance, it is marked as a direct connection; otherwise, it is marked as
        undirected.

        Additionally, it updates the network conditions for each connection based on the distance,
        ensuring that the current state is reflected accurately.

        Args:
            None: This function does not take any arguments.

        Raises:
            KeyError: If a connection address is not found during the process.
            Exception: For any other errors that may occur while changing connections.

        Notes:
            - The method expects the mobility type to be either "topology" or "both".
            - It logs the distance evaluations and changes made for tracking and debugging purposes.
        """
        if self.mobility and (self.mobility_type == "topology" or self.mobility_type == "both"):
            try:
                # logging.info(f"📍  Checking connections based on distance")
                connections_topology = await self.cm.get_addrs_current_connections()
                # logging.info(f"📍  Connections of the topology: {connections_topology}")
                if len(connections_topology) < 1:
                    # logging.error(f"📍  Not enough connections for mobility")
                    return
                # Nodes that are too far away should be marked as undirected connections, and closer nodes should be marked as directed connections.
                for addr in connections_topology:
                    distance = self.cm.connections[addr].get_neighbor_distance()
                    if distance is None:
                        # If the distance is not found, we skip the node
                        continue
                    # logging.info(f"📍  Distance to node {addr}: {distance}")
                    if (
                        not self.cm.connections[addr].get_direct()
                        and distance < self.max_distance_with_direct_connections
                    ):
                        logging.info(f"📍  Node {addr} is close enough [{distance}], adding to direct connections")
                        self.cm.connections[addr].set_direct(True)
                    else:
                        # 10% margin to avoid oscillations
                        if (
                            self.cm.connections[addr].get_direct()
                            and distance > self.max_distance_with_direct_connections * 1.1
                        ):
                            logging.info(
                                f"📍  Node {addr} is too far away [{distance}], removing from direct connections"
                            )
                            self.cm.connections[addr].set_direct(False)
                    # Adapt network conditions of the connection based on distance
                    for threshold in sorted(self.network_conditions.keys()):
                        if distance < threshold:
                            conditions = self.network_conditions[threshold]
                            break
                    # Only update the network conditions if they have changed
                    if (
                        addr not in self.current_network_conditions
                        or self.current_network_conditions[addr] != conditions
                    ):
                        # eth1 is the interface of the container that connects to the node network - eth0 is the interface of the container that connects to the frontend/backend
                        self.cm._set_network_conditions(
                            interface="eth1",
                            network=addr.split(":")[0],
                            bandwidth=conditions["bandwidth"],
                            delay=conditions["delay"],
                            delay_distro="10ms",
                            delay_distribution="normal",
                            loss="0%",
                            duplicate="0%",
                            corrupt="0%",
                            reordering="0%",
                        )
                        self.current_network_conditions[addr] = conditions
            except KeyError:
                # Except when self.cm.connections[addr] is not found (disconnected during the process)
                logging.exception(f"📍  Connection {addr} not found")
                return
            except Exception:
                logging.exception("📍  Error changing connections based on distance")
                return

    async def change_connections(self):
        """
        Changes the connections of the entity based on the specified mobility scheme.

        This coroutine evaluates the current and potential connections at specified intervals (based
        on the round frequency) and makes adjustments according to the mobility scheme in use. If
        the mobility type is appropriate and the current round is a multiple of the round frequency,
        it will proceed to change connections.

        Args:
            None: This function does not take any arguments.

        Raises:
            None: This function does not raise exceptions, but it logs errors related to connection counts
            and unsupported mobility schemes.

        Notes:
            - The function currently supports a "random" mobility scheme, where it randomly selects
            a current connection to disconnect and a potential connection to connect.
            - If there are insufficient connections available, an error will be logged.
            - All actions and decisions made by the function are logged for tracking purposes.
        """
        if (
            self.mobility
            and (self.mobility_type == "topology" or self.mobility_type == "both")
            and self.round % self.round_frequency == 0
        ):
            logging.info("📍  Changing connections")
            current_connections = await self.cm.get_addrs_current_connections(only_direct=True)
            potential_connections = await self.cm.get_addrs_current_connections(only_undirected=True)
            logging.info(
                f"📍  Current connections: {current_connections} | Potential future connections: {potential_connections}"
            )
            if len(current_connections) < 1 or len(potential_connections) < 1:
                logging.error("📍  Not enough connections for mobility")
                return

            if self.scheme_mobility == "random":
                random_neighbor = random.choice(current_connections)  # noqa: S311
                random_potential_neighbor = random.choice(potential_connections)  # noqa: S311
                logging.info(f"📍  Selected node(s) to disconnect: {random_neighbor}")
                logging.info(f"📍  Selected node(s) to connect: {random_potential_neighbor}")
                await self.cm.disconnect(random_neighbor, mutual_disconnection=True)
                await self.cm.connect(random_potential_neighbor, direct=True)
                logging.info(f"📍  New connections: {self.get_current_connections(only_direct=True)}")
                logging.info(f"📍  Neighbors in config: {self.config.participant['network_args']['neighbors']}")
            else:
                logging.error(f"📍  Mobility scheme {self.scheme_mobility} not implemented")
                return
