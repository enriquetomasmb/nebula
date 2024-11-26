import logging
import os

import docker


class Utils:
    @classmethod
    def check_path(cls, base_path, relative_path):
        full_path = os.path.normpath(os.path.join(base_path, relative_path))
        base_path = os.path.normpath(base_path)

        if not full_path.startswith(base_path):
            raise Exception("Not allowed")
        return full_path


class DockerUtils:
    @classmethod
    def create_docker_network(cls, network_name, subnet=None, prefix=24):
        try:
            # Connect to Docker
            client = docker.from_env()
            base_subnet = "192.168"

            # Obtain existing docker subnets
            existing_subnets = []
            networks = client.networks.list()
            for network in networks:
                ipam_config = network.attrs.get("IPAM", {}).get("Config", [])
                if ipam_config:
                    for config in ipam_config:
                        logging.info("[FER]1")
                        if "Subnet" in config:
                            existing_subnets.append(config["Subnet"])

            # If no subnet is provided or it exists, find the next available one
            if not subnet or subnet in existing_subnets:
                for i in range(2, 255):  # Iterate over 192.168.2.0 to 192.168.254.0
                    subnet = f"{base_subnet}.{i}.0/{prefix}"
                    potential_base = f"{base_subnet}.{i}"
                    if subnet not in existing_subnets:
                        break
                else:
                    raise ValueError("No available subnets found.")

            # Create the Docker network
            ipam_pool = docker.types.IPAMPool(subnet=subnet)
            ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
            network = client.networks.create(name=network_name, driver="bridge", ipam=ipam_config)

            logging.info(f"Network created: {network.name} with subnet {subnet}")
            return potential_base

        except docker.errors.APIError:
            logging.exception("Error interacting with Docker")
            return None
        except Exception:
            logging.exception("Unexpected error")
            return None
        finally:
            client.close()  # Ensure the Docker client is closed

    @classmethod
    def remove_docker_network(cls, network_name):
        try:
            # Connect to Docker
            client = docker.from_env()

            # Get the network by name
            network = client.networks.get(network_name)

            # Remove the network
            network.remove()

            logging.info(f"Network {network_name} removed successfully.")
        except docker.errors.NotFound:
            logging.exception(f"Network {network_name} not found.")
        except docker.errors.APIError:
            logging.exception("Error interacting with Docker")
        except Exception:
            logging.exception("Unexpected error")

    @classmethod
    def remove_containers_by_prefix(cls, prefix):
        try:
            # Connect to Docker client
            client = docker.from_env()

            containers = client.containers.list(all=True)  # `all=True` to include stopped containers

            # Iterate through containers and remove those with the matching prefix
            for container in containers:
                if container.name.startswith(prefix):
                    logging.info(f"Removing container: {container.name}")
                    container.remove(force=True)  # force=True to stop and remove if running
                    logging.info(f"Container {container.name} removed successfully.")

        except docker.errors.APIError:
            logging.exception("Error interacting with Docker")
        except Exception:
            logging.exception("Unexpected error")
