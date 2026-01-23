# Deployment Guide for odi.cioosatlantic.ca

This document describes how the Ocean Data Imputation JupyterHub was deployed and configured. For detailed instructions on any step, refer to [The Littlest JupyterHub (TLJH) documentation](https://tljh.jupyter.org/).

## Infrastructure

The service runs on the Arbutus cloud via the Digital Research Alliance of Canada.

| Resource | Value |
|----------|-------|
| Cloud console | https://arbutus.cloud.computecanada.ca/ |
| Instance name | `ocean-data-imputation` |
| Flavor | p8-30gb (8 vCPU, 32 GB RAM) |
| Base image | Ubuntu-24.04-Noble-x64-2024-06 |
| Floating IP | Mapped via DNS to `odi.cioosatlantic.ca` |

### Storage

A 500 GB volume named `ocean-data-imputation-home` is attached to the instance as `/dev/vdb` and mounted at `/home`. This provides persistent storage for user home directories.

## Software Setup

### JupyterHub Installation

TLJH was installed following the [custom server instructions](https://tljh.jupyter.org/en/latest/install/custom-server.html).

### HTTPS

HTTPS is configured using Let's Encrypt through TLJH's built-in support.

### Conda Environment

Packages listed in `binder/environment.yml` were installed for all users via the shared conda environment.

### Authentication

Users authenticate with their GitHub usernames. See the [GitHub authentication guide](https://tljh.jupyter.org/en/latest/howto/auth/github.html) for configuration details.

### Content Distribution

Notebooks from this repository are distributed automatically using nbgitpuller. See the [nbgitpuller guide](https://tljh.jupyter.org/en/latest/howto/content/nbgitpuller.html) for setup instructions.

## Decommissioning

When this instance is no longer needed, it is safe to delete:

- The `ocean-data-imputation` instance
- The `ocean-data-imputation-home` volume

No persistent data needs to be retained.
