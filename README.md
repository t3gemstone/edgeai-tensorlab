<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset=".meta/logo-dark.png" width="40%" />
        <source media="(prefers-color-scheme: light)" srcset=".meta/logo-light.png" width="40%" />
        <img alt="T3 Foundation" src=".meta/logo-light.png" width="40%" />
    </picture>
</p>

# T3 Gemstone edgeai-tensorlab

 [![T3 Foundation](./.meta/t3-foundation.svg)](https://www.t3vakfi.org/en) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Built with Distrobox](https://img.shields.io/badge/Built_with-distrobox-red)](https://github.com/89luca89/distrobox) [![Built with Devbox](https://www.jetify.com/img/devbox/shield_galaxy.svg)](https://www.jetify.com/devbox/docs/contributor-quickstart/) [![Documentation](https://img.shields.io/badge/Documentation-gray?style=flat&logo=Mintlify)](https://docs.t3gemstone.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/t3gemstone/edgeai-tensorlab)

## What is it?

This project includes all the necessary work for develop the vision process model for T3 Gemstone boards.

All details related to the project can be found at https://docs.t3gemstone.org/en/visionprocesses/edgeaitensorlab. Below, only a summary of how to perform the installation is provided.

##### 1. Install Docker and jetify-devbox on the host computer.

```bash
user@host:$ ./setup.sh
```

<a name="section-ii"></a>
##### 2. After the installation is successful, activate the jetify-devbox shell to automatically install tools such as Distrobox, taskfile, etc.

```bash
user@host:$ devbox shell
```

##### 3. Download the repositories, create a Docker image, and enter it.

```bash
📦 devbox:edgeai-tensorlab> task box
```

##### 4. Installs the necessary packages and develop new vision process model.

```bash
# Installs the necessary packages
🚀 distrobox:workdir> task install

# Runs the Label Studio
🚀 distrobox:workdir> task label-studio

# Runs the Model Maker
🚀 distrobox:workdir> task make CONFIG=config_classification.yaml

```

# Troubleshooting

#### 1. First Installation of Docker

Docker is installed on your system via the `./setup.sh` command. If you are installing Docker for the first time, you must log out and log in again after the installation is complete.

#### 2. Failed `task box` command

```sh
📦 devbox:edgeai-tensorlab> task box
task: Failed to run task "box": exit status 1
Error: An error occurred
```

To figure out what exact problem is, run `distrobox-enter --additional-flags "--tty" --name gemstone-edgeai --no-workdir --verbose`

```sh
*** update-locale: Error: invalid locale settings:  LC_ALL=en_EN.UTF-8 LANG=en_EN.UTF-8
```

To solve this problem, try to update locales

```bash
📦 devbox:edgeai-tensorlab> sudo dpkg-reconfigure locales 
```

