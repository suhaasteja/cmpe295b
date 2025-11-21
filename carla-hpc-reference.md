# Guide: Running CARLA 0.9.15 on HPC (Rocky Linux)

**Target System:** HPC Cluster (Rocky Linux 9)
**Tools:** Apptainer (Singularity), Python 3.9
**Storage:** Scratch Directory (Lustre Filesystem)

-----

## Phase 1: Preparation & Building the Image

**Crucial Note:** Do not attempt to install or build CARLA in your `/home` directory. The memory/storage quota is insufficient. You must use the scratch partition.

1.  **Navigate to your Scratch Folder**
    Find your professor's folder, then your specific user directory:

    ```bash
    cd /scratch/PROFESSOR_NAME/YOUR_USERNAME/
    ```

2.  **Check Storage Space**
    Ensure you have at least 20GB free.

    ```bash
    df -h .
    ```

3.  **Configure Apptainer Temporary Storage**
    The build process requires massive temporary space. If you don't redirect this, the build will fail with "No space left on device" because the system `/tmp` is too small.

    ```bash
    mkdir -p apptainer_tmp apptainer_cache
    export APPTAINER_TMPDIR=$(pwd)/apptainer_tmp
    export APPTAINER_CACHEDIR=$(pwd)/apptainer_cache
    ```

4.  **Build the CARLA Container**
    This pulls the Docker image and converts it to a Singularity Image File (`.sif`).
    *Time estimate: 20-30 minutes.*

    ```bash
    apptainer build carla_0.9.15.sif docker://carlasim/carla:0.9.15
    ```

    **Success Output:**

    > `INFO: Creating SIF file...`
    > `INFO: Build complete: carla_0.9.15.sif`

-----

## Phase 2: Setting up Python Dependencies (The Offline Fix)

We need to run the **Server** inside the container, but the **Client** (your Python scripts) on the Host machine (`hpc3`).

> **Why do we have to install pygame/numpy manually?**
> The container is "Read-Only" and contains an old version of Python (3.6). Your Host machine (`hpc3`) uses a newer, better Python (3.9). Because we are running the client script on the Host (to avoid version conflicts), the Host's Python needs the libraries installed. Since `hpc3` has no internet, we must download them on `hpc1` first.

### Step A: Download Files on Login Node (`hpc1`)

1.  SSH into `hpc1` (Login Node).
2.  Navigate to your scratch folder:
    ```bash
    cd /scratch/PROFESSOR_NAME/YOUR_USERNAME/
    ```
3.  Create a folder for the offline files:
    ```bash
    mkdir -p carla_offline_files
    cd carla_offline_files
    ```
4.  **Create a Virtual Environment for Downloading:**
    We do this to bypass system permission locks and outdated pip versions on the login node.
    ```bash
    python3 -m venv temp_downloader
    source temp_downloader/bin/activate
    pip install --upgrade pip
    ```
5.  **Download the Packages:**
    Run this specific command to grab files compatible with the GPU node.
    ```bash
    python3 -m pip download carla==0.9.15 pygame numpy \
    --python-version 3.9 \
    --only-binary=:all: \
    --platform manylinux_2_27_x86_64 \
    --platform manylinux2014_x86_64 \
    --dest .
    ```

> **Why use two platforms (`manylinux_2_27` and `manylinux2014`)?**
> Rocky Linux 9 is strict. `manylinux_2_27` is the exact match for the OS, but the `pygame` library hasn't published a file with that exact tag—they use the older standard `manylinux2014`. By listing both, we tell pip: *"Get the perfect match if it exists (CARLA), but accept the slightly older standard (PyGame) if that's all you have."* This prevents the "No matching distribution found" error.

6.  **Clean up:**
    ```bash
    deactivate
    rm -rf ../temp_downloader
    ```

### Step B: Install Files on GPU Node (`hpc3`)

1.  SSH into `hpc3` (Compute Node).
2.  Navigate to the folder:
    ```bash
    cd /scratch/PROFESSOR_NAME/YOUR_USERNAME/carla_offline_files/
    ```
3.  Install the downloaded wheels using the system Python:
    ```bash
    /usr/bin/python3 -m pip install *.whl
    ```

-----

## Phase 3: Running the Simulation

You need **two** terminal windows connected to `hpc3` (the same node).

### Terminal 1: The Server

Navigate to your scratch folder and start the simulator.

```bash
apptainer exec --nv carla_0.9.15.sif /home/carla/CarlaUE4.sh -RenderOffScreen -nosound -opengl -quality-level=Low
```

  * **Note:** It may take \~30-60 seconds to initialize.
  * **Status:** If the terminal "hangs" (stops printing text), the server is running successfully. **Do not close this window.**

### Terminal 2: The Client

1.  **Extract Examples:**
    Pull the example scripts out of the container so we can use them.
    ```bash
    cd /scratch/PROFESSOR_NAME/YOUR_USERNAME/
    apptainer exec carla_0.9.15.sif cp -r /home/carla/PythonAPI/examples ./carla_examples
    ```
2.  **Run Traffic Generation:**
    ```bash
    cd carla_examples
    /usr/bin/python3 generate_traffic.py
    ```
    **Success Output:** `spawned 30 vehicles and 8 walkers`

### Troubleshooting

  * **"Carla is not found":** Run `apptainer exec carla_0.9.15.sif python3 --version`. If it is not 3.9, you are running inside the container by mistake. Ensure you run `/usr/bin/python3` on the host.
  * **Connection Timeout / Port Error:** The server might be stuck in a "zombie" state. Run this command to force-kill old servers, then restart Terminal 1:
    ```bash
    killall -9 CarlaUE4-Linux-Shipping
    ```

-----

## Phase 4: Capturing Data (Headless Mode)

Since we cannot open a window to see the cars, we use a script to save camera frames.

1.  Create the script `capture_images.py` inside `carla_examples`:

    ```python
    import glob
    import os
    import sys
    import time
    import carla

    def main():
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()

        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        
        if vehicle is None:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[1])

        vehicle.set_autopilot(True)

        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('sensor_tick', '1.0') 

        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

        output_folder = 'output_images'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Recording images to {output_folder}...")
        camera.listen(lambda image: image.save_to_disk(f'{output_folder}/{image.frame}.png'))

        time.sleep(10)

        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("Done.")

    if __name__ == '__main__':
        main()
    ```

2.  **Run the capture:**

    ```bash
    /usr/bin/python3 capture_images.py
    ```

-----

## Phase 5: Viewing Results (Local Machine)

To view the images, transfer them from the HPC to your local computer.

**Run this on your Laptop's terminal (not the HPC):**

```bash
scp -r YOUR_USERNAME@coe-hpc1.sjsu.edu:/scratch/PROFESSOR_NAME/YOUR_USERNAME/carla_examples/output_images ~/Desktop/
```

  * Enter your password when prompted.
  * Open the `output_images` folder on your Desktop to view the simulation results.## Documentation: Running CARLA on Rocky Linux HPC via Apptainer

**System Environment:** Rocky Linux 9 (HPC Cluster)
**CARLA Version:** 0.9.15
**Container Tool:** Apptainer (formerly Singularity) v1.3.6

This workflow solves common HPC issues: small `/home` quotas, lack of root access, and offline compute nodes.

-----

### Phase 1: Build the Simulator Image

*Perform this on a node with internet access (e.g., Login Node).*

**1. Prepare Storage (Avoid `/home`)**
CARLA is large (\~20GB). Use your scratch directory.

```bash
# Navigate to your scratch space

cd .. and ls until you see /scratch folder
cd /scratch/cmpe295-junliu/016197935
df -h . to check avaialble space.

# Create a clean project folder
mkdir carla_project
cd carla_project
```

**2. Redirect Temporary Folders**
To prevent "No space left on device" errors in `/tmp` during the build:

```bash
mkdir -p apptainer_tmp apptainer_cache
export APPTAINER_TMPDIR=$(pwd)/apptainer_tmp
export APPTAINER_CACHEDIR=$(pwd)/apptainer_cache
```

**3. Build the Image**
Pull the official Docker image and convert it to a Singularity Image File (`.sif`).

we use apptainer as hpc runs on Rocky Linux but carla only supports windows and ubuntu

we use pre built docker images to build carla

```bash
apptainer build carla_0.9.15.sif docker://carlasim/carla:0.9.15
```

-----

### Phase 2: Install Python Client Libraries (Offline Method)

*The container is read-only, so the client runs on the Host. Since Compute Nodes often lack internet, we download libraries on the Login Node and install them on the Compute Node.*

**1. Download Wheels (On Login Node)**
Create a temporary environment to bypass system lock issues and download files specifically for **Python 3.9** (the version on the GPU node).
temporary env is important

```bash
# Create a temp downloader environment
python3 -m venv temp_downloader
source temp_downloader/bin/activate
pip install --upgrade pip

# Create folder for wheels
mkdir -p carla_offline_files
cd carla_offline_files

# Download specific wheels for Python 3.9 Linux
python3 -m pip download carla==0.9.15 pygame numpy \
--python-version 3.9 \
--only-binary=:all: \
--platform manylinux_2_27_x86_64 \
--platform manylinux2014_x86_64 \
--dest .

# Cleanup
deactivate
rm -rf ../temp_downloader
```

**2. Install Wheels (On GPU Node)**
Log into your GPU node (e.g., `g17`), navigate to the folder, and install using the **system Python**.

```bash
cd /scratch/your_username/carla_project/carla_offline_files
/usr/bin/python3 -m pip install *.whl
```

-----

### Phase 3: Setup Client Scripts

Extract the Python examples from inside the locked container to your scratch folder so they can be edited and run.

```bash
cd /scratch/your_username/carla_project/
apptainer exec carla_0.9.15.sif cp -r /home/carla/PythonAPI/examples ./carla_examples
```

-----

### Phase 4: Running the Simulation

*Requires TWO separate terminal windows connected to the **same** GPU node.*

should also check is ports 2000 and 2001 are not occupied

#### Terminal 1: The Server (Simulator)

1.  **Clean up old processes:** (Prevents "Port 2000 occupied" errors)
    ```bash
    killall -9 CarlaUE4-Linux-Shipping
    ```
2.  **Start the Server:**
      * `--nv`: Passes GPU to container.
      * `-RenderOffScreen`: Headless mode (no window).
      * `-nosound`: Prevents audio crashes.
    <!-- end list -->
    ```bash
    apptainer exec --nv carla_0.9.15.sif /home/carla/CarlaUE4.sh -RenderOffScreen -nosound
    ```
      * **Wait 30-60 seconds** until the terminal stops printing and "hangs" (listens).

#### Terminal 2: The Client (Python Script)

1.  **Navigate to scripts:**
    ```bash
    cd /scratch/your_username/carla_project/carla_examples
    ```
2.  **Run the traffic generation script:**
      * Use `/usr/bin/python3` to match the libraries we installed.
      * Increase timeout to prevent disconnects on load.
    <!-- end list -->
    ```bash
    /usr/bin/python3 generate_traffic.py --timeout 60.0
    ```

-----

### Phase 5: Saving Data (Optional)

To save camera images from the headless simulation:

**1. Create the Script (`capture.py`)**
*See previous chat history for full code.*

**2. Run Data Collection**

```bash
/usr/bin/python3 capture.py
```

**3. Download Results to Laptop**
Run this **on your local computer**:

```bash
scp -r your_user@hpc.edu:/scratch/your_user/carla_project/carla_examples/output_images ~/Desktop/
```
