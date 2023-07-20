# RayTracer

A CUDA Mesh RayTracer with BVH acceleration.

### First Install CUDA 11.7
```
#!/bin/bash 

# this bash installs Cuda toolkit 11.7

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-11-7_11.7.1-1_amd64.deb
sudo dpkg -i cuda-11-7_11.7.1-1_amd64.deb
sudo apt-key add /var/cuda-11-7_11.7.1-1_amd64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

```

### Easy, but Not guaranteed Install
You can open the folder called "raytracing_gpu_ready_library" and copy the contents of it and paste inside of the virtual environment site-packages. Then you should move to the next step, if it did not work, use normal install. 


### Normal Install

```python
git clone https://github.com/shinawy/raytracing_EBBAMS
cd raytracing_EBBAMS
pip install .
```

### Usage
You should first export the path of the repo folder in the repo in PYTHONPATH variable using:
```
export PYTHONPATH="/path/to/raytracing_EBBAMS/:$PYTHONPATH"
```
Example for a mesh normal renderer:

```bash
python renderer.py # default, show a dodecahedron
```


Example code:

```python
import numpy as np
import trimesh

import torch
import raytracing


def gpu_ray_tracer(gpu_raytracer, part_tricenters, part_fnorm):
    gpu_intersections, gpu_fn, gpu_depth, gpu_rhq = gpu_raytracer.trace(
        torch.from_numpy(part_tricenters),
        torch.from_numpy(part_fnorm),
    )

    gpu_int = gpu_intersections.cpu().data.numpy()
    gpu_dep_arr = gpu_depth.cpu().data.numpy()
    gpu_fn_arr = gpu_fn.cpu().data.numpy()
    gpu_ray_hit_freq = gpu_rhq.cpu().data.numpy()

    return gpu_int, gpu_dep_arr, gpu_ray_hit_freq

def main():
  part_trimesh: trimesh.Trimesh = trimesh.load_mesh(part_filename)
  zmin: float = part_trimesh.bounds[0][-1]
  zmax: float = part_trimesh.bounds[1][-1]
  part_height: float = zmax - zmin
  max_dist: float = part_height
  min_dist: float = 0.0001
  support_ray_tracer = raytracing.RayTracer(
      part_trimesh.vertices, part_trimesh.faces, max_dist, min_dist
  )
  gpu_int, gpu_dep_arr, gpu_ray_hit_freq = gpu_ray_tracer(
          support_ray_tracer, tri_centers, face_normals
      )

```



### Acknowledgement
* This is a forked repository from the original Repo:: https://github.com/ashawkey/raytracing
* Credits to [Thomas Müller](https://tom94.net/)'s amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp)!
