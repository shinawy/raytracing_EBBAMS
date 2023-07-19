
import numpy as np
import torch

# CUDA extension
import _raytracing as _backend

class RayTracer():
    def __init__(self, vertices, triangles, max_dist, min_dist):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        self.impl = _backend.create_raytracer(vertices, triangles, max_dist, min_dist)

    def trace(self, rays_o, rays_d, inplace=False):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]
        # inplace: write positions to rays_o, face_normals to rays_d

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        if not inplace:
            # allocate
            positions = torch.empty_like(rays_o)
            face_normals = torch.empty_like(rays_d)
        else:
            positions = rays_o
            face_normals = rays_d

        depth = torch.empty_like(rays_o[:, 0])
        ray_hit_freq_arr= torch.zeros_like(rays_o[:, 0])
        
        # inplace write intersections back to rays_o
        self.impl.trace(rays_o, rays_d, positions, face_normals, depth, ray_hit_freq_arr) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_normals = face_normals.view(*prefix, 3)
        depth = depth.view(*prefix)
        ray_hit_freq_arr= ray_hit_freq_arr.view(*prefix)
        # print ("_raytracing location: {}".format(_backend.__file__))


        return positions, face_normals, depth, ray_hit_freq_arr