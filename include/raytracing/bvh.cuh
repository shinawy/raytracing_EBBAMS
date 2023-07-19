#pragma once

#include <raytracing/common.h>
#include <raytracing/triangle.cuh>
#include <raytracing/bounding_box.cuh>
#include <raytracing/gpu_memory.h>

#include <memory>

namespace raytracing {

struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx; // negative values indicate leaves
    int right_idx;
};


template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
    __host__ __device__ void push(T val) {
        if (m_count >= MAX_SIZE-1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

using FixedIntStack = FixedStack<int>;


class TriangleBvh {

private: 
    // here should go the max_dist and min_dist declarations

protected:
    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
    TriangleBvh() {};
    
    // max_dist is the maximum distance after which the ray will be considered to have 0 intersections with the mesh
    // min_dist is the minimum distance that filters the faces from which the ray exits
    // without the  minimum distance, the code will count the face from which the ray exits as an intersection face
    TriangleBvh(float attr_max_dist, float attr_min_dist) {};


public:

    float max_dist= 10.0f;
    float min_dist= 0.0001f;
    float max_dist_sq= max_dist*max_dist;


    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;
    // max_dist is the maximum distance after which the ray will be considered to have 0 intersections with the mesh
    // min_dist is the minimum distance that filters the faces from which the ray exits
    // without the  minimum distance, the code will count the face from which the ray exits as an intersection face
    virtual void set_max_min_distance(float max_dist, float min_dist)=0;
    virtual float get_max_distance()=0;
    virtual float get_min_distance()=0;

    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, float* normals, float* depth, float* ray_hit_freq_arr, const Triangle* gpu_triangles, cudaStream_t stream) = 0;

    // KIUI: not supported now.
    // virtual void signed_distance_gpu(uint32_t n_elements, EMeshSdfMode mode, const Eigen::Vector3f* gpu_positions, float* gpu_distances, const Triangle* gpu_triangles, bool use_existing_distances_as_upper_bounds, cudaStream_t stream) = 0;
    // virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
    // virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const {
        return m_nodes_gpu.data();
    }

};

}