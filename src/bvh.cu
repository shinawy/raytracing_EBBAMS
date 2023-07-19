// #include <Eigen/Dense>
#include <raytracing/common.h>
#include <raytracing/triangle.cuh>
#include <raytracing/bvh.cuh>

#include <stack>
#include <iostream>
#include <bits/stdc++.h>
#include <cstdio>

using namespace Eigen;
using namespace raytracing;

namespace raytracing
{

    constexpr float MAX_DIST = 10.0f;
    constexpr float MAX_DIST_SQ = MAX_DIST * MAX_DIST;
    constexpr float MIN_DIST = 0.0001f;


    __global__ void raytrace_kernel(uint32_t n_elements, const Vector3f *__restrict__ rays_o, const Vector3f *__restrict__ rays_d, Vector3f *__restrict__ positions, Vector3f *__restrict__ normals, float *__restrict__ depth, float* __restrict__ ray_hit_freq_arr, const TriangleBvhNode *__restrict__ nodes, const Triangle *__restrict__ triangles, float max_dist, float min_dist);

    struct DistAndIdx
    {
        float dist;
        uint32_t idx;

        // Sort in descending order!
        __host__ __device__ bool operator<(const DistAndIdx &other)
        {
            return dist < other.dist;
        }
    };

    template <typename T>
    __host__ __device__ void inline compare_and_swap(T &t1, T &t2)
    {
        if (t1 < t2)
        {
            T tmp{t1};
            t1 = t2;
            t2 = tmp;
        }
    }

    // Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
    template <uint32_t N, typename T>
    __host__ __device__ void sorting_network(T values[N])
    {
        static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
        if (N <= 1)
        {
            return;
        }
        else if (N == 2)
        {
            compare_and_swap(values[0], values[1]);
        }
        else if (N == 3)
        {
            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[1], values[2]);
        }
        else if (N == 4)
        {
            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[3]);
            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[1], values[2]);
        }
        else if (N == 5)
        {
            compare_and_swap(values[0], values[3]);
            compare_and_swap(values[1], values[4]);

            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[3]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[4]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);

            compare_and_swap(values[2], values[3]);
        }
        else if (N == 6)
        {
            compare_and_swap(values[0], values[5]);
            compare_and_swap(values[1], values[3]);
            compare_and_swap(values[2], values[4]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);

            compare_and_swap(values[0], values[3]);
            compare_and_swap(values[2], values[5]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);
        }
        else if (N == 7)
        {
            compare_and_swap(values[0], values[6]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);

            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[4]);
            compare_and_swap(values[3], values[6]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[5]);
            compare_and_swap(values[3], values[4]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[4], values[6]);

            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);
            compare_and_swap(values[5], values[6]);
        }
        else if (N == 8)
        {
            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[3]);
            compare_and_swap(values[4], values[6]);
            compare_and_swap(values[5], values[7]);

            compare_and_swap(values[0], values[4]);
            compare_and_swap(values[1], values[5]);
            compare_and_swap(values[2], values[6]);
            compare_and_swap(values[3], values[7]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);
            compare_and_swap(values[6], values[7]);

            compare_and_swap(values[2], values[4]);
            compare_and_swap(values[3], values[5]);

            compare_and_swap(values[1], values[4]);
            compare_and_swap(values[3], values[6]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);
            compare_and_swap(values[5], values[6]);
        }
    }

    template <uint32_t BRANCHING_FACTOR>
    class TriangleBvhWithBranchingFactor : public TriangleBvh
    {
    public:
        __host__ __device__ static std::tuple<int, float, int> ray_intersect(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode *__restrict__ bvhnodes, const Triangle *__restrict__ triangles, float max_dist, float min_dist)
        {
            FixedIntStack query_stack;
            query_stack.push(0);

            // float mint = MAX_DIST;
            float mint = max_dist ;

            int shortest_idx = -1;
            int ray_hit_frequency = 0;

            while (!query_stack.empty())
            {
                int idx = query_stack.pop();

                const TriangleBvhNode &node = bvhnodes[idx];

                if (node.left_idx < 0)
                {
                    int end = -node.right_idx - 1;
                    for (int i = -node.left_idx - 1; i < end; ++i)
                    {
                        float t = triangles[i].ray_intersect(ro, rd);

                        if (t < MAX_DIST && t> min_dist)
                            ray_hit_frequency= ray_hit_frequency + 1;
                            // ray_hit_frequency= -1;


                        if (t < mint)
                        {
                            mint = t;
                            shortest_idx = i;
                        }
                    }
                }
                else
                {
                    DistAndIdx children[BRANCHING_FACTOR];

                    uint32_t first_child = node.left_idx;

#pragma unroll
                    for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i)
                    {
                        children[i] = {bvhnodes[i + first_child].bb.ray_intersect(ro, rd).x(), i + first_child};
                    }

                    sorting_network<BRANCHING_FACTOR>(children);

#pragma unroll
                    for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i)
                    {
                        if (children[i].dist < mint)
                        {
                            query_stack.push(children[i].idx);
                        }
                    }
                }
            }

            return {shortest_idx, mint, ray_hit_frequency};
        }

        void ray_trace_gpu(uint32_t n_elements, const float *rays_o, const float *rays_d, float *positions, float *normals, float *depth, float* ray_hit_freq_arr, const Triangle *gpu_triangles, cudaStream_t stream) override
        {

            
            

            
            // cast float* to Vector3f*
            const Vector3f *rays_o_vec = (const Vector3f *)rays_o;
            const Vector3f *rays_d_vec = (const Vector3f *)rays_d;
            Vector3f *positions_vec = (Vector3f *)positions;
            Vector3f *normals_vec = (Vector3f *)normals;
            // float max_dist= this ->get_max_distance(); 
            // float min_dist= this -> get_min_distance(); 
            float max_dist= MAX_DIST; 
            float min_dist= MIN_DIST; 

            // #ifdef NGP_OPTIX
            //         if (m_optix.available) {
            //             m_optix.raytrace->invoke({rays_o_vec, rays_d_vec, gpu_triangles, m_optix.gas->handle()}, {n_elements, 1, 1}, stream);
            //         } else
            // #endif //NGP_OPTIX
            
            {

          
                linear_kernel(raytrace_kernel, 0, stream,
                              n_elements,
                              rays_o_vec,
                              rays_d_vec,
                              positions_vec,
                              normals_vec,
                              depth,
                              ray_hit_freq_arr,
                              m_nodes_gpu.data(),
                              gpu_triangles,
                               max_dist,
                               min_dist);
            }
        }

        void set_max_min_distance(float par_max_dist, float par_min_dist) override
        {

            this -> max_dist = par_max_dist;
            this -> min_dist = par_min_dist;
            this -> max_dist_sq = par_max_dist * par_max_dist;
           


        }
        float  get_max_distance() override
        {

           return this -> max_dist; 

        }
        float  get_min_distance() override
        {

           
            return this -> min_dist;

        }
        void build(std::vector<Triangle> &triangles, uint32_t n_primitives_per_leaf) override
        {
            m_nodes.clear();

            // Root
            m_nodes.emplace_back();
            m_nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));

            struct BuildNode
            {
                int node_idx;
                std::vector<Triangle>::iterator begin;
                std::vector<Triangle>::iterator end;
            };

            std::stack<BuildNode> build_stack;
            build_stack.push({0, std::begin(triangles), std::end(triangles)});

            while (!build_stack.empty())
            {
                const BuildNode &curr = build_stack.top();
                size_t node_idx = curr.node_idx;

                std::array<BuildNode, BRANCHING_FACTOR> children;
                children[0].begin = curr.begin;
                children[0].end = curr.end;

                build_stack.pop();

                // Partition the triangles into the children
                int n_children = 1;
                while (n_children < BRANCHING_FACTOR)
                {
                    for (int i = n_children - 1; i >= 0; --i)
                    {
                        auto &child = children[i];

                        // Choose axis with maximum standard deviation
                        Vector3f mean = Vector3f::Zero();
                        for (auto it = child.begin; it != child.end; ++it)
                        {
                            mean += it->centroid();
                        }
                        mean /= (float)std::distance(child.begin, child.end);

                        Vector3f var = Vector3f::Zero();
                        for (auto it = child.begin; it != child.end; ++it)
                        {
                            Vector3f diff = it->centroid() - mean;
                            var += diff.cwiseProduct(diff);
                        }
                        var /= (float)std::distance(child.begin, child.end);

                        Vector3f::Index axis;
                        var.maxCoeff(&axis);

                        auto m = child.begin + std::distance(child.begin, child.end) / 2;
                        std::nth_element(child.begin, m, child.end, [&](const Triangle &tri1, const Triangle &tri2)
                                         { return tri1.centroid(axis) < tri2.centroid(axis); });

                        children[i * 2].begin = children[i].begin;
                        children[i * 2 + 1].end = children[i].end;
                        children[i * 2].end = children[i * 2 + 1].begin = m;
                    }

                    n_children *= 2;
                }

                // Create next build nodes
                m_nodes[node_idx].left_idx = (int)m_nodes.size();
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i)
                {
                    auto &child = children[i];
                    assert(child.begin != child.end);
                    child.node_idx = (int)m_nodes.size();

                    m_nodes.emplace_back();
                    m_nodes.back().bb = BoundingBox(child.begin, child.end);

                    if (std::distance(child.begin, child.end) <= n_primitives_per_leaf)
                    {
                        m_nodes.back().left_idx = -(int)std::distance(std::begin(triangles), child.begin) - 1;
                        m_nodes.back().right_idx = -(int)std::distance(std::begin(triangles), child.end) - 1;
                    }
                    else
                    {
                        build_stack.push(child);
                    }
                }
                m_nodes[node_idx].right_idx = (int)m_nodes.size();
            }

            m_nodes_gpu.resize_and_copy_from_host(m_nodes);

            // std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;
        }

        TriangleBvhWithBranchingFactor() {}
    };

    using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

    std::unique_ptr<TriangleBvh> TriangleBvh::make()
    {
        return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
    }

    __global__ void raytrace_kernel(uint32_t n_elements, const Vector3f *__restrict__ rays_o, const Vector3f *__restrict__ rays_d, Vector3f *__restrict__ positions, Vector3f *__restrict__ normals, float *__restrict__ depth, float* __restrict__ ray_hit_freq_arr,const TriangleBvhNode *__restrict__ nodes, const Triangle *__restrict__ triangles, float max_dist, float min_dist)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        Vector3f ro = rays_o[i];
        Vector3f rd = rays_d[i];


        auto ret_tuple = TriangleBvh4::ray_intersect(ro, rd, nodes, triangles, max_dist, min_dist);

        int shortest_idx = std::get<0>(ret_tuple);
        float depth_var = std::get<1>(ret_tuple);
        int ray_hit_freqency = std::get<2>(ret_tuple);


        // write frequency
        ray_hit_freq_arr[i]= ray_hit_freqency;
        // write depth
        depth[i] = depth_var;

        // intersection point is written back to positions.
        // non-intersect point reaches at most 10 depth
        positions[i] = ro + depth_var * rd;

        // face normal is written to directions.
        if (shortest_idx >= 0)
        {
            normals[i] = triangles[shortest_idx].normal();
        }
        else
        {
            normals[i].setZero();
        }

        // shall we write the depth? (depth_var)
    }

}