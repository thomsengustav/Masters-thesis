//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "optixTriangle.h"
#include <cuda/helpers.h>
#include "random.h"

#include <sutil/vec_math.h>


extern "C" {
__constant__ Params params;
}



struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))// fabs: absolute value af input (double)
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}


static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void setPayload(perRayData prd) //fætter der gemmer ting til ray payload, så værdier kan bruges på tværs af programmer
{
    optixSetPayload_0( __float_as_uint( prd.farve.x ) );
    optixSetPayload_1( __float_as_uint( prd.farve.y ) );
    optixSetPayload_2( __float_as_uint( prd.farve.z ) );

    optixSetPayload_3(__float_as_uint(prd.origin.x));
    optixSetPayload_4(__float_as_uint(prd.origin.y));
    optixSetPayload_5(__float_as_uint(prd.origin.z));

    optixSetPayload_6(__float_as_uint(prd.direction.x));
    optixSetPayload_7(__float_as_uint(prd.direction.y));
    optixSetPayload_8(__float_as_uint(prd.direction.z));

    optixSetPayload_9(prd.seed);
    optixSetPayload_10(prd.depth);
    optixSetPayload_11(prd.done);
    optixSetPayload_12(__float_as_uint(prd.afstand));
    
}

static __forceinline__ __device__ perRayData loadPayload() {
    perRayData prd = {};
    
    prd.farve.x = __uint_as_float(optixGetPayload_0());
    prd.farve.y = __uint_as_float(optixGetPayload_1());
    prd.farve.z = __uint_as_float(optixGetPayload_2());

    //prd.origin.x = __uint_as_float(optixGetPayload_3());
    //prd.origin.y = __uint_as_float(optixGetPayload_4());
    //prd.origin.z = __uint_as_float(optixGetPayload_5());

    //prd.direction.x = __uint_as_float(optixGetPayload_6());
    //prd.direction.y = __uint_as_float(optixGetPayload_7());
    //prd.direction.z = __uint_as_float(optixGetPayload_8());

    prd.seed = optixGetPayload_9();
    prd.depth = optixGetPayload_10();
    //prd.done = optixGetPayload_11();
    prd.afstand = __uint_as_float(optixGetPayload_12());
    return prd;
}

//Værdier der skal hentes til PRD fra payload
static __forceinline__ __device__ perRayData loadPayloadMS() {
    perRayData prd = {};
    prd.farve.x = __uint_as_float(optixGetPayload_0());
    prd.farve.y = __uint_as_float(optixGetPayload_1());
    prd.farve.z = __uint_as_float(optixGetPayload_2());

    prd.origin.x = __uint_as_float(optixGetPayload_3());
    prd.origin.y = __uint_as_float(optixGetPayload_4());
    prd.origin.z = __uint_as_float(optixGetPayload_5());

    //prd.direction.x = __uint_as_float(optixGetPayload_6());
    //prd.direction.y = __uint_as_float(optixGetPayload_7());
    //prd.direction.z = __uint_as_float(optixGetPayload_8());

    prd.seed = optixGetPayload_9();
    prd.depth = optixGetPayload_10();
    //prd.done = optixGetPayload_11();
    prd.afstand = __uint_as_float(optixGetPayload_12());
    
    return prd;
}
//Udsender rays til et pinhole camera
static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction, unsigned int seed)
{
    // The center of each pixel is at fraction (0.5,0.5) jitter
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
    //##################################################################
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            (static_cast<float>( idx.x) + subpixel_jitter.x) / static_cast<float>( dim.x ),
            (static_cast<float>( idx.y) + subpixel_jitter.y) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}

static __forceinline__ __device__ void computePlanar(uint3 idx, uint3 dim, float3& origin, float3& direction, unsigned int seed) {
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

    const float3 v1 = { 0.0f,1.0f,0.0f };
    const float3 v2 = normalize(-params.cam_eye);
    const float3 e1 = normalize(v1 - dot(v1, v2) / dot(v2, v2) * v2);
    const float3 u3 = normalize(cross(e1, v2));

    float dl = 0.06f;
    float lx = static_cast<float>(dim.x) * dl;
    float ly = static_cast<float>(dim.y) * dl;


    origin = params.cam_eye + e1 * ((static_cast<float>(idx.y)+ subpixel_jitter.y)*dl-ly/2 ) + u3 * ((static_cast<float>(idx.x) + subpixel_jitter.x)*dl-lx/2 );
    direction = v2;


}


static __forceinline__ __device__ void computeSphericalWaveFront(uint3 idx, uint3 dim, float3& origin, float3& direction, unsigned int seed) {
    // The center of each pixel is at fraction (0.5,0.5) jitter
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
    //##################################################################
    const float theta_0 = params.theta_look;
    const float phi_0 = params.phi_look;
    const float vinkelIntValx = M_PIf/2 ;
    const float vinkelIntValy = M_PIf/2 ;
    const float2 d = make_float2(
        static_cast<float>((idx.x+subpixel_jitter.x)* vinkelIntValx) / static_cast<float>(dim.x)- vinkelIntValx/2+theta_0,
        //static_cast<float>(-(idx.y- static_cast<float>(dim.y) )*M_PIf/100) / static_cast<float>(dim.y)-M_PIf/100/2+phi_0);
        static_cast<float>((idx.y+subpixel_jitter.y) * vinkelIntValy) / static_cast<float>(dim.y) - vinkelIntValy / 2 + phi_0);

    
    
    origin = params.cam_eye;
    float3 direction3 ={ sinf(d.x) * cosf(d.y),sinf(d.x) * sinf(d.y),cosf(d.x) };
    direction = normalize(direction3);
    //direction = normalize(direction);
}



static __forceinline__ __device__ void computeSphericalWaveFront2(uint3 idx, uint3 dim, float3& origin, float3& direction, unsigned int seed) {
    // The center of each pixel is at fraction (0.5,0.5) jitter
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
    //##################################################################
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;

    const float vinkelIntValx = M_PIf / 60.0f/5.0f;
    const float vinkelIntValy = M_PIf / 120.0f/5.0f;
    const float2 d = make_float2(
        (static_cast<float>(idx.x)+subpixel_jitter.x ) * vinkelIntValx / static_cast<float>(dim.x) + M_PIf / 2.0f - vinkelIntValx / 2.0f, (static_cast<float>(idx.y ) + subpixel_jitter.y) * vinkelIntValy / static_cast<float>(dim.y) + M_PIf / 2.0f - vinkelIntValy / 2.0f);
        //static_cast<float>(-(idx.y- static_cast<float>(dim.y) )*M_PIf/100) / static_cast<float>(dim.y)-M_PIf/100/2+phi_0);
    
    float3 angle = make_float3(cos(d.x) * sin(d.y),
        -cos(d.y),
        sin(d.x) * sin(d.y));


    origin = params.cam_eye;
    direction = normalize(angle.x * normalize(U) +
        angle.y * normalize(V) +
        angle.z * normalize(W));
    
    //direction = normalize(direction);
}

//Efter et registreret miss, dvs ray ikke længere rammer mesh, så tjekker vi om den intersecter med en sphere, som er vores "detector"
static __forceinline__ __device__ float computeDetection(float3 origin, float3 direction)
{
    const float3 c = params.detector_center;
    const float R = params.detector_radius;
    float3 r0 = origin;
    float3 v = direction;
    float3 r0mc = r0 - c;
    float D = (2.0f * dot(v, r0mc))*(2.0f * dot(v, r0mc))-4.0f*(dot(r0mc, r0mc)-R*R);
        
    return D;
}
//regner en ray ud fra origin og direction, kalder miss eller closests hit
static __forceinline__ __device__ void traceRay(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    perRayData& prd
)
{
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12;
    u0 = __float_as_uint(prd.farve.x);
    u1 = __float_as_uint(prd.farve.y);
    u2 = __float_as_uint(prd.farve.z);
    //u5 = 0;
    //u3 = 0;
    //u4 = 0;
    //u6 = 0;
    //u7 = 0;
    //u8 = 0;
    u9 = prd.seed;
    u10 = prd.depth;
    //u11 = 0;
    u12 = __float_as_uint(prd.afstand);
    

    // Note:
    // This demonstrates the usage of the OptiX shader execution reordering 
    // (SER) API.  In the case of this computationally simple shading code, 
    // there is no real performance benefit.  However, with more complex shaders
    // the potential performance gains offered by reordering are significant.
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                     // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,                        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        0,                        // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8,u9,u10,u11,u12);
    //optixReorder(
        // Application specific coherence hints could be passed in here
    //);

    optixInvoke(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12);
    
    prd.farve = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd.origin = make_float3(__uint_as_float(u3), __uint_as_float(u4), __uint_as_float(u5));
    prd.direction = make_float3(__uint_as_float(u6), __uint_as_float(u7), __uint_as_float(u8));
    
    prd.seed = u9; //smid flere værdier her  
    prd.depth = u10;
    prd.done = u11;
    prd.afstand = __uint_as_float(u12);
}

// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                         // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        0                          // missSBTIndex
    );
    return optixHitObjectIsHit();
}




extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    //int indexd = params.index_rnd;
    //++params.index_rnd;
    
    
    

    
    int i = params.samples_per_launch;
    
    float3 result = make_float3(0.0f);
    do {
        // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
        
        unsigned int seed = tea<4>(idx.y * params.image_width + idx.x+i, clock()*i);

        float3 ray_origin, ray_direction;
        computeSphericalWaveFront2(idx, dim, ray_origin, ray_direction,seed);
        perRayData prd = {};
        prd.seed = seed;
        prd.depth = 0;
        prd.farve.z = 1;
        prd.farve.y = 1;


        float P_pol[3][3] = { {0,0,0},{0,0,0},{0,0,0} }; //P matricen til pol
        float(*pP_pol)[3][3] = &P_pol;
        
        float afstand22 = 0;
        int k = 0;
        
        do {
            //prd.afstand = afstand22;
            traceRay(params.handle,
                ray_origin,
                ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                prd);


            
            //++done;
            const bool done = prd.done;
            if (done) {
                break;
            }
            //break;
            // const int depth = prd.depth;
             //if (depth > 2)
             //    break;
            ray_origin = prd.origin;
            ray_direction = prd.direction;
            prd.depth += 1;
            
        } while (prd.depth<5);
        
        result.x += prd.farve.x;
        //result.y += prd.farve.y;
        //result.z += prd.farve.z;
        
    } while (--i);
    
    
    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result );
    
    
}


extern "C" __global__ void __miss__ms()
{
    optixSetPayloadTypes(
        OPTIX_PAYLOAD_TYPE_DEFAULT);
    perRayData prd = loadPayloadMS();
    const float3 direction = optixGetWorldRayDirection();
    const float3 origin = optixGetWorldRayOrigin();

    const float D = computeDetection(origin, direction);
    const float3 c = params.detector_center;
    float3 udret = c - origin;
    udret = normalize(udret);

    if (D >= 0 && dot(udret, direction) > 0) {

        float afstandT = prd.afstand + length(c - origin);
        
        int index = (afstandT-970-4500*2-params.detector_radius) / params.afstand_step_rangeVector;
        int* dd_range_vector = reinterpret_cast<int*>(params.rangeVector);
        int* dd_range_vector_point_til_index = &dd_range_vector[index];
        int et = 1;
        atomicAdd(dd_range_vector_point_til_index, et);

        
        float addRealClut = 1 * prd.farve.z;
        float* dd_range_vectorRealClut = reinterpret_cast<float*>(params.rangeVectorRealClut);
        float* dd_range_vector_RealClut_point_til_index = &dd_range_vectorRealClut[index];
        atomicAdd(dd_range_vector_RealClut_point_til_index, addRealClut);

        //float addImagClut = 1 * prd.farve.y;
        //float* dd_range_vectorImagClut = reinterpret_cast<float*>(params.rangeVectorImagClut);
        //float* dd_range_vector_ImagClut_point_til_index = &dd_range_vectorImagClut[index];
        //atomicAdd(dd_range_vector_ImagClut_point_til_index, addImagClut);


        prd.farve = { 0.1,0.0,0.0 };






    }
    else {
        prd.farve = { 0,0,0 };
    }
    
    prd.done = true;
    setPayload(prd);
    
}







extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    //const float2 barycentrics = optixGetTriangleBarycentrics();

    //setPayload( make_float3( barycentrics, 1.0f ) );
    optixSetPayloadTypes(
        OPTIX_PAYLOAD_TYPE_DEFAULT);
    perRayData prd = loadPayload();
    
    
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer(); //henter data omkring ramt trekant
    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int sbtIdx = optixGetSbtGASIndex();

    const int matIndex = rt_data->materialIndex;
    
    

    const int    prim_idx = optixGetPrimitiveIndex(); //En indgang til en trekant
    const int    vert_idx_offset = prim_idx * 3; // til trekants vertices

    const float3 v0 = make_float3(rt_data->vertices[vert_idx_offset + 0]);//finder de tre vertices til ramt trekant 
    const float3 v1 = make_float3(rt_data->vertices[vert_idx_offset + 1]); 
    const float3 v2 = make_float3(rt_data->vertices[vert_idx_offset + 2]); 

    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));//regner normal til ramt trekant
    
    
    
    if (matIndex == 0) {
        unsigned int clutter_index = prim_idx % params.clutter_triangle_count;
        float* d_arrayZd = reinterpret_cast<float*>(params.d_arrayZ);

        //const float3 ray_dir = optixGetWorldRayDirection();
        float3 P = (v0 + v1 + v2) / 3; //intersectionpoint P=origin+t*retningsvector
        float TMax = length(optixGetWorldRayOrigin() - P); 
        float3 ray_dir = normalize(P - optixGetWorldRayOrigin()); 

        prd.farve.z = prd.farve.z * (d_arrayZd[clutter_index * 2])*params.loss;
        //prd.farve.y = prd.farve.y * (d_arrayZd[clutter_index * 2+1]) * params.loss;

        
        //int index = afstandT / params.afstand_step_rangeVector;
        //float* dd_range_vectorRealClut = reinterpret_cast<float*>(params.rangeVectorRealClut);
        //float* dd_range_vector_RealClut_point_til_index = &dd_range_vectorRealClut[index];
        //float addRealClut = d_arrayZd[clutter_index * 2];
        //atomicAdd(dd_range_vector_RealClut_point_til_index, addRealClut);

        //float* dd_range_vectorImagClut = reinterpret_cast<float*>(params.rangeVectorImagClut);
        //float* dd_range_vector_ImagClut_point_til_index = &dd_range_vectorImagClut[index];
        //float addImagClut = d_arrayZd[clutter_index * 2 + 1];
        //atomicAdd(dd_range_vector_ImagClut_point_til_index, addImagClut); 
        // 
        // 
        const float3 N = faceforward(N_0, -ray_dir, N_0);//smart faceforward fkt, finder normal der peger mod incomming    
        const float3 ray_dir_spec = ray_dir - 2 * dot(ray_dir, N) * N; //perfekt specular reflektion retning

        
        prd.afstand += TMax;

        unsigned int seed = prd.seed;//seed til random number

        if(params.scatterProfile==1){
            const float z1 = rnd(seed); //random number til position på disk/hemisphere
            const float z2 = rnd(seed);

            float3 w_in; //definere position på hemisphere
            cosine_sample_hemisphere(z1, z2, w_in); //regner position på hemisphere (w_in)
            Onb onb(N); // struct til at finde to vectorer orthogonalt på hinanden i trekantens plan
            onb.inverse_transform(w_in);//x-coord på hemisphere ganges med tangent til normal, y med binormal, z med normal til trekant

            prd.direction = normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * ray_dir_spec); //ny retning til ray efter intersection //REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
            prd.origin = P; // intersectionpoint nyt origin
        }

        if (params.scatterProfile == 2) {
            const float z1 = rnd(seed); //random number til position på disk/hemisphere
            const float z2 = rnd(seed);

            float3 w_in; //definere position på hemisphere
            cosine_sample_hemisphere(z1, z2, w_in); //regner position på hemisphere (w_in)
            Onb onb(N); // struct til at finde to vectorer orthogonalt på hinanden i trekantens plan
            onb.inverse_transform(w_in);//x-coord på hemisphere ganges med tangent til normal, y med binormal, z med normal til trekant

            const float3 direction_forward= normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * ray_dir_spec);
            const float3 direction_backward= normalize((1-rt_data->scatter_coef) * w_in + ( rt_data->scatter_coef) * -ray_dir);
            const float forward_prop = 0.8;
            
            const float z3 = rnd(seed);
            float propabilityVal=1;
            if (z3 > forward_prop) {
                propabilityVal = 0;
                prd.farve.z = prd.farve.z * dot(ray_dir, N);

            }
            prd.direction = direction_forward * propabilityVal + direction_backward * (1 - propabilityVal); //ny retning til ray efter intersection //REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
            prd.origin = P; // intersectionpoint nyt origin
        }

        if (params.scatterProfile == 3) {
            const float z1 = rnd(seed); //random number til position på disk/hemisphere
            const float z2 = rnd(seed);

            float3 w_in; //definere position på hemisphere
            cosine_sample_hemisphere(z1, z2, w_in); //regner position på hemisphere (w_in)
            Onb onb(N); // struct til at finde to vectorer orthogonalt på hinanden i trekantens plan
            onb.inverse_transform(w_in);//x-coord på hemisphere ganges med tangent til normal, y med binormal, z med normal til trekant

            const float3 direction_forward = normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * ray_dir_spec);
            const float3 direction_backward = normalize(w_in+N);
            const float forward_prop = 0.8;

            const float z3 = rnd(seed);
            float propabilityVal = 1;
            if (z3 > forward_prop) {
                propabilityVal = 0;

            }
            prd.direction = direction_forward * propabilityVal + direction_backward * (1 - propabilityVal); //ny retning til ray efter intersection //REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
            prd.origin = P; // intersectionpoint nyt origin
        }

        prd.seed = seed;
        prd.done = false;
        
        
        
    }
    else {
        
        //prd.afstand += gauss_fordeling;
        unsigned int seed = prd.seed;//seed til random number
        

        if(params.scatterProfile==1){
            float3 ray_dir = optixGetWorldRayDirection(); 
            float TMax = optixGetRayTmax();
            // Henter en rays retning optixGetWorldRayOrigin findes også
            float3 P = optixGetWorldRayOrigin() + TMax * ray_dir; //intersectionpoint P=origin+t*retningsvector} 

            

            const float3 N = faceforward(N_0, -ray_dir, N_0);//smart faceforward fkt, finder normal der peger mod incomming    
            const float3 ray_dir_spec = ray_dir - 2 * dot(ray_dir, N) * N; //perfekt specular reflektion retning

            prd.farve.z = prd.farve.z * params.loss;
            //prd.farve.y = prd.farve.y * params.loss;
            //const float rand1 = rnd(seed);//rand til gaussisk usikkerhed i afstand
            //const float rand2 = rnd(seed);
            //const float gauss_fordeling = sqrtf(-2 * logf(rand1)) * cosf(2 * 3.14159 * rand2); //gauss fordeling
            prd.afstand += TMax;

            //#####################################################3

            const float z1 = rnd(seed); //random number til position på disk/hemisphere
            const float z2 = rnd(seed);

            float3 w_in; //definere position på hemisphere
            cosine_sample_hemisphere(z1, z2, w_in); //regner position på hemisphere (w_in)
            Onb onb(N); // struct til at finde to vectorer orthogonalt på hinanden i trekantens plan
            onb.inverse_transform(w_in);//x-coord på hemisphere ganges med tangent til normal, y med binormal, z med normal til trekant

            prd.direction = normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * ray_dir_spec); //ny retning til ray efter intersection //REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
            prd.origin = P; // intersectionpoint nyt origin   
        }
        //alpha_scatter(seed, N, prd, P, rt_data, ray_dir_spec);
        if (params.scatterProfile == 2) {
            const float forward_prop = 0.8;
            const float z3 = rnd(seed);
            float propabilityVal = 1;
            float3 P;
            float TMax;
            float3 ray_dir;
            if (z3 > forward_prop) {
                propabilityVal = 0;
                P=(v0 + v1 + v2) / 3;
                TMax = length(optixGetWorldRayOrigin() - P);
                ray_dir = normalize(P - optixGetWorldRayOrigin()); 
                prd.farve.z = prd.farve.z * dot(ray_dir, faceforward(N_0, -ray_dir, N_0)) * dot(ray_dir, faceforward(N_0, -ray_dir, N_0));
            }
            else {
                ray_dir = optixGetWorldRayDirection(); 
                TMax = optixGetRayTmax();
                P = optixGetWorldRayOrigin() + TMax * ray_dir;
            }

            const float3 N = faceforward(N_0, -ray_dir, N_0);//smart faceforward fkt, finder normal der peger mod incomming    
            const float3 ray_dir_spec = ray_dir - 2 * dot(ray_dir, N) * N; //perfekt specular reflektion retning

            const float z1 = rnd(seed); //random number til position på disk/hemisphere
            const float z2 = rnd(seed);

            float3 w_in; //definere position på hemisphere
            cosine_sample_hemisphere(z1, z2, w_in); //regner position på hemisphere (w_in)
            Onb onb(N); // struct til at finde to vectorer orthogonalt på hinanden i trekantens plan
            onb.inverse_transform(w_in);//x-coord på hemisphere ganges med tangent til normal, y med binormal, z med normal til trekant

            const float3 direction_forward = normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * ray_dir_spec);
            const float3 direction_backward = normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * -ray_dir);
            
            prd.afstand += TMax;
            
            prd.origin = P;
            prd.direction = direction_forward * propabilityVal + direction_backward * (1 - propabilityVal); //ny retning til ray efter intersection //REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
           
        }

        if (params.scatterProfile == 3) {
            const float forward_prop = 0.8;
            const float z3 = rnd(seed);
            float propabilityVal = 1;
            float3 P;
            float TMax;
            float3 ray_dir;
            if (z3 > forward_prop) {
                propabilityVal = 0;
                P = (v0 + v1 + v2) / 3;
                TMax = length(optixGetWorldRayOrigin() - P);
                ray_dir = normalize(P - optixGetWorldRayOrigin());
            }
            else {
                ray_dir = optixGetWorldRayDirection();
                TMax = optixGetRayTmax();
                P = optixGetWorldRayOrigin() + TMax * ray_dir;
            }

            const float3 N = faceforward(N_0, -ray_dir, N_0);//smart faceforward fkt, finder normal der peger mod incomming    
            const float3 ray_dir_spec = ray_dir - 2 * dot(ray_dir, N) * N; //perfekt specular reflektion retning

            const float z1 = rnd(seed); //random number til position på disk/hemisphere
            const float z2 = rnd(seed);

            float3 w_in; //definere position på hemisphere
            cosine_sample_hemisphere(z1, z2, w_in); //regner position på hemisphere (w_in)
            Onb onb(N); // struct til at finde to vectorer orthogonalt på hinanden i trekantens plan
            onb.inverse_transform(w_in);//x-coord på hemisphere ganges med tangent til normal, y med binormal, z med normal til trekant

            const float3 direction_forward = normalize(rt_data->scatter_coef * w_in + (1 - rt_data->scatter_coef) * ray_dir_spec);
            const float3 direction_backward = normalize(w_in+N);

            prd.afstand += TMax;

            prd.origin = P;
            prd.direction = direction_forward * propabilityVal + direction_backward * (1 - propabilityVal); //ny retning til ray efter intersection //REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET

        }
        prd.seed = seed;
        prd.done = false;
        
    }
    

        
    
    

    ;
        
    setPayload(prd);
}
