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

constexpr unsigned int RAY_TYPE_COUNT = 1;
constexpr unsigned int MAT_COUNT = 2; //2;
constexpr unsigned int VERTEX_COUNT = 595986;//1360404;
constexpr unsigned int TRIANGLE_COUNT = VERTEX_COUNT/3;
constexpr unsigned int CLUTTER_TRIANGLE_COUNT =  60000 / 3;

constexpr float vP1 = -33.288;
constexpr float vP2 = 32.980;
constexpr float vP3 = 0.510;
constexpr float vP4 = -1.343;
constexpr float vP5 = 4.874;
constexpr float vP6 = -3.142;





struct perRayData
{
    unsigned int seed;
    int          depth;
    float3       farve;

    float3 origin;
    float3 direction;
    int done;
    float afstand;
    
};


struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_lookAt;//til fish eye
    float                  theta_look; //fish
    float                  phi_look;//fish
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
    int                    samples_per_launch;
    int                    time;//til tid, bruger vi ikke

    float3                 detector_center;
    float                  detector_radius;
    CUdeviceptr            rangeVector;//pointer til device buffer for rangeVector
    CUdeviceptr            rangeVectorRealClut;
    CUdeviceptr            rangeVectorImagClut;


    CUdeviceptr            d_arrayZ;//pointer til Clutter array på GPU
    unsigned int           indgange_rangeVector; //hvor mange indgange der er i range vector
    float                  afstand_step_rangeVector; // hvad bin size er i meter i range vector
    unsigned int           clutter_triangle_count;
    unsigned int           scatterProfile;
    float                  loss;

};

struct RangeData {
    int* rangeV;
    float* rangeVRealClut; 
    float* rangeVImagClut;
};

struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    float4* vertices;//til at kunne hente information om vertexes til ramte trekanter, bruges til at regne en surface normal
    int materialIndex;
    float scatter_coef;
};
