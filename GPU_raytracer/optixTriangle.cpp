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
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixTriangle.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>// ##################################################### TILFØJET
#include "im.h"// ###############################################################
#include <ctime>//##############################################################
#include<fstream>//##################################################
#include <sutil/vec_math.h>//####################################
#include <math.h>  //####################################################
#include <fstream>//####################################################
#include <iterator>//##################################################
#include <algorithm>//###################

#include <sutil/Camera.h>
#include <sutil/Trackball.h>




template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

struct RayTracerState
{
    OptixDeviceContext context = 0; //context

    OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices = 0;

    OptixModule module = 0; //module
    OptixPipelineCompileOptions pipeline_compile_options = {};

    OptixProgramGroup raygen_prog_group = nullptr; //program groups
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixPipeline pipeline = nullptr; //pipeline

    OptixShaderBindingTable sbt = {}; //shader binding table (SBT)

    int         width = 768;
    int         height = 384;

};

//###########################################################TILFØJET
std::string fileName = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\T_62_MSTAR595986.txt"; //filnavn for mesh(vertex liste)
std::string fileName_triangleIndex = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\T_62_MSTAR595986trekantlist60000.txt"; //fil navn for trekant liste med materiale index


std::array<float4, VERTEX_COUNT>* pvertices = import_mesh(fileName);
std::array<uint32_t, TRIANGLE_COUNT>* p_mat_indices = import_mesh_mat_indices(fileName_triangleIndex);
//################################################################


void triangle_clutter_reflectivity_func(float PI, float slant_angle, Params& params){ //regner reflektans til clutter baggrund
    std::cout << "     ___" << std::endl;
    std::cout << "  __(   )==== " << std::endl;
    std::cout << "/ ~~~~~~~~~| " << std::endl;
    std::cout << "| O.O.O.O.O/ " << std::endl;
    
    float* arrayZ;
    arrayZ = new float[CLUTTER_TRIANGLE_COUNT*2];
    float slant2 = -15.0f;
    float sigma_0 = vP1 + vP2 * exp(-vP3 * slant2 * PI / 180) + vP4 * cos(vP5 * slant2 * PI / 180 + vP6);
    std::cout << "- - - - - - - - - - - - - - - - Clutter parameters - - - - - - - - - - - - - - - - - -" << std::endl;
    std::cout << "sigma_0=" << sigma_0 << std::endl;
    float Areal = 0.2;//ca areal af facets
    float skalering = 0.8*6; //0.8*3-tanktests;//2.5;//0.4;//0.2 fin ved 0.2 scatter
    float varians = sqrt(sigma_0*Areal/(2*cos(slant_angle*PI/180)))*skalering;
    std::srand(std::time(0)); //sørger for random baggrund for hvert billede... #####################################################!!!!!!!!!!!!!!!!!!########################################
    std::cout << "varians=" << varians<<std::endl;
    float maxrandtal= RAND_MAX;
    for (int j = 0; j < CLUTTER_TRIANGLE_COUNT; j++) {
        
        float U = rand()/ maxrandtal;
        float V = rand() / maxrandtal;
        
        
        (arrayZ)[j*2] = sqrt(-2 * log(U)) * cos(2 * PI * V) * varians;
        (arrayZ)[j * 2+1] = sqrt(-2 * log(U)) * sin(2 * PI * V) * varians;

       // std::cout << arrayZ[2*j] << " "<< arrayZ[2*j+1];
    } 

    CUdeviceptr d_arrayZ1 = 0; // laver en GPU pointer
    const size_t arrayZ_size = sizeof(float) * CLUTTER_TRIANGLE_COUNT*2; //regner plads til vector
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_arrayZ1), arrayZ_size)); //allokere plads på GPU ram
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_arrayZ1),arrayZ, arrayZ_size, cudaMemcpyHostToDevice));

    params.d_arrayZ = d_arrayZ1;
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - -" << std::endl;
   
    delete[] arrayZ; 
}

const std::array<float, MAT_COUNT> scatter_coef =
{
  0.85f, 0.05f //0.4båd

};

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    //cam.setEye( {10.0f, 10.0f, 0.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, -1.0f, 0.0f} );
    cam.setFovY( 0.2f ); //2.0//4//17båd//10sidst
    cam.setAspectRatio( (float)width / (float)height );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}
/////createContext burde være fin her#########################################
void createContext(RayTracerState& state) {
    //
        // Initialize CUDA and create OptiX context
        //
    OptixDeviceContext context;
    {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
#ifdef DEBUG
        // This may incur significant performance cost and should only be done during development.
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        state.context = context;
    }
} 
//buildMeshAccel skal måske revideres i fremtiden#################################
void buildMeshAccel(RayTracerState& state) {
    //
        // accel handling
        //
    const size_t vertices_size = sizeof(float4) * VERTEX_COUNT;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_vertices),
        pvertices,
        vertices_size,
        cudaMemcpyHostToDevice
    ));
    //trekants index til GPU
    CUdeviceptr  d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = (*p_mat_indices).size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        p_mat_indices,
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice));

    // til materialer
    uint32_t triangle_input_flags[MAT_COUNT];
    for (int j = 0; j < MAT_COUNT; j++) {
        triangle_input_flags[j] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }
        
    
    
    //###
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float4);//måske problem her, aner ikke hvad den gør
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(TRIANGLE_COUNT*3);
    triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = MAT_COUNT;//##
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);//##
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);//##


    
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    //accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &triangle_input,
        1, // Number of build inputs
        &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));


    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),compactedSizeOffset + 8));


    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,                  // CUDA stream
        &accel_options,
        &triangle_input,
        1,                  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty,            // emitted property list tilføj &emitProperty
        1                   // num emitted properties
    ));

    
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));
    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_vertices))); vi skal bruge d_vertices til HitGroupData

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));
        

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size; 
    }
        
}

void createModule(RayTracerState& state) {
    //
        // Create module
        //
    
    
    {
        OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

        state.pipeline_compile_options.usesMotionBlur = false;
        state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        state.pipeline_compile_options.numPayloadValues = 13;
        state.pipeline_compile_options.numAttributeValues = 3;
        state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        size_t      inputSize = 0;
        const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixTriangle.cu", inputSize);

        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.module
        ));
    }
}

void createProgramGroups(RayTracerState& state) {
    //
        // Create program groups
        //
    
    {
        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {}; //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &raygen_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.raygen_prog_group
        ));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &miss_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.miss_prog_group
        ));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = state.module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hitgroup_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.hitgroup_prog_group
        ));
    }
}

void createPipeline(RayTracerState& state) {
    //
        // Link pipeline
        //
    
    {
        const uint32_t    max_trace_depth = 2;
        OptixProgramGroup program_groups[] = { state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        OPTIX_CHECK_LOG(optixPipelineCreate(
            state.context,
            &state.pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            LOG, &LOG_SIZE,
            &state.pipeline
        ));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, state.pipeline));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
            0,  // maxCCDepth
            0,  // maxDCDEpth
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state, continuation_stack_size,
            1  // maxTraversableDepth ################################################################################var 1 før
        ));
    }
}

void createSBT(RayTracerState& state) {
    //
        // Set up shader binding table
        //
    
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size * RAY_TYPE_COUNT));

        MissSbtRecord ms_sbt;
        ms_sbt.data = { 0.3f, 0.1f, 0.2f };
        OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size *RAY_TYPE_COUNT*MAT_COUNT));


        //######################################## lidt skecthy, taget fra pathtracer, sørger for vi kan hente vertex data fra hitgroup record
        HitGroupSbtRecord hg_sbt[RAY_TYPE_COUNT * MAT_COUNT];
        for (int i = 0; i < MAT_COUNT; ++i)
        {
            {
                const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

                
                OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt[sbt_idx]));
                hg_sbt[sbt_idx].data.vertices = reinterpret_cast<float4*>(state.d_vertices);
                hg_sbt[sbt_idx].data.materialIndex = i; // <-------- HOVSA DET HER SKAL FIKSES, var "i" førhen......####################################!!!!!!!!!!!!!!!!!!!##############################
                hg_sbt[sbt_idx].data.scatter_coef = scatter_coef[i];
            }

            // Note that we do not need to use any program groups for occlusion
            // rays as they are traced as 'probe rays' with no shading.
        }
        //###################################################################################################################



        


        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size*RAY_TYPE_COUNT*MAT_COUNT,
            cudaMemcpyHostToDevice
        ));

        state.sbt.raygenRecord = raygen_record;
        state.sbt.missRecordBase = miss_record;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size); 
        state.sbt.missRecordCount = RAY_TYPE_COUNT;
        state.sbt.hitgroupRecordBase = hitgroup_record;
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
        state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT*MAT_COUNT;
    }
}

void launch(sutil::CUDAOutputBuffer<uchar4>& output_buffer, RayTracerState& state, Params& params, float3 cam_pos, RangeData& rangeData) {//int*

    
    //
        // launch
        //
    {
        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        
        float3 cam_posnorm = normalize(cam_pos);
        sutil::Camera cam;
        cam.setEye(cam_pos);//flyttet fra configure camera
        configureCamera(cam, state.width, state.height);
        params.scatterProfile = 2; //1:alpha 2: gamma 3:Lambda
        
        params.time = std::time(nullptr);
        params.samples_per_launch = 512/4;
        params.image = output_buffer.map();
        params.image_width = state.width;
        params.image_height = state.height;
        params.handle = state.gas_handle;
        params.cam_eye = cam_pos;//cam.eye();
        params.cam_lookAt = { 0.0f, 0.0f, 0.0f };
        float3 look_retning = normalize(params.cam_lookAt - params.cam_eye);
        params.phi_look = std::copysignf(1.0, look_retning.y) * acosf(look_retning.x / std::sqrtf(look_retning.x * look_retning.x + look_retning.y * look_retning.y));
        params.theta_look = acosf(look_retning.z);
        cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);
        
        params.loss = 1;
        
        params.detector_radius = 40*10;
        params.detector_center = cam_pos+cam_posnorm*1.1*params.detector_radius;
        params.afstand_step_rangeVector = 0.016/8/2;//range bin length i meter.
        params.indgange_rangeVector = 50000*2;
        params.clutter_triangle_count = CLUTTER_TRIANGLE_COUNT;

        //std::cout <<"eye " << params.cam_eye.x << " " << params.cam_eye.y << " " << params.cam_eye.z << std::endl;
        //std::cout << "detector " << params.detector_center.x << " " << params.detector_center.y << " " << params.detector_center.z << std::endl;

        //########################### 
        int* rangeV; //pointer til int
        rangeV = new int[params.indgange_rangeVector](); //allokere plads på host ram til int*indgange
        float* rangeVRealClut;
        float* rangeVImagClut;//pointer til int
        rangeVRealClut = new float[params.indgange_rangeVector];
        rangeVImagClut = new float[params.indgange_rangeVector];
                
        CUdeviceptr d_rangeVector = 0; // laver en GPU pointer
        const size_t rangeVector_size = sizeof(int)*params.indgange_rangeVector; //regner plads til vector
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rangeVector), rangeVector_size)); //allokere plads på GPU ram
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_rangeVector), 0, rangeVector_size)); 

        CUdeviceptr d_rangeVectorRealClut = 0; // laver en GPU pointer
        const size_t rangeVectorRealClut_size = sizeof(float) * params.indgange_rangeVector; //regner plads til vector
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rangeVectorRealClut), rangeVectorRealClut_size)); //allokere plads på GPU ram
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_rangeVectorRealClut), 0, rangeVectorRealClut_size));

        CUdeviceptr d_rangeVectorImagClut = 0; // laver en GPU pointer
        const size_t rangeVectorImagClut_size = sizeof(float) * params.indgange_rangeVector; //regner plads til vector
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rangeVectorImagClut), rangeVectorImagClut_size)); //allokere plads på GPU ram
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_rangeVectorImagClut), 0, rangeVectorImagClut_size));
        
        params.rangeVector = d_rangeVector; // gemmer pointer i params struct (meget vigtigt for optix)
        params.rangeVectorRealClut = d_rangeVectorRealClut;
        params.rangeVectorImagClut = d_rangeVectorImagClut;

       
        //#############################
        
               
        //smidder params struct over på GPU
        CUdeviceptr d_param;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_param),
            &params, sizeof(params),
            cudaMemcpyHostToDevice
        ));
        
        OPTIX_CHECK(optixLaunch(state.pipeline, stream, d_param, sizeof(Params), &state.sbt, state.width, state.height, /*depth=*/1));
        CUDA_SYNC_CHECK();

        //##########################################################
        //optix launch har lige kørt vores ray tracer, rangevector skal flyttes tilbage fra GPU til host        
        CUDA_CHECK(cudaMemcpy(rangeV, reinterpret_cast<void*>(params.rangeVector), rangeVector_size,
            cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(rangeVRealClut, reinterpret_cast<void*>(params.rangeVectorRealClut), rangeVectorRealClut_size,
            cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(rangeVImagClut, reinterpret_cast<void*>(params.rangeVectorImagClut), rangeVectorImagClut_size,
            cudaMemcpyDeviceToHost
        ));
        
        
        //##########################################################

        output_buffer.unmap();
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.rangeVector))); //rydder lidt op så mit stakkels grafikkort ikke kradser af
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.rangeVectorRealClut)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.rangeVectorImagClut)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));



        //##########gemmer range pointers
        
        rangeData.rangeV = new int[params.indgange_rangeVector]();
        rangeData.rangeV = rangeV;

        rangeData.rangeVRealClut= new float[params.indgange_rangeVector];
        rangeData.rangeVRealClut = rangeVRealClut;

        rangeData.rangeVImagClut = new float[params.indgange_rangeVector];
        rangeData.rangeVImagClut = rangeVImagClut;
        
        //return rangeV;
    } 
    
}

void cleanup(RayTracerState& state) {
    //
        // Cleanup
        //
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));

        OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
        OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
        OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
        OPTIX_CHECK(optixModuleDestroy(state.module));

        OPTIX_CHECK(optixDeviceContextDestroy(state.context));
    }
}



int main( int argc, char* argv[] )
{

    






    RayTracerState state;
    std::string outfile;
    
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), state.width, state.height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }
    
    try
    {
        
        createContext(state);
        buildMeshAccel(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        
        createSBT(state);
        
        //Noget der render billeder?! Må man jo lige finde ud af en anden dag... Ny dag: den henter billede data fra GPU´en.
        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, state.width, state.height);
        
        //############################################ vinkel loop#######################################################################
        //const std::array<float, 5> center_vinkel =
        //{
        //  -92.0f, -137.0f, -185.5f, -226.5f, -270.0f

        //};
        
        std::string file_named = "T62_17_deg.txt";
        

        for (int v = 0; v < 150; v++) {
            //##############PARAMETER HJØRNET#########################################
            unsigned int målinger = 153;
            float vinkel_interval = 2.94;//i grader
            float range_til_center = 250 * 2+4500; //i meter
            float slant_angle = -17;//i grader 
            float PI = 3.14159265;

            float maxrandtal = RAND_MAX;
            float azVink = static_cast<float>(v) * 2.4+rand()/maxrandtal*2.39;
            //#######################################################################
            Params params;
            triangle_clutter_reflectivity_func(PI, slant_angle, params); //laver pointer til clutter array

            //float* rangeSlut; //pointer til int
            //rangeSlut = new float[params.indgange_rangeVector*målinger]();
            //lav en txt fil
            std::string azangle_file_named = std::to_string(azVink) + file_named;
            std::ofstream MyFile("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\T62_17_deg\\" + azangle_file_named);
            MyFile.close();
            for (int l = 0; l < målinger; l++) {
                RangeData rangeData;
                float theta = l * vinkel_interval / målinger * PI / 180 - vinkel_interval * PI / 180 / 2 - (azVink) * PI / 180; //PI*0.2477;//*0.7139; //+110 * PI / 180; //-122*PI/180; //PI / 2;
                float x = cos(theta) * cos(-slant_angle * PI / 180) * range_til_center;
                float y = sin(theta) * cos(-slant_angle * PI / 180) * range_til_center;
                float z = sin(slant_angle * PI / 180) * range_til_center;
                float3 cam_pos = { x,z,y };
                launch(output_buffer, state, params, cam_pos, rangeData);//sætter hele programmet igang.int* rangeV = 
                //########gem rangeV
                //std::cout << "set " << x << " " << z << " " << y << std::endl;

                //for (int j = 0; j < 640000; j++) {
                 //   *(rangeSlut+j+l*640000)=*(rangeData.rangeVRealClut+j); // << " " << rangeData.rangeVImagClut[j] 
                //}
                std::ofstream ofile;
                
                ofile.open("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\T62_17_deg\\"+azangle_file_named, std::ios::app);

                for (int j = 0; j < params.indgange_rangeVector; j++) {
                    ofile << rangeData.rangeVRealClut[j] << '\n'; //" " << rangeData.rangeVImagClut[j] << 
                }

                delete[] rangeData.rangeV;
                delete[] rangeData.rangeVRealClut;
                delete[] rangeData.rangeVImagClut;
                //gemt
                std::cout << "vinkel nr:" << l << std::endl;
            }

        }
        //lav en txt fil
        //std::ofstream MyFile("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\datamappen\\t72_15deg_aznr.txt");
        //MyFile.close();
        //std::ofstream ofile;
        //ofile.open("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\datamappen\\t72_15deg_aznr.txt", std::ios::app);
        //std::cout << "Skriver"<< std::endl;

        //std::cout << rangeSlut[0] << std::endl;
      
        //for (int j = 0; j < params.indgange_rangeVector*målinger; j++) {
          //  ofile << rangeSlut[j] << std::endl; // << " " << rangeData.rangeVImagClut[j] 
        //}
        //ofile.close();
        std::cout << "Gemt" << std::endl;
        //delete[] rangeSlut;












        
        
        //
        // Display results
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = state.width;
            buffer.height       = state.height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                sutil::displayBufferWindow( argv[0], buffer );
            else
                sutil::saveImage( outfile.c_str(), buffer, false );
        }

        cleanup(state);
    }
    
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
