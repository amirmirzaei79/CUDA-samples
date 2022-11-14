#include <stdio.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Device Count: %d\n", deviceCount);
    
    cudaDeviceProp devProp;
    char computeModeName[40];
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaGetDeviceProperties(&devProp, i);


        switch(devProp.computeMode)
        {
            case 0:
                sprintf(computeModeName, "Default compute mode");
                break;
            case 1:
                sprintf(computeModeName, "Compute-exclusive-thread mode");
                break;
            case 2:
                sprintf(computeModeName, "Compute-prohibited mode");
                break;
            case 3:
                sprintf(computeModeName, "Compute-exclusive-process mode");
                break;
            default:
                sprintf(computeModeName, "---");
        }

        printf("\n------------------------------------------------------------------------------\n");

        printf("Device %d:\n", i);
        printf("\n");

        printf("Device name: %s\n", devProp.name);
        // printf("16-byte unique identifier: %d\n", devProp.uuid);
        printf("\n");

        printf("Compute capabilty: %d.%d\n", devProp.major, devProp.minor);
        printf("\n");

        printf("Clock frequency in kilohertz: %d\n", devProp.clockRate);
        printf("\n");

        printf("Number of multiprocessors on device: %d\n", devProp.multiProcessorCount);
        printf("\n");
        
        printf("Warp size in threads: %d\n", devProp.warpSize);
        printf("Maximum number of threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block: %d, %d, %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid: %d, %d, %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("Maximum resident threads per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("\n");

        printf("Global memory available on device in bytes: %lu\n", devProp.totalGlobalMem);
        printf("Shared memory available per block in bytes: %lu\n", devProp.sharedMemPerBlock);
        printf("32-bit registers available per block: %d\n", devProp.regsPerBlock);
        printf("Maximum pitch in bytes allowed by memory copies %lu\n", devProp.memPitch);
        printf("Peak memory clock frequency in kilohertz: %d\n", devProp.memoryClockRate);
        printf("Global memory bus width in bits: %d\n", devProp.memoryBusWidth);
        printf("\n");
        
        printf("Device can concurrently copy memory and execute a kernel: %d\n", devProp.deviceOverlap);
        printf("Number of asynchronous engines: %d\n", devProp.asyncEngineCount);
        printf("\n");

        printf("Device can possibly execute multiple kernels concurrently: %d\n", devProp.concurrentKernels);
        printf("Device supports launching cooperative kernels via cudaLaunchCooperativeKernel: %d\n", devProp.cooperativeLaunch);
        printf("\n");

        printf("Specified whether there is a run time limit on kernels: %d\n", devProp.kernelExecTimeoutEnabled);
        printf("\n");
                
        printf("Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: %d\n", devProp.canMapHostMemory);
        printf("Device shares a unified address space with the host: %d\n", devProp.unifiedAddressing);
        printf("\n");
        
        // printf("Compute mode: %d - %s\n", devProp.computeMode, computeModeName);
        // printf("\n");
        
        // printf("Constant memory available on device in bytes: %lu\n", devProp.totalConstMem);

        // printf("Alignment requirement for textures: %lu\n", devProp.textureAlignment);
        // printf("Pitch alignment requirement for texture references bound to pitched memory: %lu\n", devProp.texturePitchAlignment);
        // printf("Maximum 1D texture size: %d\n", devProp.maxTexture1D);
        // printf("Maximum 1D mipmapped texture size: %d\n", devProp.maxTexture1DMipmap);
        // printf("Maximum 1D linear texture size: %d\n", devProp.maxTexture1DLinear);
        // printf("Maximum 2D texture dimensions: %d, %d\n", devProp.maxTexture2D[0], devProp.maxTexture2D[1]);
        // printf("Maximum 2D mipmapped texture dimensions: %d, %d\n", devProp.maxTexture2DMipmap[0], devProp.maxTexture2DMipmap[1]);
        // printf("Maximum 2D linear texture dimensions: %d, %d\n", devProp.maxTexture2DLinear[0], devProp.maxTexture2DLinear[1]);
        // printf("Maximum 2D texture dimensions (if texture gather operations have to be performed): %d, %d\n", devProp.maxTexture2DGather[0], devProp.maxTexture2DGather[1]);
        // printf("Maximum 3D texture dimensions: %d, %d, %d\n", devProp.maxTexture3D[0], devProp.maxTexture3D[1], devProp.maxTexture3D[2]);
        // printf("Maximum alternate 3D texture dimensions: %d, %d, %d\n", devProp.maxTexture3DAlt[0], devProp.maxTexture3DAlt[1], devProp.maxTexture3DAlt[2]);
        // printf("Maximum Cubemap texture dimensions: %d\n", devProp.maxTextureCubemap);
        // printf("Maximum 1D layered texture dimensions: %d, %d\n", devProp.maxTexture1DLayered[0], devProp.maxTexture1DLayered[1]);
        // printf("Maximum 2D layered texture dimensions: %d, %d, %d\n", devProp.maxTexture2DLayered[0], devProp.maxTexture2DLayered[1], devProp.maxTexture2DLayered[2]);
        // printf("Maximum Cubemap layered texture dimensions: %d, %d\n", devProp.maxTextureCubemapLayered[0], devProp.maxTextureCubemapLayered[1]);
        // printf("Maximum 1D surface size: %d\n", devProp.maxSurface1D);
        // printf("Maximum 2D surface dimensions: %d, %d\n", devProp.maxSurface2D[0], devProp.maxSurface2D[1]);
        // printf("Maximum 3D surface dimensions: %d, %d, %d\n", devProp.maxSurface3D[0], devProp.maxSurface3D[1], devProp.maxSurface3D[2]);
        // printf("Maximum 1D layered surface dimensions: %d, %d\n", devProp.maxSurface1DLayered[0], devProp.maxSurface1DLayered[1]);
        // printf("Maximum 2D layered surface dimensions: %d, %d, %d\n", devProp.maxSurface2DLayered[0], devProp.maxSurface2DLayered[1], devProp.maxSurface2DLayered[2]);
        // printf("Maximum Cubemap surface dimensions: %d\n", devProp.maxSurfaceCubemap);
        // printf("Maximum Cubemap layered surface dimensions: %d, %d\n", devProp.maxSurfaceCubemapLayered[0], devProp.maxSurfaceCubemapLayered[1]);
        // printf("Alignment requirements for surfaces: %lu\n", devProp.surfaceAlignment);
        // printf("\n");

        printf("Size of L2 cache in bytes: %d\n", devProp.l2CacheSize);
        // printf("Device's maximum l2 persisting lines capacity setting in bytes: %d\n", devProp.persistingL2CacheMaxSize);
        printf("\n");

        printf("Device supports caching globals in L1: %d\n", devProp.globalL1CacheSupported);
        printf("Device supports caching locals in L1: %d\n", devProp.localL1CacheSupported);
        printf("\n");
        
        // printf("Device supports allocating managed memory on this system: %d\n", devProp.managedMemory);

        // printf("Ratio of single precision performance (in floating-point operations per second) to double precision performance: %d\n", devProp.singleToDoublePrecisionPerfRatio);

        // printf("Device supports coherently accessing pageable memory without calling cudaHostRegister on it: %d\n", devProp.pageableMemoryAccess);
        // printf("Device can coherently access managed memory concurrently with the CPU: %d\n", devProp.concurrentManagedAccess);
        
        // printf("Device supports Compute Preemption: %d\n", devProp.computePreemptionSupported);

        // printf("Device can access host registered memory at the same virtual address as the CPU: %d\n", devProp.canUseHostPointerForRegisteredMem);

        // printf("Device accesses pageable memory via the host's page tables: %d\n", devProp.pageableMemoryAccessUsesHostPageTables);

        // printf("Host can directly access managed memory on the device without migration: %d\n", devProp.directManagedMemAccessFromHost);

        // printf("The maximum value of cudaAccessPolicyWindow::num_bytes: %d\n", devProp.accessPolicyMaxWindowSize);        

        // printf("Device supports stream priorities: %d\n", devProp.streamPrioritiesSupported);
        // printf("Device has ECC support enabled: %d\n", devProp.ECCEnabled);
        // printf("1 if device is a Tesla device using TCC driver, 0 otherwise: %d\n", devProp.tccDriver);

        // printf("PCI bus ID of device: %d\n", devProp.pciBusID);
        // printf("PCI device ID of device: %d\n", devProp.pciDeviceID);
        // printf("PCI domain ID of device: %d\n", devProp.pciDomainID);

        // printf("Device is integrated as opposed to discrete: %d\n", devProp.integrated);

        // printf("Device is on a multi-GPU board: %d\n", devProp.isMultiGpuBoard);
        // printf("Unique identifier for a group of devices on the same multi-GPU board: %d\n", devProp.multiGpuBoardGroupID);
    }

    return 0;
}
