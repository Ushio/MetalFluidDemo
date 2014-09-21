#import <simd/simd.h>

//#define FUILD_SIZE 512.0
#define FUILD_SIZE 320.0
//#define FUILD_SIZE 256.0
//#define FUILD_SIZE 128.0

namespace MyShaderTypes {
    struct QuardVertex{
        simd::float2 position;
        simd::float2 texcoord;
    };
    
    
    struct FuildConstant{
        // 外力
        simd::float2 a, b;
        simd::float2 force;
    };
}
