#import "ViewController.h"

#import <Metal/Metal.h>
#import <MobileCoreServices/MobileCoreServices.h>
#import <Accelerate/Accelerate.h>

#include <vector>
#include <memory>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <half.hpp>

#import "MetalView.h"
#include "MyShaderTypes.hpp"

namespace {
    simd::float4 bridge(const glm::vec4& v) {
        return simd::float4{v.x, v.y, v.z, v.w};
    }
    
    float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
    {
        return (value - inputMin) * ((outputMax - outputMin) / (inputMax - inputMin)) + outputMin;
    }
        
    inline float checker_fast(simd::float2 uv)
    {
        return fmod(floor(uv.x) + floor(uv.y), 2.0f);
    }
    inline size_t align16(size_t size)
    {
        if(size == 0)
        return 0;
        
        return (((size - 1) >> 4) << 4) + 16;
    }
    
    typedef glm::detail::tvec4<half_float::half, glm::highp> hvec4;

    template <typename T>
    using cf_shared_ptr = std::shared_ptr<typename std::remove_pointer<T>::type>;
    
    static CGRect aspect_fit(CGSize source, CGSize destination)
    {
        float srcAspect = source.width / source.height;
        float dstAspect = destination.width / destination.height;
        
        float zoomFactor;
        if(dstAspect < srcAspect)
        {
            //出力先のほうが幅がでかい -> 幅に合わせてリサイズ
            zoomFactor = destination.width / source.width;
        }
        else
        {
            //縦に合わせてリサイズ
            zoomFactor = destination.height / source.height;
        }
        
        CGSize size = { source.width * zoomFactor, source.height * zoomFactor };
        CGPoint point = CGPointMake(-(size.width - destination.width) * 0.5,
                                    -(size.height - destination.height) * 0.5);
        return (CGRect){point, size};
    }
}

@implementation ViewController
{
    IBOutlet UIView *_metalPlaceholderView;
    MetalView *_metalView;
    
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLLibrary> _library;
    id<MTLRenderPipelineState> _renderPipelineState;
    
    // geometry buffers
    id<MTLBuffer> _vertexBuffer;
    id<MTLTexture> _fuild_rgb_texture0;
    id<MTLTexture> _fuild_rgb_texture1;
    
    // compute
    id <MTLComputePipelineState> _step_fuild_kernel;
    
    id<MTLTexture> _fuild_simulation_texture0;
    id<MTLTexture> _fuild_simulation_texture1;

    id<MTLBuffer> _fuildConstantBuffer;
    
    // create rendering loop
    CADisplayLink *_displayLink;
    
    // control
    dispatch_semaphore_t _semaphore;
    CGPoint _previousTouch;
    
    // image picker
    UIPopoverController *_popoverController;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    _metalPlaceholderView.translatesAutoresizingMaskIntoConstraints = NO;
    
    // create metal view dynamic
    _metalView = [[MetalView alloc] init];
    _metalView.translatesAutoresizingMaskIntoConstraints = NO;
    [_metalPlaceholderView addSubview:_metalView];
    
    
    MetalView *metalView = _metalView;
    NSDictionary *views = NSDictionaryOfVariableBindings(metalView);
    NSArray *vConstraints = [NSLayoutConstraint constraintsWithVisualFormat:@"V:|-0-[metalView]-0-|"
                                                                    options:NSLayoutFormatAlignAllCenterX
                                                                    metrics:nil
                                                                      views:views];

    [self.view addConstraints:vConstraints];
    NSArray *hConstraints = [NSLayoutConstraint constraintsWithVisualFormat:@"H:|-0-[metalView]-0-|"
                                                                    options:NSLayoutFormatAlignAllCenterX
                                                                    metrics:nil
                                                                      views:views];
    [self.view addConstraints:hConstraints];
    
    [self.view layoutIfNeeded];
    
    // タッチ
    UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(didPan:)];
    pan.minimumNumberOfTouches = 1;
    pan.maximumNumberOfTouches = 1;
    [_metalView addGestureRecognizer:pan];
    
    // Metal 初期化
    _device = MTLCreateSystemDefaultDevice();
    
    metalView.metalLayer.device = _device;
    metalView.metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    _metalView.metalLayer.framebufferOnly = YES;
    
    _commandQueue = [_device newCommandQueue];
    _library = [_device newDefaultLibrary];

    MyShaderTypes::QuardVertex quad[] = {
        {simd::float2{-1.0f, 1.0f}, simd::float2{0.0f, 1.0f}},
        {simd::float2{ 1.0f, 1.0f}, simd::float2{1.0f, 1.0f}},
        {simd::float2{-1.0f,-1.0f}, simd::float2{0.0f, 0.0f}},
        {simd::float2{ 1.0f,-1.0f}, simd::float2{1.0f, 0.0f}},
    };
    _vertexBuffer = [_device newBufferWithBytes:quad
                                         length:sizeof(quad)
                                        options:0];
    
    auto create_rgba16float_texture = ^{
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                        width:FUILD_SIZE
                                                                                       height:FUILD_SIZE
                                                                                    mipmapped:NO];
        return [_device newTextureWithDescriptor:desc];
    };
    _fuild_rgb_texture0 = create_rgba16float_texture();
    _fuild_rgb_texture1 = create_rgba16float_texture();
    
    [self didSelectedChecker:nil];
    
    _step_fuild_kernel = ^{
        NSError *error;
        return [_device newComputePipelineStateWithFunction:[_library newFunctionWithName:@"step_fuild"]
                                                      error:&error];

    }();
    
    _fuild_simulation_texture0 = create_rgba16float_texture();
    _fuild_simulation_texture1 = create_rgba16float_texture();
    
    // 流体初期化
    std::vector<hvec4> fuild_initial_data(FUILD_SIZE * FUILD_SIZE, hvec4(0.0f, 0.0f, 1.0f, 1.0f));
    MTLRegion region = MTLRegionMake2D(0, 0, _fuild_simulation_texture0.width, _fuild_simulation_texture0.height);
    [_fuild_simulation_texture0 replaceRegion:region
                    mipmapLevel:0
                      withBytes:fuild_initial_data.data()
                    bytesPerRow:sizeof(decltype(fuild_initial_data)::value_type) * FUILD_SIZE];

    MyShaderTypes::FuildConstant fuildConstant = {0};
    _fuildConstantBuffer = [_device newBufferWithBytes:&fuildConstant
                                                length:sizeof(fuildConstant)
                                               options:0];
    
    _renderPipelineState = ^{
        NSError *error;
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = [_library newFunctionWithName:@"myVertexShader"];
        desc.fragmentFunction = [_library newFunctionWithName:@"myFragmentShader"];
        desc.colorAttachments[0].pixelFormat = metalView.metalLayer.pixelFormat;
        desc.sampleCount = 1;
        return [_device newRenderPipelineStateWithDescriptor:desc error:&error];
    }();
    
    _displayLink = [CADisplayLink displayLinkWithTarget:self
                                               selector:@selector(update:)];
    _displayLink.frameInterval = 1;
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];

    
    _semaphore = dispatch_semaphore_create(1);
}
- (void)viewDidLayoutSubviews
{
    [super viewDidLayoutSubviews];
    
    int width = _metalView.bounds.size.width * _metalView.contentScaleFactor;
    int height = _metalView.bounds.size.height * _metalView.contentScaleFactor;
    _metalView.metalLayer.drawableSize = CGSizeMake(width, height);
}
- (void)didPan:(UIPanGestureRecognizer *)sender
{
    MyShaderTypes::FuildConstant *fuildConstant = (MyShaderTypes::FuildConstant *)[_fuildConstantBuffer contents];
    
    CGPoint location = [sender locationInView:_metalView];
    CGPoint velocity = [sender velocityInView:_metalView];
    
    fuildConstant->force = simd::float2{
        (float)velocity.x * 0.003f,
        (float)-velocity.y * 0.003f,
    };
    
    if(sender.state == UIGestureRecognizerStateBegan)
    {
        fuildConstant->a = fuildConstant->b = simd::float2{
            remap(location.x, 0, _metalView.bounds.size.width, 0, FUILD_SIZE),
            remap(location.y, _metalView.bounds.size.height, 0, 0, FUILD_SIZE)
        };
    }
    else if(sender.state == UIGestureRecognizerStateChanged)
    {
        fuildConstant->a = simd::float2{
            remap(_previousTouch.x, 0, _metalView.bounds.size.width, 0, FUILD_SIZE),
            remap(_previousTouch.y, _metalView.bounds.size.height, 0, 0, FUILD_SIZE)
        };
        fuildConstant->b = simd::float2{
            remap(location.x, 0, _metalView.bounds.size.width, 0, FUILD_SIZE),
            remap(location.y, _metalView.bounds.size.height, 0, 0, FUILD_SIZE)
        };
    }
    else if(sender.state == UIGestureRecognizerStateEnded)
    {
        fuildConstant->force = simd::float2{0, 0};
    }
    _previousTouch = location;
}
- (IBAction)didSelectedChecker:(UIButton *)sender
{
    std::vector<hvec4> texture_data(FUILD_SIZE * FUILD_SIZE);
    for(int y = 0 ; y < FUILD_SIZE ; ++y)
    {
        for(int x = 0 ; x < FUILD_SIZE ; ++x)
        {
            float c = checker_fast(simd::float2{float(x), float(y)} * 0.05f);
            
            float h = remap(y, 0, FUILD_SIZE - 1, 0, 270);
            glm::vec3 hsv = glm::vec3(h, 1.0, 1.0);
            glm::vec3 rgb = glm::rgbColor(hsv);
            
            glm::vec3 color = rgb * c;
            texture_data[y * FUILD_SIZE + x] = hvec4(color, 1.0f);
        }
    }
    
    for(id<MTLTexture> texture in @[_fuild_rgb_texture0, _fuild_rgb_texture1])
    {
        [texture replaceRegion:MTLRegionMake2D(0, 0, _fuild_rgb_texture0.width, _fuild_rgb_texture0.height)
                   mipmapLevel:0
                     withBytes:texture_data.data()
                   bytesPerRow:sizeof(decltype(texture_data)::value_type) * FUILD_SIZE];
    }
}
- (IBAction)didSelectedImage:(UIButton *)sender
{
    UIImagePickerController *ipc = [[UIImagePickerController alloc] init];
    ipc.delegate = self;
    ipc.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    ipc.mediaTypes = @[(__bridge id)kUTTypeImage];
    ipc.allowsEditing = YES;
    
    if(UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad)
    {
        _popoverController = [[UIPopoverController alloc] initWithContentViewController:ipc];
        [_popoverController presentPopoverFromRect:sender.frame inView:self.view permittedArrowDirections:UIPopoverArrowDirectionAny animated:YES];
    }
    else
    {
        [self presentViewController:ipc animated:YES completion:^{}];
    }
}
- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary *)info
{
    UIImage *image = [info objectForKey:UIImagePickerControllerEditedImage];
    if(image == nil)
    {
        return;
    }

    CGFloat srcMin = std::min(image.size.width, image.size.height);
    CGRect cropRect = aspect_fit(CGSizeMake(srcMin, srcMin), image.size);
    cf_shared_ptr<CGImageRef> cropImage(CGImageCreateWithImageInRect(image.CGImage, cropRect), CGImageRelease);
    
    CGImageRef srcImage = cropImage.get();
    
    size_t srcWidth = CGImageGetWidth(srcImage);
    size_t srcHeight = CGImageGetHeight(srcImage);
    size_t srcBitsPerComponent = 8;
    size_t srcBytesPerRow = align16(4 * srcWidth);
    std::vector<uint8_t> srcBytes(srcBytesPerRow * srcHeight);
    
    cf_shared_ptr<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceRGB(), CGColorSpaceRelease);
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big;
    cf_shared_ptr<CGContextRef> context(CGBitmapContextCreate(srcBytes.data(), srcWidth, srcHeight, srcBitsPerComponent, srcBytesPerRow, colorSpace.get(), bitmapInfo), CGContextRelease);
    CGContextSetBlendMode(context.get(), kCGBlendModeCopy);
    CGContextSetInterpolationQuality(context.get(), kCGInterpolationNone);
    CGContextDrawImage(context.get(), CGRectMake(0, 0, srcWidth, srcHeight), srcImage);

    vImage_Buffer srcImageBuffer = {
        .data = srcBytes.data(),
        .width = srcWidth,
        .height = srcHeight,
        .rowBytes = srcBytesPerRow,
    };

    size_t dstBytesPerRow = align16(4 * FUILD_SIZE);
    std::vector<uint8_t> dstBytes(dstBytesPerRow * FUILD_SIZE);
    vImage_Buffer dstImageBuffer = {
        .data = dstBytes.data(),
        .width = static_cast<vImagePixelCount>(FUILD_SIZE),
        .height = static_cast<vImagePixelCount>(FUILD_SIZE),
        .rowBytes = dstBytesPerRow,
    };
    
    vImageScale_ARGB8888(&srcImageBuffer, &dstImageBuffer, NULL, kvImageHighQualityResampling);
    
    std::vector<hvec4> texture_data(FUILD_SIZE * FUILD_SIZE);
    
    float div255 = 1.0f / 255.0f;
    for(int y = 0 ; y < FUILD_SIZE ; ++y)
    {
        uint8_t *lineHead = dstBytes.data() + dstBytesPerRow * ((int)FUILD_SIZE - y - 1);
        for(int x = 0 ; x < FUILD_SIZE ; ++x)
        {
            uint8_t *pixelHead = lineHead + x * 4;
            float r = (float)pixelHead[0] * div255;
            float g = (float)pixelHead[1] * div255;
            float b = (float)pixelHead[2] * div255;
            
            texture_data[y * FUILD_SIZE + x] = hvec4(r, g, b, 1.0f);
        }
    }
    for(id<MTLTexture> texture in @[_fuild_rgb_texture0, _fuild_rgb_texture1])
    {
        [_fuild_rgb_texture0 replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
                     mipmapLevel:0
                       withBytes:texture_data.data()
                     bytesPerRow:sizeof(decltype(texture_data)::value_type) * FUILD_SIZE];
    }
    
    if(UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPhone)
    {
        [self dismissViewControllerAnimated:YES completion:^{}];
    }
}


- (void)update:(id)sender
{
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    id<CAMetalDrawable> drawable = [_metalView.metalLayer nextDrawable];
    if(drawable == nil)
    {
        return;
    }
    
    // 流体シミュレーション
    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    MTLSize threadsPerGroup = {16, 16, 1};
    MTLSize numThreadgroups = {(int)FUILD_SIZE / threadsPerGroup.width, (int)FUILD_SIZE / threadsPerGroup.height, 1};
    
    id <MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:_step_fuild_kernel];
    
    [computeEncoder setTexture:_fuild_simulation_texture0
                       atIndex:0];
    [computeEncoder setTexture:_fuild_simulation_texture1
                       atIndex:1];
    [computeEncoder setBuffer:_fuildConstantBuffer offset:0 atIndex:0];
    
    [computeEncoder setTexture:_fuild_rgb_texture0
                       atIndex:2];
    [computeEncoder setTexture:_fuild_rgb_texture1
                       atIndex:3];
    
    [computeEncoder dispatchThreadgroups:numThreadgroups
                   threadsPerThreadgroup:threadsPerGroup];
    
    [computeEncoder endEncoding];
    
    std::swap(_fuild_simulation_texture0, _fuild_simulation_texture1);
    
    // 描画
    id <MTLRenderCommandEncoder> commandEncoder = ^{
        MTLRenderPassDescriptor *desc = [MTLRenderPassDescriptor renderPassDescriptor];
        
        desc.colorAttachments[0].texture = [drawable texture];
        desc.colorAttachments[0].storeAction = MTLStoreActionStore;
        
        return [commandBuffer renderCommandEncoderWithDescriptor:desc];
    }();
    

    [commandEncoder setRenderPipelineState:_renderPipelineState];
    
    [commandEncoder setTriangleFillMode:MTLTriangleFillModeFill];
    [commandEncoder setCullMode:MTLCullModeBack];
    [commandEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
    [commandEncoder setFragmentTexture:_fuild_rgb_texture0 atIndex:0];
    [commandEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [commandEncoder endEncoding];
    
    std::swap(_fuild_rgb_texture0, _fuild_rgb_texture1);
    
    __block dispatch_semaphore_t semaphore = _semaphore;
    [commandBuffer addCompletedHandler:^(id <MTLCommandBuffer> cmdb){
        dispatch_semaphore_signal(semaphore);
    }];
    
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

@end
