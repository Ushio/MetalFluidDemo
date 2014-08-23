#import "MetalView.h"
#import <Metal/Metal.h>
@implementation MetalView
{
    id <MTLDevice> _device;
}
+ (id)layerClass
{
    return [CAMetalLayer class];
}
- (instancetype)init
{
    if(self = [super init])
    {
        _metalLayer = (CAMetalLayer *)self.layer;
        self.contentScaleFactor = [UIScreen mainScreen].scale;
    }
    return self;
}
@end
