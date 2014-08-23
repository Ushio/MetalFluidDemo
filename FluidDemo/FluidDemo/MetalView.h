#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

@interface MetalView : UIView
@property (nonatomic, strong) CAMetalLayer *metalLayer;
@end
