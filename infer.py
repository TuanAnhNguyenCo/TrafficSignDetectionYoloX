from mmdet.apis import DetInferencer
import sys
import glob
import os

if __name__ == '__main__':
    
    img = sys.argv[1]
    if not os.path.exists(img) :
        print("Please check your image path")
    else:
        # Choose to use a config
        config = 'configs/yolox/yolox_s_8xb8-300e_coco.py'
        # Setup a checkpoint file to load
        checkpoint = glob.glob('./work_dirs/yolox_s_8xb8-300e_coco/best_model.pth')[0]

        # Set the device to be used for evaluation
        device = 'cuda:0'

        # Initialize the DetInferencer
        inferencer = DetInferencer(config, checkpoint, device)

        result = inferencer(img, out_dir='./out')
        img_entered = os.path.basename(img)
        ext = os.path.splitext(img_entered)[1]
        if len(sys.argv) > 2:
            filename = sys.argv[2]
           
            os.rename(f'./out/vis/{img_entered}',f'./out/vis/{filename}{ext}')
        else:
            os.rename(f'./out/vis/{img_entered}',f'./out/vis/output{ext}')
        
            
            
        
