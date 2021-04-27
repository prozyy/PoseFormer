import numpy as np
from common.arguments import parse_args
import torch
import os
import sys
from common.camera import *
from common.model_poseformer import *
from common.utils import *
from common.skeleton import Skeleton
from common.visualization import render_animation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parse_args()

data_skeleton = Skeleton(parents=[-1,  0,  0,  1,  2,  6,  5,  5,  6,  7,  8,  5, 6, 11, 12, 13, 14, 15, 16, 15, 16, 0, 9, 10],
       joints_left=[1,3,5,7,9,11,13,15,17,19,22],
       joints_right=[2,4,6,8,10,12,14,16,18,20,23])

test_info = {
    "fps":30,
    "res_w":1000,
    "res_h":1000,
    "receptive_field":27,
    "num_joints":24,
    "azimuth":70,
    "rot":np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],dtype='float32')
}

keypoints_metadata = {'layout_name': 'coco', 'num_joints': 24, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]]}
joints_left, joints_right = list(data_skeleton.joints_left()), list(data_skeleton.joints_right())

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()

## normalize_screen_coordinates
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=test_info['res_w'], h=test_info['res_h'])
            keypoints[subject][action][cam_idx] = kps

#########################################PoseTransformer
model_pos = PoseTransformer(num_frame=test_info["receptive_field"], num_joints=test_info["num_joints"], in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
#################

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

def eval_data_prepare(inputs_2d,receptive_field = test_info["receptive_field"]):
    inputs_2d_p = torch.squeeze(inputs_2d)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d

# Evaluate
def evaluate(batch_2d):
    with torch.no_grad():
        model_pos.eval()
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

        ##### apply test-time-augmentation (following Videopose3d)
        inputs_2d_flip = inputs_2d.clone()
        inputs_2d_flip [:, :, :, 0] *= -1
        inputs_2d_flip[:, :, joints_left + joints_right,:] = inputs_2d_flip[:, :, joints_right + joints_left,:]

        ##### convert size
        inputs_2d = eval_data_prepare(inputs_2d)
        inputs_2d_flip = eval_data_prepare(inputs_2d_flip)

        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
            inputs_2d_flip = inputs_2d_flip.cuda()

        predicted_3d_pos = model_pos(inputs_2d)
        predicted_3d_pos_flip = model_pos(inputs_2d_flip)

        predicted_3d_pos_flip[:, :, :, 0] *= -1
        predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,joints_right + joints_left]

        predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,keepdim=True)
        return predicted_3d_pos.squeeze(0).cpu().numpy()

if args.render:
    print('Rendering...')
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    pad = (test_info["receptive_field"] -1) // 2 # Padding on each side
    batch_2d = np.expand_dims(np.pad(input_keypoints,((pad, pad), (0, 0), (0, 0)),'edge'), axis=0)
    prediction = evaluate(batch_2d)
    if args.viz_output is not None:
        # Invert camera transformation
        prediction = camera_to_world(prediction, R=test_info["rot"], t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :,:, 2] -= np.min(prediction[:, :,:, 2])
        anim_output = {'Reconstruction': prediction}
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=test_info['res_w'], h=test_info['res_h'])
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         data_skeleton, test_info['fps'], args.viz_bitrate, test_info['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(test_info['res_w'], test_info['res_h']),
                         input_video_skip=args.viz_skip)