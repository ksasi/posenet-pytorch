import cv2
import time
import argparse
import os
import torch
import numpy
import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    dirs_level1 = [f.path for f in os.scandir(args.image_dir) if f.is_dir()]
    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    coord_list = []
    for d in dirs_level1:
        ref_image = [f.path for f in os.scandir(d) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        dirs_level2 = [f.path for f in os.scandir(d) if f.is_dir()]
        for d2 in dirs_level2:
            filenames = [f.path for f in os.scandir(d2) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
            for f in filenames:
                input_image, draw_image, output_scale = posenet.read_imgfile(
                    f, scale_factor=args.scale_factor, output_stride=output_stride)
                input_image_ref, draw_image_ref, output_scale_ref = posenet.read_imgfile(
                    ref_image, scale_factor=args.scale_factor, output_stride=output_stride)

                with torch.no_grad():
                    input_image = torch.Tensor(input_image).cuda()
                    input_image_ref = torch.Tensor(input_image_ref).cuda()

                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
                    heatmaps_result_ref, offsets_result_ref, displacement_fwd_result_ref, displacement_bwd_result_ref = model(input_image_ref)

                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                        heatmaps_result.squeeze(0),
                        offsets_result.squeeze(0),
                        displacement_fwd_result.squeeze(0),
                        displacement_bwd_result.squeeze(0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.25)
                    
                    pose_scores_ref, keypoint_scores_ref, keypoint_coords_ref = posenet.decode_multiple_poses(
                        heatmaps_result_ref.squeeze(0),
                        offsets_result_ref.squeeze(0),
                        displacement_fwd_result_ref.squeeze(0),
                        displacement_bwd_result_ref.squeeze(0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.25)

                keypoint_coords *= output_scale
                keypoint_coords_ref *= output_scale
                coord_list.append(keypoint_coords_ref[0]+keypoint_coords[0])
    print(coord_list)
    numpy.savetxt("coord_list.csv", numpy.array(coord_list).reshape(-1,(17+17)*2), delimiter=",") 

'''
        if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

        if not args.notxt:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
    f.close()
    print('Average FPS:', len(filenames) / (time.time() - start))
'''

if __name__ == "__main__":
    main()
