import os
import fire
import numpy as np
import torch
import onnx
import onnxruntime
import matplotlib.pyplot as plt
from pypcd import pypcd
from google.protobuf import text_format

from second.protos import pipeline_pb2
from second.pytorch.train import build_network
from second.utils import config_tool
from second.utils import simplevis
from second.core import box_np_ops


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def read_pointcloud(data_path):

    #is velodyne_reduced ,not velodyne
    if os.path.splitext(data_path)[-1] == '.bin':
        points = np.fromfile(data_path, dtype=np.float32, count=-1).reshape([-1, 4])
    
    elif os.path.splitext(data_path)[-1] == '.pcd':
        cloud = pypcd.PointCloud.from_path(data_path)
        x = cloud.pc_data['x'].reshape(-1,1)
        y = cloud.pc_data['y'].reshape(-1,1)
        z = cloud.pc_data['z'].reshape(-1,1)
        intensity = cloud.pc_data['intensity']/255.
        intensity = intensity.reshape(-1,1)
        points = np.concatenate((x,y,z,intensity),-1)
    
    else:
        print('not suport file format')
        return None
    
    return points

def generate_example(net,model_cfg,points,device):

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors_np = target_assigner.generate_anchors(feature_map_size)["anchors"]

    anchors = torch.tensor(anchors_np, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors_np[:, [0, 1, 3, 4, 6]])

    res = voxel_generator.generate(points, max_voxels=12000)
    voxels = res["voxels"]
    coordinates = res["coordinates"]
    num_points = res["num_points_per_voxel"]

    # add batch idx to coords
    coords = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.float32, device=device)

    # generate anchor mask
    # slow with high resolution. recommend disable this forever.
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    coors = coordinates
    dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
        coors, tuple(grid_size[::-1][1:]))
    dense_voxel_map = dense_voxel_map.cumsum(0)
    dense_voxel_map = dense_voxel_map.cumsum(1)
    anchors_area = box_np_ops.fused_get_anchors_area(
        dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
    anchors_mask = anchors_area > 1
    anchors_mask = torch.tensor(anchors_mask, dtype=torch.uint8, device=device)
    # example['anchors_mask'] = anchors_mask.astype(np.uint8)

    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
        'anchors_mask': anchors_mask,
    }

    return example


def onnx_inference(config_path,data_path,pfe_path,rpn_path):

    config = pipeline_pb2.TrainEvalPipelineConfig()

    with open(config_path, "r") as f:
	    proto_str = f.read()
	    text_format.Merge(proto_str, config)
    
    model_cfg = config.model.second
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = build_network(model_cfg).to(device).eval()
    points = read_pointcloud(data_path)
    example = generate_example(net,model_cfg,points,device)

    #onnx inference
    ort_session_pfe = onnxruntime.InferenceSession(pfe_path)
    ort_session_rpn = onnxruntime.InferenceSession(rpn_path)

    # compute ONNX Runtime output prediction
    ort_inputs_pfe = {ort_session_pfe.get_inputs()[0].name: to_numpy(example["voxels"]),
                  ort_session_pfe.get_inputs()[1].name: to_numpy(example["num_points"]),
                  ort_session_pfe.get_inputs()[2].name: to_numpy(example["coordinates"])}
    
    ort_outs_pfe = ort_session_pfe.run(None, ort_inputs_pfe)

    voxel_features = torch.from_numpy(ort_outs_pfe[0]).to(device)

    spatial_features = net.middle_feature_extractor(voxel_features, example["coordinates"], 1)

    ort_inputs_rpn = {ort_session_rpn.get_inputs()[0].name: to_numpy(spatial_features)}
    
    ort_outs_rpn = ort_session_rpn.run(None, ort_inputs_rpn)

    preds_dict = {}
    preds_dict["box_preds"] = torch.from_numpy(ort_outs_rpn[0]).to(device)
    preds_dict["cls_preds"] = torch.from_numpy(ort_outs_rpn[1]).to(device)
    preds_dict["dir_cls_preds"] = torch.from_numpy(ort_outs_rpn[2]).to(device)

    with torch.no_grad():
        pred = net.predict(example, preds_dict)[0]


    boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()

    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
    
    plt.imsave("result_onnx.png",bev_map)


def pytorch_inference(config_path=None,
              ckpt_path=None,
              data_path=None):

    config = pipeline_pb2.TrainEvalPipelineConfig()

    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = build_network(model_cfg).to(device).eval()
    net.load_state_dict(torch.load(ckpt_path))

    points = read_pointcloud(data_path)
    example = generate_example(net,model_cfg,points,device)

    pred = net(example)[0]
    
    boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()

    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
    
    plt.imsave("result.png",bev_map)

if __name__ == '__main__':
    fire.Fire()
