
import sys
import torch
import torchplus
from google.protobuf import text_format
from second.protos import pipeline_pb2
from second.pytorch.train import build_network


def trans_onnx(config_path,ckpt_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    
    net = build_network(model_cfg, measure_time=False).to(device)
    net.load_state_dict(torch.load(ckpt_path))
    
    voxels = torch.ones([12000, 100, 4],dtype=torch.float32, device=device)
    num_points = torch.ones([12000],dtype=torch.float32, device=device)
    coors = torch.ones([12000, 4],dtype=torch.float32, device=device)

    example1 = (voxels, num_points,coors)

    spatial_features = torch.ones([1, 64, 496, 432],dtype=torch.float32, device=device)

    example2 = (spatial_features,)
    
    torch.onnx.export(net.voxel_feature_extractor,example1, "pfe.onnx", verbose=False)
    torch.onnx.export(net.rpn,example2, "rpn.onnx", verbose=False)

if __name__ == '__main__':

    config_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    trans_onnx(config_path,ckpt_path)


