import torch

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images_square


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = torch.nn.functional.interpolate(
        images, size=(resolution, resolution),
        mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

if __name__ == "__main__":
    device = "cpu"
    dtype  = torch.bfloat16

    model = VGGT()
    model.load_state_dict(torch.load("./Model/model.pt", map_location=device))
    model.eval()
    model = model.to(device)

    images, _ = load_and_preprocess_images_square([
        "./IMG_6174.JPG", "./IMG_6175.JPG", "./IMG_6176.JPG"
    ], target_size=1024)

    result = run_VGGT(model, images, dtype, resolution=518)
    print(result)
