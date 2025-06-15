import cv2
from omni_dc.omni_dc_wrapper import DepthPredictor
import os
import numpy as np
import open3d as o3d
import time

# 1. load network
net = DepthPredictor(checkpoint_path="checkpoints/modelv1.1_best_72epochs.pt", da_path="checkpoints/depth_anything_v2_vitl.pth", device="cuda")


# load .npy file
data = np.load("datasets/uniformat_release/ETH3D_Outdoor_test_2150/000000.npy", allow_pickle=True)
rgb = data.item()['rgb']
dep = data.item()['dep']
K = data.item()["K"]
print(K)
# Iterate over all elements in the 'dep' array and print their indices and values
for idx, value in np.ndenumerate(dep):
    print(f"Index {idx}: {value}")

# 3. predict depth
t = time.time()
depth_map = net(rgb, dep)
print( "processing time in ms: ", (time.time() - t) )
print("dep min:", dep.min(), "dep max:", dep.max())
print("dep shape", depth_map.shape)


# Convert rgb from RGB to BGR for proper cv2 display and normalize dep and depth_map
rgb_disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
dep_disp = cv2.normalize(dep, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_map_disp = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# If depth images are single channel, convert them to BGR
dep_disp = cv2.cvtColor(dep_disp, cv2.COLOR_GRAY2BGR)
depth_map_disp = cv2.cvtColor(depth_map_disp, cv2.COLOR_GRAY2BGR)

# Optional: Resize images for a consistent display size (adjust height as needed)
display_height = 480
rgb_disp = cv2.resize(rgb_disp, (int(rgb_disp.shape[1] * display_height / rgb_disp.shape[0]), display_height))
dep_disp = cv2.resize(dep_disp, (int(dep_disp.shape[1] * display_height / dep_disp.shape[0]), display_height))
depth_map_disp = cv2.resize(depth_map_disp, (int(depth_map_disp.shape[1] * display_height / depth_map_disp.shape[0]), display_height))

# Tile images horizontally
tiled = cv2.hconcat([rgb_disp, dep_disp, depth_map_disp])

# Convert the numpy arrays to open3d images
color_o3d = o3d.geometry.Image(rgb)
depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

# Create an RGBD image (adjust depth_scale if depth values need to be scaled appropriately)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d,
    depth_scale=1.0,
    depth_trunc=1000.0,
    convert_rgb_to_intensity=False
)

# Build the camera intrinsic parameters using the provided K matrix
height, width, _ = rgb.shape
# assume ideal pinhole camera
fx = fy = 525.0  # assumed focal length in pixels
cx = width / 2.0
cy = height / 2.0
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Generate a point cloud from the RGBD image and the intrinsic parameters
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# Transform the point cloud to correct the orientation for visualization
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Save the resulting point cloud as a PLY file
o3d.io.write_point_cloud("output.ply", pcd)