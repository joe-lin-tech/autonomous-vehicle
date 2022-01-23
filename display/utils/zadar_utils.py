import numpy as np
import open3d as o3d
import csv
import plotly.graph_objects as go
import os

# from nuScenes


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def box_to_corners(box):
    """
    Convert box corners to 3D points.
    :param boxes: <np.float32: 9,>. x, y, z, d, w, h, x_rot, y_rot, z_rot.
    :return: <np.float32: 3, 8>. 3D box corners.
    """
    x, y, z, d, w, h, x_rot, y_rot, z_rot = box

    # Get rotation matrix.
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
        np.array([x_rot, y_rot, z_rot]))

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = d / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(rotation_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def get_pointcloud_scatter(frame_id):
    pc_path = os.path.join(
        os.getcwd(),
        "dataset/ZadarLabsDataset/zadar_zsignal_dataset_1car_1bicycle_1human/segment-1/" + frame_id + "/zsignal0_zvue.csv")
    pc_points = []
    # retrieve pointcloud points from csv
    with open(pc_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for row in reader:
            pc_points.append(list(map(float, row)))
    pc_points = np.array(pc_points)

    pointcloud = go.Figure(
        data=go.Scatter3d(
            x=pc_points[:, 0],
            y=pc_points[:, 1],
            z=pc_points[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                #     color=colors,
                colorscale="rainbow"
            )
        ),
        layout=go.Layout(
            title=go.layout.Title(text=frame_id + " Zadar Pointcloud"),
            height=500,
            # scene=dict(
            #     xaxis = dict(nticks=10, range=[-100, 100]),
            #     yaxis = dict(nticks=10, range=[-100, 100]),
            #     zaxis = dict(nticks=10, range=[-100, 100])
            # )
            scene_aspectmode="data"
        )
    )
    return pointcloud


def draw_box(scatter, box_corners):
    box_corners = np.transpose(box_corners)
    scatter.add_mesh3d(
        x=box_corners[:, 0],
        y=box_corners[:, 1],
        z=box_corners[:, 2],
        i=[1, 6, 6, 6, 5, 5, 0, 0, 5, 6, 2, 3],
        j=[2, 5, 7, 3, 4, 0, 4, 3, 6, 7, 0, 2],
        k=[6, 1, 3, 2, 0, 1, 7, 7, 4, 4, 1, 0],
        opacity=0.1,
        color="blue"
    )

def draw_detection(scatter, detection_corners, score):
    detection_corners = np.transpose(detection_corners)
    print(score)
    scatter.add_mesh3d(
        x=detection_corners[:, 0],
        y=detection_corners[:, 1],
        z=detection_corners[:, 2],
        i=[1, 6, 6, 6, 5, 5, 0, 0, 5, 6, 2, 3],
        j=[2, 5, 7, 3, 4, 0, 4, 3, 6, 7, 0, 2],
        k=[6, 1, 3, 2, 0, 1, 7, 7, 4, 4, 1, 0],
        opacity=0.1,
        # opacity=score[0],
        color="red",
        # name=str(score[0])
    )


def get_layout_plots(targets, detections, losses):
    all_pc_scatters = []
    boxes, scores = detections
    for target, detection_boxes, detection_scores, loss in zip(targets, boxes, scores, losses):
        target_boxes = target["boxes"]
        labels = target["labels"]
        frame_id = str(target["frame_id"][0])
        volume = target["volume"]
        pc_scatter = get_pointcloud_scatter(frame_id)
        for box in target_boxes:
            box_corners = box_to_corners(box)
            draw_box(pc_scatter, box_corners)
        for box, score in zip(detection_boxes, detection_scores):
            detection_corners = box_to_corners(box)
            draw_detection(pc_scatter, detection_corners, score)
        all_pc_scatters.append(pc_scatter)
    return all_pc_scatters




# visualization
import json
import platform
from pathlib import Path

import fire
import cv2
from tqdm import tqdm
import pandas
import imageio
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

if platform.system() == 'Darwin':
    import matplotlib

    matplotlib.use('agg')


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def points_to_img(points, img, intrinsic_matrix, fig, ax, min_dist=1.0, dot_size=20, coloring=None, boxes=None,
                  transformation_matrix=None):
    if transformation_matrix is not None:
        points = transform_points(points, transformation_matrix)
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = points[2, :]
    if coloring is None:
        coloring = depths
    points = view_points(points, intrinsic_matrix, normalize=True)
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    plt.cla()
    ax.imshow(img[:, :, ::-1])
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')

    if boxes is not None:
        for box in boxes:
            box.render(ax, intrinsic_matrix, image_size=(img.shape[1], img.shape[0]),
                       transformation_matrix=transformation_matrix)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def transform_points(points, T):
    nbr_points = points.shape[1]
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = T @ points
    points = points[:3, :]
    return points


def draw_boxes_in_points(viewer, boxes, bbox_color=(0, 1, 0), transformation_matrix=None):
    for i in range(len(boxes)):
        box3d = boxes[i].box3d
        if transformation_matrix is not None:
            q = Quaternion(matrix=transformation_matrix)
            box3d = box3d.rotate(q.rotation_matrix)
            box3d = box3d.translate(transformation_matrix[:3, 3])
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        viewer.add_geometry(line_set)


def draw_boxes(img, bboxes):
    bboxes = np.array(bboxes).reshape(-1, 4)
    bboxes[:, 2:4] += bboxes[:, 0:2]
    bboxes = bboxes.astype('int')
    for i in range(len(bboxes)):
        box = bboxes[i]
        p1 = (box[0], box[1])
        p2 = (box[2], box[3])
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)


class RadarBox(object):
    def __init__(self, bbox3d):
        center = bbox3d[0:3]
        dim = bbox3d[3:6]
        xyz = bbox3d[6:9]
        xyz = xyz / 180 * np.pi
        roll, pitch, yaw = xyz
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.array([roll, pitch, yaw]))
        self.box3d = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, dim)

    @classmethod
    def corners(cls, box3d):
        l, w, h = box3d.extent

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(box3d.R, corners)

        # Translate
        x, y, z = box3d.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def render(self, axis, view, image_size=None, normalize=True, colors=('b', 'r', 'k'), linewidth=2,
               transformation_matrix=None):
        box3d = self.box3d
        corners = self.corners(box3d)
        if transformation_matrix is not None:
            corners = transform_points(corners, transformation_matrix)
        corners = view_points(corners, view, normalize=normalize)[:2, :]

        if image_size is not None:
            corners[0, :] = np.clip(corners[0, :], 0, image_size[0] - 1)
            corners[1, :] = np.clip(corners[1, :], 0, image_size[1] - 1)

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)


def vis(seg_folder,
        save_dir, undistort: bool = True):
    seg_folder = Path(seg_folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if undistort:
        save_name = f"{seg_folder.parent.name}_{seg_folder.name}_undistort.gif"
    else:
        save_name = f"{seg_folder.parent.name}_{seg_folder.name}.gif"
    frames_folder_lst = sorted(filter(lambda x: x.name.isdigit(), seg_folder.iterdir()), key=lambda x: int(x.name))
    common_folder = seg_folder / "common"
    calibration_file = common_folder / "calibration.json"
    calibration = edict(json.load(calibration_file.open()))
    camera_params = calibration.cameraParams
    intrinsic_matrix = np.array(camera_params.IntrinsicMatrix).T
    lidar_to_cam_t = np.array(calibration.lidarToCam.T).T
    radar_to_cam_t = np.array(calibration.radarToCam.T).T
    radar_to_lidar_t = np.array(calibration.radarToLidar.T).T
    radar_to_imu_t = np.array(calibration.radarToImu.T).T
    lidar_pcd = o3d.geometry.PointCloud()
    pcd_file = frames_folder_lst[0] / 'os2_32.pcd'

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcd_file.as_posix())
    raw_points = np.array(pcd.points)
    lidar_pcd.points = o3d.utility.Vector3dVector(raw_points)
    lidar_pcd.colors = pcd.colors
    radar_pcd = o3d.geometry.PointCloud()
    lidar_points_color = (0.5, 0.5, 0.5)
    radar_points_color = (1, 0., 0.)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=1920, height=1080)
    viewer.add_geometry(mesh_frame)
    viewer.add_geometry(lidar_pcd)
    viewer.add_geometry(radar_pcd)

    viewer.run()
    param = viewer.get_view_control().convert_to_pinhole_camera_parameters()

    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = False
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])
    min_dist = 1.0
    # Init axes.
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(1, 1, figsize=(1920 * px, 1080 * px))
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
    dot_size = 50
    image_lst = []

    for frames_folder in tqdm(frames_folder_lst):
        gt_file = frames_folder / 'gtruth_labels.json'
        gt = edict(json.load(gt_file.open()))
        radar_cuboids = gt.radar_cuboids
        bbox3d = np.array(radar_cuboids).reshape(-1, 9)
        radar_boxes = []
        for i in range(len(bbox3d)):
            radar_boxes.append(RadarBox(bbox3d[i]))
        bboxes = gt.bboxes
        radar_csv = pandas.read_csv(frames_folder / 'zsignal0_zvue.csv')
        raw_radar_points = radar_csv.to_numpy()[:, :3]
        radar_points = raw_radar_points.T
        cameras = [f"camera{idx}.png" for idx in range(4)]
        camera0_file = frames_folder / cameras[0]
        img0 = cv2.imread(camera0_file.as_posix(), -1)
        pcd_file = frames_folder / 'os2_32.pcd'

        if undistort:
            img0_distorted = cv2.undistort(img0, intrinsic_matrix, (
                *camera_params.RadialDistortion[:2], *camera_params.TangentialDistortion,
                camera_params.RadialDistortion[2]))
            img0 = img0_distorted
        pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcd_file.as_posix())
        raw_points = np.array(pcd.points)
        points = raw_points.T
        lidar_pcd.points = o3d.utility.Vector3dVector(raw_points)
        # lidar_points_colors = np.tile(np.array(lidar_points_color), (raw_points.shape[0], 1))
        # lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_points_colors)
        lidar_pcd.colors = pcd.colors

        # radar to lidar
        radar_pcd.points = o3d.utility.Vector3dVector(transform_points(raw_radar_points.T, radar_to_lidar_t).T)
        radar_points_colors = np.tile(np.array(radar_points_color), (raw_radar_points.shape[0], 1))
        radar_pcd.colors = o3d.utility.Vector3dVector(radar_points_colors)

        lidar2cam_img = points_to_img(points, img0, intrinsic_matrix, fig, ax, min_dist=min_dist, dot_size=dot_size,
                                      transformation_matrix=lidar_to_cam_t)
        radar2cam_img = points_to_img(radar_points, img0, intrinsic_matrix, fig, ax, min_dist=min_dist,
                                      dot_size=dot_size, boxes=radar_boxes, transformation_matrix=radar_to_cam_t)
        draw_boxes(lidar2cam_img, bboxes)
        draw_boxes(radar2cam_img, bboxes)


        viewer.clear_geometries()
        viewer.add_geometry(lidar_pcd)
        viewer.add_geometry(radar_pcd)
        draw_boxes_in_points(viewer, radar_boxes, transformation_matrix=radar_to_lidar_t)
        ctr = viewer.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

        viewer.poll_events()
        viewer.update_renderer()
        img3d = viewer.capture_screen_float_buffer(False)
        img3d = (np.asarray(img3d)[:, :, ::-1] * 255).astype('uint8')

        img = np.zeros((1080 * 3, 1920, 3), dtype=np.uint8)
        img[0:1080, :, :] = lidar2cam_img
        img[1080:1080 * 2, :, :] = radar2cam_img
        img[1080 * 2:, :, :] = img3d
        image_lst.append(cv2.resize(img, (1920 // 2, 1080 * 3 // 2))[:, :, ::-1])
        cv2.imshow("img", img)
        # cv2.waitKey(0)

    viewer.destroy_window()
    imageio.mimsave((save_dir / save_name).as_posix(), image_lst, fps=10)


if __name__ == '__main__':
    fire.Fire(vis)
