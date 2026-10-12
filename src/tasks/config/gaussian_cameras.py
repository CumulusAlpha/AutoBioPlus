from src.tasks.config.gaussian_scenes import GaussianSceneCfg
from src.utils.camera_distortion import DistortedGaussianSensorMjlabCfg


GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = ("terrain",)
GAUSSIAN_FOREGROUND_EXCLUDE_BODY_PREFIXES = ("table",)


def make_gaussian_camera(
    *,
    name: str,
    background_scene: GaussianSceneCfg,
    background_frame_body_name: str | None = None,
    camera_name: str | None = None,
    parent_body: str | None = None,
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    fovy: float | None = None,
    intrinsics: tuple[float, float, float, float] | None = None,
    distortion_model: str | None = None,
    distortion: tuple[float, float, float, float, float] | None = None,
    width: int = 640,
    height: int = 480,
    foreground_exclude_geom_names: tuple[str, ...] = GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    foreground_exclude_body_prefixes: tuple[
        str, ...
    ] = GAUSSIAN_FOREGROUND_EXCLUDE_BODY_PREFIXES,
) -> DistortedGaussianSensorMjlabCfg:
    return DistortedGaussianSensorMjlabCfg(
        name=name,
        camera_name=camera_name,
        parent_body=parent_body,
        pos=pos,
        quat=quat,
        fovy=fovy,
        intrinsics=intrinsics,
        distortion_model=distortion_model,
        distortion=distortion,
        width=width,
        height=height,
        background_ply_path=background_scene.ply_path,
        background_frame_body_name=background_frame_body_name,
        background_position=background_scene.position,
        background_quat=background_scene.quat,
        background_scale=background_scene.scale,
        render_mode="hybrid",
        foreground_mode="all",
        foreground_exclude_geom_names=foreground_exclude_geom_names,
        foreground_exclude_body_prefixes=foreground_exclude_body_prefixes,
    )


def make_gaussian_camera_triplet(
    *,
    background_scene: GaussianSceneCfg,
    background_frame_body_name: str | None = None,
    front_pos: tuple[float, float, float],
    front_quat: tuple[float, float, float, float],
    left_pos: tuple[float, float, float],
    left_quat: tuple[float, float, float, float],
    scene_parent_body: str | None = None,
    wrist_camera_name: str | None = "robot/wrist_cam",
    wrist_parent_body: str | None = None,
    wrist_pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wrist_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    wrist_fovy: float | None = 58.0,
    wrist_intrinsics: tuple[float, float, float, float] | None = None,
    wrist_distortion_model: str | None = None,
    wrist_distortion: tuple[float, float, float, float, float] | None = None,
    wrist_width: int = 640,
    wrist_height: int = 480,
    front_fovy: float = 58.0,
    front_intrinsics: tuple[float, float, float, float] | None = None,
    front_width: int = 640,
    front_height: int = 480,
    left_fovy: float = 58.0,
    left_width: int = 640,
    left_height: int = 480,
    foreground_exclude_geom_names: tuple[str, ...] = GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    foreground_exclude_body_prefixes: tuple[
        str, ...
    ] = GAUSSIAN_FOREGROUND_EXCLUDE_BODY_PREFIXES,
) -> tuple[
    DistortedGaussianSensorMjlabCfg,
    DistortedGaussianSensorMjlabCfg,
    DistortedGaussianSensorMjlabCfg,
]:
    gs_wrist_cam = make_gaussian_camera(
        name="gs_wrist_cam_rgb",
        camera_name=wrist_camera_name,
        parent_body=wrist_parent_body,
        pos=wrist_pos,
        quat=wrist_quat,
        fovy=wrist_fovy,
        intrinsics=wrist_intrinsics,
        distortion_model=wrist_distortion_model,
        distortion=wrist_distortion,
        width=wrist_width,
        height=wrist_height,
        background_scene=background_scene,
        background_frame_body_name=background_frame_body_name,
        foreground_exclude_geom_names=foreground_exclude_geom_names,
        foreground_exclude_body_prefixes=foreground_exclude_body_prefixes,
    )
    gs_front_cam = make_gaussian_camera(
        name="gs_front_cam_rgb",
        parent_body=scene_parent_body,
        pos=front_pos,
        quat=front_quat,
        fovy=front_fovy,
        intrinsics=front_intrinsics,
        width=front_width,
        height=front_height,
        background_scene=background_scene,
        background_frame_body_name=background_frame_body_name,
        foreground_exclude_geom_names=foreground_exclude_geom_names,
        foreground_exclude_body_prefixes=foreground_exclude_body_prefixes,
    )
    gs_left_cam = make_gaussian_camera(
        name="gs_left_cam_rgb",
        parent_body=scene_parent_body,
        pos=left_pos,
        quat=left_quat,
        fovy=left_fovy,
        width=left_width,
        height=left_height,
        background_scene=background_scene,
        background_frame_body_name=background_frame_body_name,
        foreground_exclude_geom_names=foreground_exclude_geom_names,
        foreground_exclude_body_prefixes=foreground_exclude_body_prefixes,
    )
    return gs_wrist_cam, gs_front_cam, gs_left_cam
