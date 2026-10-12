import mujoco


def patch_viser_sdf_geom_rendering() -> None:
    """Render MuJoCo SDF geoms in viser via their backing mesh."""
    import mjlab.viewer.viser.conversions as conversions
    import mjlab.viewer.viser.scene as scene

    if getattr(scene, "_sdf_geom_patch_installed", False):
        return

    def merge_geoms_with_sdf(mj_model: mujoco.MjModel, geom_ids: list[int]):
        meshes = []
        for geom_id in geom_ids:
            geom_type = mj_model.geom_type[geom_id]

            if geom_type == mujoco.mjtGeom.mjGEOM_MESH or (
                geom_type == mujoco.mjtGeom.mjGEOM_SDF
                and 0 <= int(mj_model.geom_dataid[geom_id]) < mj_model.nmesh
            ):
                mesh = conversions.mujoco_mesh_to_trimesh(
                    mj_model, geom_id, verbose=False
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_SDF:
                continue
            else:
                mesh = conversions.create_primitive_mesh(mj_model, geom_id)

            pos = mj_model.geom_pos[geom_id]
            quat = mj_model.geom_quat[geom_id]
            transform = conversions.np.eye(4)
            transform[:3, :3] = conversions.vtf.SO3(quat).as_matrix()
            transform[:3, 3] = pos
            mesh.apply_transform(transform)
            meshes.append(mesh)

        if not meshes:
            mesh = conversions.trimesh.Trimesh(
                vertices=conversions.np.zeros((3, 3)),
                faces=conversions.np.array([[0, 1, 2]]),
                process=False,
            )
            mesh.visual = conversions.trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=conversions.np.zeros((3, 4), dtype=conversions.np.uint8),
            )
            return mesh
        if len(meshes) == 1:
            return meshes[0]
        return conversions.trimesh.util.concatenate(meshes)

    conversions.merge_geoms = merge_geoms_with_sdf
    scene.merge_geoms = merge_geoms_with_sdf
    scene._sdf_geom_patch_installed = True
