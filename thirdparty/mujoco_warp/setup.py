from setuptools import setup

setup(
    name="mujoco_warp",
    version="3.6.0",
    packages=[
        "mujoco_warp",
        "mujoco_warp._src",
        "mujoco_warp.test_data",
    ],
    package_dir={
        "mujoco_warp": ".",
        "mujoco_warp._src": "_src",
        "mujoco_warp.test_data": "test_data",
    },
    include_package_data=True,
    package_data={
        "mujoco_warp.test_data": ["**/*"],
    },
)
