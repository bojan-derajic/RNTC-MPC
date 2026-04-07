import os
import glob
from setuptools import find_packages, setup

package_name = "mpc_local_planner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Pre-trained hypernetwork weights
        (
            os.path.join("share", package_name, "hypernet_weights"),
            glob.glob("hypernet_weights/*.pth"),
        ),
        # Directory for compiled CasADi solver shared libraries (.so files).
        # The directory must exist in the install tree so the node can write to it.
        (os.path.join("share", package_name, "assets", "casadi_compiled"), []),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bojan Derajic",
    maintainer_email="derajicbojan@gmail.com",
    description="MPC local planners (RNTC, NTC, DCBF, VO, SDF) for dynamic obstacle avoidance",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sdf_mpc  = mpc_local_planner.sdf_mpc:main",
            "dcbf_mpc = mpc_local_planner.dcbf_mpc:main",
            "vo_mpc   = mpc_local_planner.vo_mpc:main",
            "ntc_mpc  = mpc_local_planner.ntc_mpc:main",
            "rntc_mpc = mpc_local_planner.rntc_mpc:main",
        ],
    },
)
