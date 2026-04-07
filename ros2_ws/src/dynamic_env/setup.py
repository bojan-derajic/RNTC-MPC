import os
import glob
from setuptools import find_packages, setup

package_name = "dynamic_env"

# Collect all asset files preserving subdirectory structure under share/
def collect_assets(src_base, dst_base, patterns):
    """Return a list of (install_dir, [src_files]) tuples for data_files."""
    entries = []
    for pattern in patterns:
        for src_file in glob.glob(os.path.join(src_base, pattern)):
            rel_dir = os.path.relpath(os.path.dirname(src_file), src_base)
            dst_dir = os.path.join(dst_base, rel_dir)
            entries.append((dst_dir, [src_file]))
    return entries


assets_dir = "assets"
share_assets = os.path.join("share", package_name, "assets")

asset_files = collect_assets(
    assets_dir,
    share_assets,
    [
        "scenarios/*.json",
        "xacro_models/*.xacro",
        "sdf_models/*.sdf",
        "urdf_models/*.urdf",
    ],
)

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *asset_files,
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bojan Derajic",
    maintainer_email="derajicbojan@gmail.com",
    description="Dynamic obstacle environment manager for RNTC-MPC evaluation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dynamic_env = dynamic_env.dynamic_env:main",
        ],
    },
)
