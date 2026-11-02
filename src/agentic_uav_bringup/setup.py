from glob import glob

from setuptools import setup

package_name = "agentic_uav_bringup"

setup(
    name=package_name,
    version="0.2.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Muhammed Nihal",
    maintainer_email="nihvp@outlook.com",
    description="Agentic UAV bringup",
    license="Apache-2.0",
)
