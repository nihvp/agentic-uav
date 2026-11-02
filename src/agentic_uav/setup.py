from setuptools import find_packages, setup

package_name = "agentic_uav"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Muhammed Nihal",
    maintainer_email="nihvp@outlook.com",
    description="Five-layer agentic UAV nodes",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "action_node = agentic_uav.action.action_node:main",
            "perception_node = agentic_uav.perception.perception_node:main",
            "reasoning_node = agentic_uav.reasoning.reasoning_node:main",
            "learning_node = agentic_uav.learning.learning_node:main",
            "mcp_server = agentic_uav.integration.mcp_server:main",
        ],
    },
)
