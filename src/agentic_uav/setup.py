from setuptools import setup, find_packages

package_name = 'agentic_uav'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if False else []),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Antigravity Autonomous Systems',
    maintainer_email='dev@agentic-ai-uav.org',
    description='5-layer Agentic UAV system in ROS 2 Humble',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = agentic_uav.perception.perception_node:main',
            'reasoning_node = agentic_uav.reasoning.reasoning_node:main',
            'action_node = agentic_uav.action.action_node:main',
            'mcp_integration_node = agentic_uav.integration.mcp_node:main',
            'learning_node = agentic_uav.learning.learning_node:main',
        ],
    },
)
