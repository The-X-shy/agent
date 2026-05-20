from setuptools import find_packages, setup


setup(
    name="optiresearch-agent",
    version="0.1.0",
    description="OptiResearch Agent MVP.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2",
        "fastapi",
        "uvicorn",
        "pyyaml",
        "numpy",
    ],
    extras_require={"dev": ["pytest"], "plot": ["matplotlib"]},
    entry_points={"console_scripts": ["optiresearch=optiresearch.cli:main"]},
)
