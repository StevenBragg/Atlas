from setuptools import setup, find_packages

setup(
    name="self_organizing_av_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Learning System Developer",
    author_email="developer@example.com",
    description="A self-organizing audio-visual learning system with biologically inspired architecture",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/self_organizing_av_system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 