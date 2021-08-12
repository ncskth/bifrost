from setuptools import setup

setup(
    name="bifrost",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=["compynator", "torch", "norse"],
    scripts=["bin/bifrost"],
)
