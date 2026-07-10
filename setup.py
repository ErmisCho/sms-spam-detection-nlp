from setuptools import find_packages, setup


setup(
    name="sms-spam-detection-nlp",
    version="0.1.0",
    description="End-to-end SMS spam detection NLP pipeline with semantic clustering.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)
