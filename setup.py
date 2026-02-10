from setuptools import setup, find_packages

setup(
    name="lora-slm-finetuning",
    version="0.1.0",
    description="LoRA Fine-tuning for Small Language Models",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "datasets>=2.14.0",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0"
    ],
    python_requires=">=3.8",
)
