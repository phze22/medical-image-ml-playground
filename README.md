Installation

Create and activate a virtual environment:

python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

Install the project:

python -m pip install --upgrade pip
pip install -e .
Configuration

Copy .env.example to .env and configure the Databricks connection and endpoint.

Never commit credentials or patient data.

Usage

Run inference on a DICOM image:

medical-image-ml-playground data/example.dcm
Pipeline
Load a DICOM image.
Convert its pixels to a PyTorch tensor.
Format the tensor for the Databricks endpoint.
Extract and print the returned text label.
