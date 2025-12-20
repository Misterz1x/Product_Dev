This is a project for the Product Development, Testing and Approval course. 

The project uses the YOLO pose model to analyze a persons gait and keypoints for knee and hip range of motion. We will compare the keypoint detection to the OpenCap 3D camera system to validate if the out of the box pose-model gives us comaprable results.

The project uses uv as a package manager.

To install uv put the following command into the Windows PowerShell: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
After installation, you will see a path such as $env:Path = ‘C:\Users\...’.
Copy this command and execute it directly in PowerShell to make uv globally available.
Important: Select the line for PowerShell and copy only up to the bracket; you can omit (PowerShell).

After cloning the repository and installing the uv manager, using the command 'uv sync', the project will be initialized and the needed packages will be installed.

You can now either run the script using 'uv run script.py' or activate the virtual environment using '.venv\Scripts\activate' and than run the script.


