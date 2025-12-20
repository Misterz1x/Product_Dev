This is a project for the Product Development, Testing and Approval course. 

The project uses the YOLO pose model to analyze a persons gait and keypoints for knee and hip range of motion. We will compare the keypoint detection to the OpenCap 3D camera system to validate if the out of the box pose-model gives us comaprable results.

The project uses uv as a package manager.

To install uv put the following command into the Windows PowerShell: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
After installation, you will see a path such as $env:Path = ‘C:\Users\...’.
Copy this command and execute it directly in PowerShell to make uv globally available.
Important: Select the line for PowerShell and copy only up to the bracket; you can omit (PowerShell).

After cloning the repository and installing the uv manager, using the command 'uv sync', the project will be initialized and the needed packages will be installed.

You can now either run the script using 'uv run script.py' or activate the virtual environment using '.venv\Scripts\activate' and than run the script.

To run the script successfully you need a .mot file and a video file, preferably .mp4. After running the 'video_analyzer.py' you will get asked to choose the 2 file ypu want to compare to each other. Additionally you can choos which body side you want to analyze. 
It is best to use the side that is best seen in the video. Afterwards you will get a plot with 4 subplots that show the knee- and hip angle calculated with the pose model from the video and the angles calculated by OpenCap. In the first subplot you can use your curser
to select as many gait cycles as you want to analyze. After closing the plot you will get a similar new plot, showing the chosen cycles and the mean value calculated from them. After closing that one aswell, the script will generate a pdf report with the most important 
data and some comparisons of the performance of both systems. The plots, along side more data for future analyzations and the report will be saved locally.


