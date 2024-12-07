# All Things Machine Learning
My journey of studying Machine Learning, all in one Repo.

## Notes:
1. Each projects have their own folders (e.g. Handwritten Number Recognition using CNN has its own folder, namely `handwrittenNumber_CNN`).
2. Most (if not all) of the code are tested on Apple Silicon Macs. These codes are not tested on neither Nvidia or AMD GPU, so keep that in mind.

## Setting a Local Environment
To set up a local virtual environment for your machine learning projects, follow these steps:
1. **Install Python**: Ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/).
2. **Install `virtualenv`**: Open your terminal and run the following command to install `virtualenv`:
    ```sh
    pip install virtualenv
    ```
3. **Create a Virtual Environment**: Navigate to your project directory and create a virtual environment by running:
    ```sh
    virtualenv venv
    ```
    This will create a directory named `venv` containing the virtual environment.
4. **Activate the Virtual Environment**:
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
5. **Install Dependencies**: With the virtual environment activated, install the necessary dependencies using:
    ```sh
    pip install -r requirements.txt
    ```
    Ensure you have a `requirements.txt` file in your project directory listing all the required packages.
6. **Deactivate the Virtual Environment**: Once you are done working, you can deactivate the virtual environment by running:
    ```sh
    deactivate
    ```
