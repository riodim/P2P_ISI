# Project Setup Guide

This guide provides the necessary steps to set up and run your project using Docker and Visual Studio Code.

## Prerequisites

Make sure the following software is installed on your machine:

1. **Docker Desktop**: Ensure that Docker Desktop is running on your system.
2. **Visual Studio Code**: You will need VS Code to work with the project.
3. **Dev Containers Extension**: Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Build and Start the Docker Container

1. Open your terminal or command prompt.
2. Navigate to the project directory where your `docker-compose.yml` file is located.
3. Run the following command to build and start the Docker container:

   ```bash
   docker-compose up --build
Once the build process is complete, Docker will start the container. You can monitor the container's status via Docker Desktop.

### 2. Attach VS Code to the Running Container

After the container is up and running, you can attach Visual Studio Code to the container for development.

1. Open **Visual Studio Code**.
2. Click on the **Remote Explorer** icon in the Activity Bar on the left side of the window. If you don't see this icon, make sure the Dev Containers extension is installed and enabled.
3. In the Remote Explorer, look for your running container under the "Containers" section.
4. Click the container name to open a context menu, and then select **Attach to Container**.

   Alternatively, you can use the Command Palette:
   
   1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette.
   2. Type `Dev Containers: Attach to Running Container...` and select it from the list.
   3. Choose your container from the list of available containers.

5. VS Code will now reload and attach to the running container, allowing you to work directly within the containerized environment.

### 3. Running Files Inside the Container

Once attached, you can open and run your code within the containerized environment:

1. Use the VS Code **Explorer** to browse files inside the container.
2. Open the terminal inside VS Code (using `Ctrl+\`` or `Cmd+\``) and run the following command to execute your main Python file:

   ```bash
   python main.py

### Debugging Instructions

To debug your Python code within the containerized environment, follow these steps:

1. Open the file you want to debug in Visual Studio Code.
2. Insert the following line at the point in your code where you want to set a breakpoint:

   ```python
   import pdb; pdb.set_trace()
    
When the execution reaches the pdb.set_trace() line, the program will pause, and you will enter the interactive debugging mode in the terminal.

3. **In the debugging mode, you can**:
    - **Inspect variables**: Simply type the variable name and press Enter.
    - **Step through the code**: Use commands like `n` (next line), `s` (step into function), `c` (continue execution), and `q` (quit debugger).

4. **Once you've finished debugging**:
    - Remove or comment out the `import pdb; pdb.set_trace()` line to continue running the program without interruptions.

### Useful `pdb` Commands

- **n (next)**: Continue execution to the next line within the same function.
- **s (step)**: Step into a function call.
- **c (continue)**: Continue execution until the next breakpoint.
- **q (quit)**: Quit the debugger and stop the program.
- **l (list)**: Display the source code around the current line.
- **p (print)**: Print the value of a variable, e.g., `p variable_name`.
- **pp (pretty-print)**: Pretty-print the value of a variable, useful for inspecting complex data structures.
- **b (breakpoint)**: Set a breakpoint at a specified line, e.g., `b 42` sets a breakpoint at line 42.
- **cl (clear)**: Clear a breakpoint at a specified line, e.g., `cl 42`.
- **bt (backtrace)**: Show the stack trace of the current position.
- **u (up)**: Move up one level in the stack trace.
- **d (down)**: Move down one level in the stack trace.
- **! (run code)**: Execute a Python command, e.g., `!variable_name = 5`.
- **w (where)**: Display the current position in the stack trace.