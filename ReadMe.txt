Introduction:

The whole project has two main parts: visualize and save date from the simulated hospital.
The further target is using the saved data to developing the machine learning algorithm.


1. setting up the gate way:
    there are two gate ways required: one for 9-pixel pods, one for 20x25 matrix

    for the 9-pixel pods, the gate way is built by MQTT and mosquitto.
    It is available on the github: https://github.com/LESA-RPI/ArpaE.9pixel

    for the matix, the gate way is build by C++ language. More complicated than 9-pixel one, but faster.
    It is available on the github: https://github.com/LESA-RPI/scr.tof_control

    After clone the two file, the matrix could directly used if the connection is good.
    For the 9-pixel one, be careful about the IP addresses, edit them to satisfy current condition.

    After setting up, the command to run these two gate way is as following:
    9-pixel in python: python subscriber.py
    20x25 Matrix in C++ : ./tof                  (the file is in the SCR folder)

2.  visualization:
    simply run the hospital map: python hospitalmap.py

    Since the paths of files might be different, please edit the heatmap function for satisfying current file structure.


3. saving data:
    simply run the hospital map: python data_save.py
    it will output four files: 3 test files and 1 video file
    there is a name rule in the excel file, please follow the rules when input the file name
