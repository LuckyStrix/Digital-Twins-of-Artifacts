import subprocess
import time
from datetime import datetime
import os
import serial
now = datetime.now()
ser = serial.Serial('COM3', baudrate=115200, timeout=2.5)
time.sleep(2)

def message_arduino(n, e, s, w, g, b, step, dir):
    msg = ""
    for i in (n, e, s, w, g, b, step, dir):
        msg += str(int(i))
    ser.write(msg.encode())
    
    while 1:
        response = ser.read(1)
        if response == b'e':
            break

def capture_image(filename):
    cr2 = f"{filename}.cr2"
    # -defterm -no-start runs gphoto2 synchronously in this process instead of
    # spawning a detached mintty window. Without it, msys2_shell.cmd returns
    # immediately (the window runs in the background), so dcraw below fires
    # before the .cr2 has been downloaded — fine from an interactive terminal,
    # but it races and fails when launched from the GUI (run.py). We also let
    # gphoto2's output stream through so its errors are visible in the log.
    cmd = (rf'C:\msys64\msys2_shell.cmd -mingw64 -defterm -no-start -here '
           rf'-c "gphoto2 --capture-image-and-download --filename {cr2}"')
    subprocess.run(cmd, shell=True, cwd=img_dir)

    if not os.path.exists(os.path.join(img_dir, cr2)):
        print(f"WARNING: {cr2} was not created by gphoto2 — skipping dcraw. "
              "Check that the camera is connected and gphoto2 can reach it.")
        print(" ")
        return

    #subprocess.run(f"exiftool -Orientation=1 -n {filename}.cr2", cwd=img_dir, shell=True)
    subprocess.run(f"dcraw -T -6 -W -o 0 -q 0 -t 0 {cr2}", cwd=img_dir, shell=True)
    print(filename + " captured!")
    print(" ")

if __name__ == "__main__":
    folder_name = now.strftime("%d-%m-%y_%H-%M-%S")
    print("Folder name = " + folder_name)
    print(" ")
    # Write captures into the app's top-level data/ folder so the launcher
    # (run.py) can find them. This script lives in <app>/backend/capture/, so
    # the app root is three levels up. Originally a hard-coded OneDrive path.
    app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    img_dir = os.path.join(app_root, "data", folder_name)
    os.makedirs(img_dir, exist_ok=True)

    #MAIN PHOTOGRAPHING LOOP
    #CROSS IMAGES
    message_arduino(1, 1, 1, 1, 0, 1, 0, 1)
    capture_image("allLight")

    message_arduino(1, 0, 0, 0, 0, 1, 0, 1)
    capture_image("ncross")

    message_arduino(0, 1, 0, 0, 0, 1, 0, 1)
    capture_image("ecross")

    message_arduino(0, 0, 1, 0, 0, 1, 0, 1)
    capture_image("scross")

    message_arduino(0, 0, 0, 1, 0, 1, 0, 1)
    capture_image("wcross")

    #ROTATE
    message_arduino(0, 0, 0, 0, 0, 1, 1, 1)

    #CO IMAGES
    message_arduino(1, 0, 0, 0, 0, 1, 0, 0)
    capture_image("nco")

    message_arduino(0, 1, 0, 0, 0, 1, 0, 0)
    capture_image("eco")

    message_arduino(0, 0, 1, 0, 0, 1, 0, 0)
    capture_image("sco")

    message_arduino(0, 0, 0, 1, 0, 1, 0, 0)
    capture_image("wco")

    #ROTATE
    message_arduino(0, 0, 0, 0, 0, 1, 1, 0)
    print("Scanning Complete!")
    print(" ")

    #SORT IMAGES
    archive_dir = f"{img_dir}\\cr2Archive"
    subprocess.run(f"mkdir \"{archive_dir}\"", shell=True)
    subprocess.run(f"move \"{img_dir}\\*.cr2\" \"{archive_dir}\"", shell=True)

    #FINISH
    message_arduino(0, 0, 0, 0, 0, 0, 0, 1)