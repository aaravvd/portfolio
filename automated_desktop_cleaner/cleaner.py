import shutil
import os
import time
import sys

def create_directories():
    file_dict = {}  
    user_file = input("Enter your username on your computer. This will be used in any file directory (e.g., aaravdixit): ")
    desktop_dir = f'/Users/{user_file}/Desktop'
    
    more_files = True
    while more_files:
        file_name = input("What do you want the folder to be called? Enter STOP for no more folders: ")
        if file_name.upper() == "STOP":
            more_files = False
        else:
            even_more = True
            counter = 0
            while even_more:
                sub_dir = input("Do you want to create a subfolder inside the original directory (e.g., Media -> Text)? If so, type Y/N: ")
                if sub_dir.upper() == "Y":
                    file_dir = input("What type of file do you want to make a folder/directory for (e.g., png, pdf, etc): ")
                    sub_name = input("What should the subfolder name be called? ")
                    new_file = {f'.{file_dir}': f'{desktop_dir}/{file_name}/{sub_name}'}
                    file_dict.update(new_file)
                    counter += 1
                elif sub_dir.upper() != "Y" and counter == 0:
                    even_more = False
                    file_dir = input("What type of file do you want to make a folder/directory for (e.g., png, pdf, etc): ")
                    new_file = {f'.{file_dir}': f'{desktop_dir}/{file_name}'}
                    file_dict.update(new_file)
                else:
                    break
    
    for directory in file_dict.values():
        try:
            os.makedirs(directory, exist_ok=True)  
            print(f"Directory created: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
    
    return file_dict


def move_files_to_directories(desktop_dir, file_dict):
    operations_done = {}
    while True:
        try:
            files_on_desk = os.listdir(desktop_dir)

            for file in files_on_desk:
                file_path = os.path.join(desktop_dir, file)
                moved = False 
        
                for ext, dir_path in file_dict.items():
                    if file.endswith(ext):
                        try:
                            shutil.move(file_path, dir_path)
                            operations_done[file] = dir_path
                            moved = True
                            break 
                        except Exception as e:
                            operations_done[f'Error with {file}'] = str(e)


                if not moved:
                    misc_dir = file_dict.get('.misc')
                    if misc_dir:
                        try:
                            shutil.move(file_path, misc_dir)
                            operations_done[file] = misc_dir
                        except Exception as e:
                            operations_done[f'Error with {file}'] = str(e)

            remaining_files = os.listdir(desktop_dir)
            print(f"Remaining files on desktop: {remaining_files}")
            

            time.sleep(10)
        except KeyboardInterrupt:
            print(operations_done)
            print("Exiting the program, run again to have the desktop cleaner running again!")
            break
            sys.exit()


file_dict = create_directories()
move_files_to_directories(f'/Users/{input("Enter your username: ")}/Desktop', file_dict)
