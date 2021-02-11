import setuptools
import shutil
import os
import glob

with open("../../README.md", "r") as fh:
    long_description = fh.read()

def remove_existing_whl_files(folder_path):
    existing_whl_files = glob.glob(os.path.join(folder_path, '*.whl'))
    for file_path in existing_whl_files:
        try:
            os.remove(file_path)
        except OSError as e:
            print("[ERROR] Error: %s : %s" % (file_path, e.strerror))

package_name = 'bb-ai-fnv-inference'
package_version = "0.0.1"
src_folder = 'dist'
dest_folder = '../deployment'
fnv_whl_file = '_'.join(package_name.split('-'))+"-"+package_version+"-py3-none-any.whl"

remove_existing_whl_files(src_folder)

setuptools.setup(
    name=package_name,
    version=package_version,
    author="Santosh Kumar Waddi",
    author_email="santosh.waddi@bigbasket.com",
    description="Fruits and Vegetable prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

remove_existing_whl_files(dest_folder)
shutil.copyfile(os.path.join(src_folder, fnv_whl_file), os.path.join(dest_folder, fnv_whl_file))