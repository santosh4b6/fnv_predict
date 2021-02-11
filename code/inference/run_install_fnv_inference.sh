source activate bb_fnv
python setup.py bdist_wheel
echo y | pip uninstall bb-ai-fnv-inference y
pip install dist/*.whl