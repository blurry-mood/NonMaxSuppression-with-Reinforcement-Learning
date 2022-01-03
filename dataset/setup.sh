# unzip data and visualize progress using `pv`
unzip -o wider_face_split.zip
unzip -o WIDER_test.zip
unzip -o WIDER_val.zip
unzip -o WIDER_train.zip

# minor cleanup
rm -rf WIDER_train.zip WIDER_val.zip WIDER_test.zip wider_face_split.zip

# copy labels next to images
python3 setup.py

# another minor cleanup
rm -rf wider_face_split