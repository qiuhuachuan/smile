cd ..
python convert_to_train_json.py
cd src/
python convert_train2jsonl.py
python tokenize_dataset_rows.py