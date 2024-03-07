Developed a Named Entity Recognition (NER) algorithm pipeline for medical notes question/answering.
Project performed under my supervision by [Pushkaraj Palnitkar](https://github.com/push44) during the course of his internship at Endeavor Health Advanced Analytics Team.

EEH was merged with NorthShore University HealthSystem (NorthShore) in 2022 and created Endeavor Health.
# Clinical-Notes-NER
Named entity recognition on clinical notes

![medicalrecord](https://user-images.githubusercontent.com/61958160/126857163-d5fa33c0-a712-475a-8d35-78bdc71ea462.jpg)

### Instructions to run this project:
Copy https project url from github green code button.<br>
Open terminal:
```
$git clone <url>
$git lfs pull
$pip install virtualenv
$cd <project folder>
$virtualenv env
$source env/bin/activate (For windows: .\env\Scripts\activate)
$pip install -r requirements.txt
$python src/predict.py
```
Enter your clinical note in terminal: [Example clinical from MeDAL dataset](https://www.kaggle.com/xhlulu/medal-emnlp)
![input](https://user-images.githubusercontent.com/61958160/127630915-40d8545d-c8d7-4fcf-a66f-add6f7e4964f.png)

Out will be displayed:
![output](https://user-images.githubusercontent.com/61958160/127630955-0c3c0547-a5e2-4e6d-a19c-7368437b2415.png)
