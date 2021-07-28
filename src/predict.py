import joblib
import torch

import config
import dataset
import model

from termcolor import colored

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield " ".join(lst[i:i + n])


def predict(test_sentence):
    meta_data = joblib.load(config.META_DATA_FILE)
    label_enc = meta_data["label_enc"]

    tokenized_sentence = config.TOKENIZER.encode(test_sentence)

    test_sentence = test_sentence.split()

    test_dataset = dataset.EntityDataset(
        text_corpus = [test_sentence],
        tag_corpus = [[0] * len(test_sentence)]
    )

    device = "cpu"

    num_tags = len(label_enc.classes_)

    entity_model = model.EntityModel(num_tags).to(device)
    entity_model.load_state_dict(torch.load(config.MODEL_FILE, map_location=torch.device("cpu")))

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        predictions, _ = entity_model(**data)
        predictions = predictions.cpu().detach().numpy()

    label_indices = label_enc.inverse_transform(
        predictions.argmax(2).reshape(-1)
    )

    tokens = config.TOKENIZER.convert_ids_to_tokens(test_dataset[0]["input_ids"].to("cpu").numpy())

    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(label_idx)
            new_tokens.append(token)

    return (new_tokens, new_labels)

if __name__ == "__main__":

    sentence = "Admission Date : 2010-05-17 Discharge Date : 2010-05-29 Date of Birth : 1923-11-09 Sex : F Service : HISTORY OF PRESENT ILLNESS : This is an 86-year-old female with a past medical history of diabetes and hypertension who presented to the Emergency Department with a two day history of shortness of breath and easy fatigability . She also noted increased dyspnea on exertion . Prior to this patient could walk upstairs but now cannot . She denies chest pain , orthopnea , paroxysmal nocturnal dyspnea . However , the patient has noted some abdominal pain , intermittent times a couple of days , none on the day of admission . Denied nausea , vomiting , diaphoresis , bowel movement changes . She has noted increased urinary frequency . Denies fever or chills . Of note , she has had recent medication changes which included discontinuing Diovan and starting terazosin . PAST MEDICAL HISTORY : 1. Diabetes mellitus . 2. Hypertension . 3. Osteoarthritis . 4. _______________ ALLERGIES : No known drug allergies . HOME MEDICATIONS : 1. Aspirin 325 mg p.o. q. day . 2. Norvasc 10 mg p.o. q. day . 3. Isosorbide mononitrate 30 q. day . 4. Terazosin 1 mg p.o. q. day . SOCIAL HISTORY : The patient lives alone . Has a remote tobacco history . No ethanol . FAMILY HISTORY : Positive for diabetes . No coronary artery disease . PHYSICAL EXAMINATION : On admission , blood pressure 151/61 , heart rate 75 , SaO2 93% on room air . In general , patient in no acute distress . Speaks in full sentences . Neck : Jugular venous pressure notable at the mid neck upright about 80 degree angles . Cardiovascular regular rate and rhythm with a 1-2/6 holosystolic murmur . Pulmonary : Crackles bilaterally one-third to one-half the way up . Abdomen : Bowel sounds positive , soft , non-tender , non-distended . Extremities : 1+ lower extremity edema . LABORATORIES ON ADMISSION : Included normal CBC , BUN 47 , creatinine 2.4 up from baseline of about 1.5 . Troponin 4.7 , CK 101 , MB 3 . RADIOLOGY : Chest x-ray showed patient was in congestive heart failure . ELECTROCARDIOGRAM : Showed normal sinus rhythm at 107 beats per minute with left axis deviation and poor R-wave progression . However , no acute changes were apparent . HOSPITAL COURSE : The patient was admitted to the Cardiac Medicine Service and treated for presumed diastolic and systolic dysfunction . Echocardiogram was obtained which showed moderately depressed left ventricular systolic function as well as hypokinesis of the lower half of septum and apex . Also of note was the distal lateral wall hypokinesis . The wall motion abnormalities were noted to be new . It was believed that a troponin on admission in addition to the wall motion abnormalities she underwent a non-Q wave myocardial infarction prior to resulted in her her current cardiac failure . Throughout hospital course patient patient 's troponin trended down to less than 0.3 . Heart Failure Service was involved . She was continued with aggressive diuresis . She was started on _________ with excellent diuresis , however , her her renal functioning worsening . A cardiac catheterization was deferred until the renal issue could be resolved . However , her her creatinine continued to increase . Diuresis was halted and without improvement in creatinine . Renal was consulted . A renal ultrasound was obtained . It showed a right kidney size of 6.3 cm and a left kidney size of 8.4 cm . Given her her hypertension which was very difficult to control , it was felt that she had renal artery stenosis and thus she underwent MRA of the kidney which showed severe right renal artery stenosis at its origin . There was also moderate to severe focal stenosis of the left renal artery approximately 1.3 cm from its origin . Dr. ______________ consulted on the case . She was transferred to the unit overnight to assess volume which was noted to be optimal . On 05-25 she underwent catheterization and subsequent stenting of the left renal artery . Due to dye load required to assess the coronary disease were not visualized . After the procedure the patient did well . However , her her creatinine has worsened up to 4.2 . However , her her urine output has improved . She has not required hemodialysis at this time . She will need close follow up of her renal functioning . The patient is discharged to an extended discharge facility . DISCHARGE DIAGNOSES : 1. Non-Q wave myocardial infarction . 2. Systolic and diastolic congestive heart failure . 3. Bilateral renal artery stenosis status post stent left renal artery . 4. Acute renal failure . 5. Hypertension . FOLLOW UP : She is to follow up with her her primary care physician in one week . She is to follow up Heart Failure and Cardiology in one week as well . In addition , patient is to have her own appointment with Podiatry . DISCHARGE MEDICATIONS : 1. Aspirin 325 mg p.o. q. day . 2. Norvasc 10 mg p.o. q. day . 3. Atorvastatin 10 mg p.o. q. day . 4. Protonix 20 mg p.o. q. day . 5. Multivitamin one p.o. q. day . 6. Carvedilol 3.125 p.o. b.i.d. 7. ___________ dinitrate 20 mg p.o. t.i.d. 8. Erythropoietin 10,000 units two times a week Tuesday and Saturday . 9. Calcium acetate Tums one p.o. t.i.d. with meals . 10. Tylenol 325 mg p.o. q. 4-6 h. p.r.n. 11. Iron sulfate 325 mg p.o. q. day . 12. Colace 100 mg p.o. b.i.d. 13. Hydralazine 10 mg p.o. q. 6h. 14. Plavix 75 mg p.o. q. day times 30 days . Kathy Tillis , M.D. 85-987 Dictated By : John V.E. Jameson , M.D. MEDQUIST36 D : 2010-05-27 14:29 T : 2010-05-27 14:10 JOB #: 80987 Signed electronically by : DR. Kathy Tillis on : MON 2010-06-07 4:00 PM ( End of Report )"
    sentence_tokens = sentence.split()

    tokens, labels = [], []
    for sentence_batch in chunks(sentence_tokens, 50):
        batch_tokens, batch_labels = predict(sentence_batch)

        for token, label in zip(batch_tokens, batch_labels):
            if not token in ["[PAD]", "[CLS]", "[SEP]"]:
                tokens.append(token)
                labels.append(label)

    print("\n----------Your text with the highlighted entities----------\n")

    for ind in range(len(tokens)):
        if labels[ind]!="O":
            label_type = labels[ind].split("-")[1]

            if label_type == "problem":
                tokens[ind] = colored(tokens[ind], "red")

            elif label_type == "treatment":
                tokens[ind] = colored(tokens[ind], "blue")

            elif label_type == "test":
                tokens[ind] = colored(tokens[ind], "green")

            elif label_type == "person":
                tokens[ind] = colored(tokens[ind], "yellow")

            else:
                tokens[ind] = colored(tokens[ind], "magenta")

    print(" ".join(tokens))

    print("\n-----------------------------------------------------------\n")