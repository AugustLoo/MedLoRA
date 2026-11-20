import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_slake_no as A  # noqa: E402


def c(q, a, ct="Abnormality"):
    return A.classify({"question": q, "answer": a, "content_type": ct})


def test_open_answers_are_not_yes_no():
    assert c("What diseases are included in the picture?", "Pneumonia") == "not_yn"


def test_non_finding_questions_keep_their_own_buckets():
    assert c("Does the picture contain liver?", "No", "Organ") == "organ"
    assert c("Is this a coronal plane?", "No", "Plane") == "other"


def test_normal_worded_question_no_means_a_finding_is_present():
    assert c("Is the lung healthy?", "No") == "pos_global"
    assert c("Does this image look normal?", "Yes") == "neg_global"


def test_abnormal_worded_question_no_means_nothing_found():
    assert c("Does the kidney look abnormal?", "No") == "neg_global"
    assert c("Are there abnormalities in this image?", "yes") == "pos_global"
    assert c("Is the lung unhealthy?", "No") == "neg_global"


def test_named_disease_question_is_specific():
    q = "Can pneumothorax be observed on the lower left lung in this picture?"
    assert c(q, "No") == "neg_specific"
    assert c(q, "Yes") == "pos_specific"
    assert c("Does the patient have brain tumor?", "Yes") == "pos_specific"


def test_summary_counts_negative_findings_among_all_samples():
    rows = [
        {"question": "Is the lung healthy?", "answer": "No", "content_type": "Abnormality", "modality": "X-Ray", "img_id": 1},
        {"question": "Can nodule be observed on the lower right lung in this picture?", "answer": "No",
         "content_type": "Abnormality", "modality": "X-Ray", "img_id": 1},
        {"question": "Does the picture contain liver?", "answer": "No", "content_type": "Organ", "modality": "CT", "img_id": 2},
        {"question": "What modality is used to take this image?", "answer": "CT", "content_type": "Modality", "modality": "CT", "img_id": 2},
    ]
    s = A.summarize(rows)
    assert s["samples"] == 4 and s["answer_no"] == 3
    assert s["no_by_kind"] == {"pos_global": 1, "neg_specific": 1, "organ": 1}
    assert s["negative_findings"] == 1          # 只有「没有结节」算「没病」
    assert s["xray"]["neg_specific"] == 1
    assert s["xray_images"] == {"abnormal": 1}


def test_named_disease_no_is_flagged_when_same_image_lists_that_disease():
    rows = [
        {"question": "Can mass be observed on the lower right lung in this picture?", "answer": "No",
         "content_type": "Abnormality", "modality": "X-Ray", "img_id": 7},
        {"question": "What diseases are included in the picture?", "answer": "Pulmonary Mass",
         "content_type": "Abnormality", "modality": "X-Ray", "img_id": 7},
        {"question": "Can nodule be observed on the lower right lung in this picture?", "answer": "No",
         "content_type": "Abnormality", "modality": "X-Ray", "img_id": 7},
    ]
    assert A.summarize(rows)["neg_specific_but_disease_elsewhere_in_image"] == 1
