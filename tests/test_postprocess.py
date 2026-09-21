from src.postprocess import entities_to_json

def test_entities_to_json():
    assert entities_to_json(["ABC", "₹500"], ["B-COMPANY", "B-TOTAL"]) == {"COMPANY": "ABC", "TOTAL": "₹500"}
