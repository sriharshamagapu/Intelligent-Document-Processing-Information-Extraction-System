import re

def normalize_entity(text: str):
    return re.sub(r"\\s+", " ", text).strip()

def entities_to_json(words, labels):
    grouped = {}
    current = None
    buffer = []
    for word, label in zip(words, labels):
        label = label.replace("B-", "").replace("I-", "")
        if label == "O":
            if current:
                grouped[current] = normalize_entity(" ".join(buffer))
                current, buffer = None, []
            continue
        if label != current:
            if current:
                grouped[current] = normalize_entity(" ".join(buffer))
            current, buffer = label, [word]
        else:
            buffer.append(word)
    if current:
        grouped[current] = normalize_entity(" ".join(buffer))
    return grouped
